#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/runtime.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/threadpool.h>
#include <cpuinfo.h>
#endif

DEFINE_string(model_path, "tinyllama_lora_fa.pte", "Path to .pte file.");
DEFINE_string(data_path, "tinyllama_lora_fa_weights.ptd", "Path to .ptd file.");
DEFINE_string(train_path, "sst2_train_tokens.tsv", "Tokenized SST2 train TSV.");
DEFINE_string(eval_path, "sst2_dev_tokens.tsv", "Tokenized SST2 eval TSV.");
DEFINE_string(log_path, "log.json", "Path to JSON evaluation log.");
DEFINE_int32(batch_size, 4, "Batch size.");
DEFINE_int32(seq_len, 64, "Maximum input length.");
DEFINE_int32(steps, 2000, "Number of zeroth-order finetuning steps.");
DEFINE_int32(eval_step, 50, "Evaluate after this many training steps, -1 disables eval.");
DEFINE_int32(vocab_size, 32000, "Token vocabulary size.");
DEFINE_int32(seed, 1234, "Random seed.");
DEFINE_double(epsilon, 1e-2, "Zeroth-order perturbation size.");
DEFINE_double(eta, 5e-5, "Learning rate.");
DEFINE_int32(cpu_threads, -1, "CPU thread count, <=0 auto-detects logical CPU count.");
DEFINE_bool(
    profile_step_timing,
    true,
    "Print per-step timing breakdown for data, noise, LoRA-B update, and forwards.");

namespace {

constexpr size_t kMaxSst2Samples = 1000;

using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::Result;
using Clock = std::chrono::steady_clock;

#if defined(ET_USE_THREADPOOL)
uint32_t detect_logical_cpu_threads() {
  if (cpuinfo_initialize()) {
    const uint32_t cpuinfo_threads = cpuinfo_get_processors_count();
    if (cpuinfo_threads > 0) {
      return cpuinfo_threads;
    }
  }
  const uint32_t hardware_threads = std::thread::hardware_concurrency();
  return hardware_threads > 0 ? hardware_threads : 1;
}
#endif

struct Sst2ClassPrompt {
  int32_t prompt_len = 0;
  int64_t label_token_id = -1;
  std::vector<int64_t> tokens;
};

struct Sst2Sample {
  int32_t label = -1;
  Sst2ClassPrompt negative;
  Sst2ClassPrompt positive;
};

struct Sst2Dataset {
  int32_t seq_len = -1;
  int64_t pad_token_id = -1;
  int64_t negative_label_token_id = -1;
  int64_t positive_label_token_id = -1;
  std::vector<Sst2Sample> samples;
};

struct EvalResult {
  size_t correct = 0;
  size_t total = 0;

  double accuracy() const {
    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
  }
};

struct EvalLogEntry {
  double accuracy = 0.0;
  double training_time_s = 0.0;
  long pss_kb = -1;
};

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

[[noreturn]] void fail(const std::string& message) {
  std::cerr << "ERROR: " << message << std::endl;
  std::exit(1);
}

void check_error(Error error, const std::string& message) {
  if (error != Error::Ok) {
    fail(message + " error=" + std::to_string(static_cast<uint32_t>(error)));
  }
}

template <typename T>
T get_or_fail(Result<T>&& result, const std::string& message) {
  if (!result.ok()) {
    fail(message + " error=" +
         std::to_string(static_cast<uint32_t>(result.error())));
  }
  return std::move(result.get());
}

long read_status_kb(const std::string& target_key) {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == target_key) {
      long value = 0;
      std::string unit;
      status >> value >> unit;
      return value;
    }
    std::string rest;
    std::getline(status, rest);
  }
  return -1;
}

long read_peak_rss_kb() {
  return read_status_kb("VmHWM:");
}

long read_pss_from_smaps(const char* path, bool sum_all) {
  std::ifstream smaps(path);
  if (!smaps.is_open()) {
    return -1;
  }

  long total_kb = 0;
  bool found = false;
  std::string line;
  while (std::getline(smaps, line)) {
    if (line.rfind("Pss:", 0) != 0) {
      continue;
    }
    std::istringstream stream(line.substr(4));
    long value_kb = 0;
    std::string unit;
    if (!(stream >> value_kb >> unit)) {
      continue;
    }
    if (!sum_all) {
      return value_kb;
    }
    total_kb += value_kb;
    found = true;
  }
  return found ? total_kb : -1;
}

long read_current_pss_kb() {
  const long rollup_kb = read_pss_from_smaps("/proc/self/smaps_rollup", false);
  if (rollup_kb >= 0) {
    return rollup_kb;
  }
  return read_pss_from_smaps("/proc/self/smaps", true);
}

void print_memory_mb(const char* key, long value_kb) {
  if (value_kb >= 0) {
    std::cout << key << "=" << (static_cast<double>(value_kb) / 1024.0);
  } else {
    std::cout << key << "=unavailable";
  }
}

double kb_to_mib(long value_kb) {
  return static_cast<double>(value_kb) / 1024.0;
}

size_t numel_from_sizes(executorch::runtime::Span<const int32_t> sizes) {
  size_t numel = 1;
  for (int32_t size : sizes) {
    numel *= static_cast<size_t>(size);
  }
  return numel;
}

size_t get_lora_b_numel(Module& module) {
  auto meta = get_or_fail(module.method_meta("forward"), "method_meta failed");
  if (meta.num_inputs() != 3) {
    fail("Expected forward(tokens, labels, lora_b_flat), got " +
         std::to_string(meta.num_inputs()) + " inputs");
  }
  auto input_meta = get_or_fail(
      meta.input_tensor_meta(2), "input_tensor_meta(2) failed");
  if (input_meta.scalar_type() != ScalarType::Half) {
    fail("Expected lora_b_flat to be fp16 input");
  }
  return numel_from_sizes(input_meta.sizes());
}

float run_forward(
    Module& module,
    const TensorPtr& tokens,
    const TensorPtr& labels,
    const TensorPtr& lora_b) {
  auto outputs = get_or_fail(
      module.forward({tokens, labels, lora_b}), "forward execution failed");
  if (outputs.empty() || !outputs[0].isTensor()) {
    fail("Expected tensor loss output");
  }
  auto loss_tensor = outputs[0].toTensor();
  if (loss_tensor.numel() != 1 ||
      loss_tensor.scalar_type() != ScalarType::Float) {
    fail("Expected scalar fp32 loss output");
  }
  return loss_tensor.const_data_ptr<float>()[0];
}

void check_finite_loss(float loss, const char* name, int32_t step) {
  if (!std::isfinite(loss)) {
    fail(
        "Non-finite " + std::string(name) + " at step=" +
        std::to_string(step) + ". Re-export the model with the stabilized "
        "fp32 single-label loss graph.");
  }
}

float half_to_float(Half value) {
  return c10::detail::fp16_ieee_to_fp32_value(value.x);
}

Half float_to_half(float value) {
  return Half(c10::detail::fp16_ieee_from_fp32_value(value), Half::from_bits());
}

void fill_noise(std::vector<float>& noise, std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (float& value : noise) {
    value = normal(rng);
  }
}

void materialize_lora_b(
    const std::vector<float>& master,
    const std::vector<float>& noise,
    float noise_scale,
    std::vector<Half>& out) {
  for (size_t i = 0; i < master.size(); ++i) {
    out[i] = float_to_half(master[i] + noise_scale * noise[i]);
  }
}

void materialize_lora_b_master(
    const std::vector<float>& master,
    std::vector<Half>& out) {
  for (size_t i = 0; i < master.size(); ++i) {
    out[i] = float_to_half(master[i]);
  }
}

void zo_update_master(
    std::vector<float>& master,
    const std::vector<float>& noise,
    float eta,
    float grad_scale) {
  for (size_t i = 0; i < master.size(); ++i) {
    master[i] -= eta * grad_scale * noise[i];
  }
}

std::vector<std::string> split_tsv(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream stream(line);
  while (std::getline(stream, field, '\t')) {
    fields.push_back(field);
  }
  if (!line.empty() && line.back() == '\t') {
    fields.emplace_back();
  }
  return fields;
}

int64_t parse_int64(
    const std::string& field,
    const std::string& path,
    size_t line_number,
    const char* name) {
  errno = 0;
  char* end = nullptr;
  const long long value = std::strtoll(field.c_str(), &end, 10);
  if (errno != 0 || end == field.c_str() || *end != '\0') {
    fail(
        "Invalid integer " + std::string(name) + " at " + path + ":" +
        std::to_string(line_number));
  }
  return static_cast<int64_t>(value);
}

int32_t parse_int32(
    const std::string& field,
    const std::string& path,
    size_t line_number,
    const char* name) {
  const int64_t value = parse_int64(field, path, line_number, name);
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    fail(
        "Out-of-range integer " + std::string(name) + " at " + path + ":" +
        std::to_string(line_number));
  }
  return static_cast<int32_t>(value);
}

void assign_metadata(
    Sst2Dataset& dataset,
    const std::vector<std::string>& fields,
    const std::string& path,
    size_t line_number) {
  if (fields.size() != 2) {
    fail("Expected key/value metadata at " + path + ":" + std::to_string(line_number));
  }
  const int64_t value = parse_int64(fields[1], path, line_number, fields[0].c_str());
  if (fields[0] == "seq_len") {
    dataset.seq_len = static_cast<int32_t>(value);
  } else if (fields[0] == "pad_token_id") {
    dataset.pad_token_id = value;
  } else if (fields[0] == "negative_label_token_id") {
    dataset.negative_label_token_id = value;
  } else if (fields[0] == "positive_label_token_id") {
    dataset.positive_label_token_id = value;
  } else if (fields[0] == "negative_token_id" ||
             fields[0] == "positive_token_id") {
    fail(
        path +
        " appears to use the old sst2_tokens_v2 format. Regenerate tokens "
        "with the updated prepare_sst2_tokens.py so class suffix prefix "
        "tokens are included in the input.");
  } else {
    fail("Unknown token TSV metadata key " + fields[0] + " in " + path);
  }
}

void check_vocab_id(
    int64_t token,
    int32_t vocab_size,
    const std::string& path,
    size_t line_number,
    const char* name) {
  if (token < 0 || token >= vocab_size) {
    fail(
        std::string(name) + " out of --vocab_size range at " + path + ":" +
        std::to_string(line_number));
  }
}

void validate_class_prompt(
    const Sst2ClassPrompt& prompt,
    int64_t expected_label_token_id,
    int32_t seq_len,
    int32_t vocab_size,
    const std::string& path,
    size_t line_number,
    const char* class_name) {
  if (prompt.prompt_len <= 0 || prompt.prompt_len > seq_len) {
    fail(
        std::string(class_name) +
        "_prompt_len must be in [1, seq_len] at " + path + ":" +
        std::to_string(line_number));
  }
  if (prompt.label_token_id != expected_label_token_id) {
    fail(
        std::string(class_name) +
        "_label_token_id does not match TSV metadata at " + path + ":" +
        std::to_string(line_number));
  }
  check_vocab_id(
      prompt.label_token_id,
      vocab_size,
      path,
      line_number,
      (std::string(class_name) + "_label_token_id").c_str());
}

void read_class_tokens(
    const std::vector<std::string>& fields,
    size_t first_field,
    int32_t seq_len,
    int32_t vocab_size,
    const std::string& path,
    size_t line_number,
    const char* class_name,
    std::vector<int64_t>& tokens) {
  tokens.reserve(static_cast<size_t>(seq_len));
  const std::string token_name = std::string(class_name) + "_token";
  for (int32_t index = 0; index < seq_len; ++index) {
    const int64_t token = parse_int64(
        fields[first_field + static_cast<size_t>(index)],
        path,
        line_number,
        token_name.c_str());
    check_vocab_id(token, vocab_size, path, line_number, token_name.c_str());
    tokens.push_back(token);
  }
}

Sst2Dataset load_sst2_token_tsv(
    const std::string& path,
    int32_t seq_len,
    int32_t vocab_size) {
  std::ifstream input(path);
  if (!input.is_open()) {
    fail("Cannot open tokenized SST2 TSV: " + path);
  }

  Sst2Dataset dataset;
  bool saw_columns = false;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty() || line[0] == '#') {
      continue;
    }

    const std::vector<std::string> fields = split_tsv(line);
    if (!saw_columns && fields[0] == "label") {
      if (dataset.seq_len != seq_len) {
        fail(
            path + " seq_len=" + std::to_string(dataset.seq_len) +
            " does not match --seq_len=" + std::to_string(seq_len));
      }
      if (fields.size() == static_cast<size_t>(seq_len) + 2 &&
          fields.size() > 1 && fields[1] == "prompt_len") {
        fail(
            path +
            " appears to use the old sst2_tokens_v2 format. Regenerate tokens "
            "with the updated prepare_sst2_tokens.py so class suffix prefix "
            "tokens are included in the input.");
      }
      if (dataset.pad_token_id < 0 || dataset.negative_label_token_id < 0 ||
          dataset.positive_label_token_id < 0) {
        fail("Tokenized SST2 TSV metadata is incomplete before sample header: " + path);
      }
      if (fields.size() != static_cast<size_t>(seq_len) * 2 + 5 ||
          fields[1] != "negative_prompt_len" ||
          fields[2] != "negative_label_token_id" ||
          fields[3] != "positive_prompt_len" ||
          fields[4] != "positive_label_token_id") {
        fail("Unexpected token TSV sample header in " + path);
      }
      saw_columns = true;
      continue;
    }
    if (!saw_columns) {
      assign_metadata(dataset, fields, path, line_number);
      continue;
    }
    if (dataset.samples.size() >= kMaxSst2Samples) {
      break;
    }
    if (fields.size() != static_cast<size_t>(seq_len) * 2 + 5) {
      fail("Unexpected token column count at " + path + ":" + std::to_string(line_number));
    }

    Sst2Sample sample;
    sample.label = parse_int32(fields[0], path, line_number, "label");
    sample.negative.prompt_len =
        parse_int32(fields[1], path, line_number, "negative_prompt_len");
    sample.negative.label_token_id =
        parse_int64(fields[2], path, line_number, "negative_label_token_id");
    sample.positive.prompt_len =
        parse_int32(fields[3], path, line_number, "positive_prompt_len");
    sample.positive.label_token_id =
        parse_int64(fields[4], path, line_number, "positive_label_token_id");
    if (sample.label < 0 || sample.label > 1) {
      fail("SST2 label must be 0 or 1 at " + path + ":" + std::to_string(line_number));
    }
    validate_class_prompt(
        sample.negative,
        dataset.negative_label_token_id,
        seq_len,
        vocab_size,
        path,
        line_number,
        "negative");
    validate_class_prompt(
        sample.positive,
        dataset.positive_label_token_id,
        seq_len,
        vocab_size,
        path,
        line_number,
        "positive");
    read_class_tokens(
        fields,
        5,
        seq_len,
        vocab_size,
        path,
        line_number,
        "negative",
        sample.negative.tokens);
    read_class_tokens(
        fields,
        5 + static_cast<size_t>(seq_len),
        seq_len,
        vocab_size,
        path,
        line_number,
        "positive",
        sample.positive.tokens);
    dataset.samples.push_back(std::move(sample));
  }

  if (!saw_columns || dataset.samples.empty()) {
    fail("Tokenized SST2 TSV has no samples: " + path);
  }
  if (dataset.pad_token_id < 0 || dataset.negative_label_token_id < 0 ||
      dataset.positive_label_token_id < 0) {
    fail("Tokenized SST2 TSV metadata is incomplete: " + path);
  }
  check_vocab_id(dataset.pad_token_id, vocab_size, path, 0, "pad_token_id");
  for (int64_t class_token :
       {dataset.negative_label_token_id, dataset.positive_label_token_id}) {
    if (class_token >= vocab_size) {
      fail("Verbalizer token id out of --vocab_size range in " + path);
    }
  }
  return dataset;
}

const Sst2ClassPrompt& class_prompt(const Sst2Sample& sample, int32_t label) {
  return label == 0 ? sample.negative : sample.positive;
}

void fill_row(
    const Sst2ClassPrompt& prompt,
    int32_t seq_len,
    size_t batch_row,
    std::vector<int64_t>& tokens,
    std::vector<int64_t>& labels) {
  const size_t offset = batch_row * static_cast<size_t>(seq_len);
  std::copy(prompt.tokens.begin(), prompt.tokens.end(), tokens.begin() + offset);
  labels[offset + static_cast<size_t>(prompt.prompt_len - 1)] =
      prompt.label_token_id;
}

void fill_train_batch(
    const Sst2Dataset& dataset,
    const std::vector<size_t>& indices,
    size_t cursor,
    int32_t batch_size,
    int32_t seq_len,
    std::vector<int64_t>& tokens,
    std::vector<int64_t>& labels) {
  std::fill(labels.begin(), labels.end(), -100);
  for (int32_t batch_row = 0; batch_row < batch_size; ++batch_row) {
    const Sst2Sample& sample = dataset.samples[indices[cursor + batch_row]];
    fill_row(
        class_prompt(sample, sample.label),
        seq_len,
        static_cast<size_t>(batch_row),
        tokens,
        labels);
  }
}

void fill_eval_batch(
    const Sst2Sample& sample,
    int32_t class_label,
    int32_t batch_size,
    int32_t seq_len,
    std::vector<int64_t>& tokens,
    std::vector<int64_t>& labels) {
  std::fill(labels.begin(), labels.end(), -100);
  for (int32_t batch_row = 0; batch_row < batch_size; ++batch_row) {
    fill_row(
        class_prompt(sample, class_label),
        seq_len,
        static_cast<size_t>(batch_row),
        tokens,
        labels);
  }
}

EvalResult evaluate(
    Module& module,
    const Sst2Dataset& dataset,
    int32_t batch_size,
    int32_t seq_len,
    std::vector<int64_t>& tokens_data,
    std::vector<int64_t>& labels_data,
    const TensorPtr& tokens,
    const TensorPtr& labels,
    const TensorPtr& lora_b) {
  EvalResult result;
  for (const Sst2Sample& sample : dataset.samples) {
    fill_eval_batch(
        sample,
        0,
        batch_size,
        seq_len,
        tokens_data,
        labels_data);
    const float negative_loss = run_forward(module, tokens, labels, lora_b);

    fill_eval_batch(
        sample,
        1,
        batch_size,
        seq_len,
        tokens_data,
        labels_data);
    const float positive_loss = run_forward(module, tokens, labels, lora_b);

    const int32_t predicted_label = positive_loss < negative_loss ? 1 : 0;
    result.correct += predicted_label == sample.label ? 1 : 0;
    ++result.total;
  }
  return result;
}

void print_eval(
    const char* tag,
    const EvalResult& result,
    bool print_training_time,
    double training_time_s,
    long pss_kb) {
  std::cout << tag << " accuracy=" << (result.accuracy() * 100.0)
            << "% correct=" << result.correct
            << " samples=" << result.total;
  if (print_training_time) {
    std::cout << " training_time_s=" << training_time_s;
  }
  std::cout << " ";
  print_memory_mb("pss_mib", pss_kb);
  std::cout << std::endl;
}

void write_eval_log_json(
    const std::string& path,
    const std::vector<EvalLogEntry>& entries) {
  std::ofstream output(path);
  if (!output.is_open()) {
    fail("Cannot open evaluation log for writing: " + path);
  }
  output << std::setprecision(10);
  output << "[\n";
  for (size_t index = 0; index < entries.size(); ++index) {
    const EvalLogEntry& entry = entries[index];
    output << "  {\n";
    output << "    \"accuracy\": " << entry.accuracy << ",\n";
    output << "    \"training_time_s\": " << entry.training_time_s << ",\n";
    output << "    \"pss_mib\": ";
    if (entry.pss_kb >= 0) {
      output << kb_to_mib(entry.pss_kb);
    } else {
      output << "null";
    }
    output << "\n";
    output << "  }";
    if (index + 1 < entries.size()) {
      output << ",";
    }
    output << "\n";
  }
  output << "]\n";
}

void record_eval(
    const std::string& tag,
    const EvalResult& result,
    bool print_training_time,
    double training_time_ms,
    std::vector<EvalLogEntry>& eval_log) {
  const long pss_kb = read_current_pss_kb();
  const double training_time_s = training_time_ms / 1000.0;
  print_eval(tag.c_str(), result, print_training_time, training_time_s, pss_kb);
  eval_log.push_back(
      EvalLogEntry{result.accuracy(), training_time_s, pss_kb});
  write_eval_log_json(FLAGS_log_path, eval_log);
}

void check_dataset_pair(const Sst2Dataset& train, const Sst2Dataset& eval) {
  if (train.seq_len != eval.seq_len || train.pad_token_id != eval.pad_token_id ||
      train.negative_label_token_id != eval.negative_label_token_id ||
      train.positive_label_token_id != eval.positive_label_token_id) {
    fail("Train and eval token TSV metadata do not match");
  }
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_batch_size <= 0 || FLAGS_seq_len <= 1 || FLAGS_steps <= 0 ||
      FLAGS_vocab_size <= 1 || FLAGS_eval_step == 0 || FLAGS_eval_step < -1) {
    fail("Invalid batch_size, seq_len, steps, eval_step, or vocab_size");
  }
  const bool eval_enabled = FLAGS_eval_step != -1;

#if defined(ET_USE_THREADPOOL)
  const uint32_t num_threads = FLAGS_cpu_threads > 0
      ? static_cast<uint32_t>(FLAGS_cpu_threads)
      : detect_logical_cpu_threads();
  if (num_threads > 0) {
    executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_threads);
  }
  std::cout << "threadpool_threads=" << num_threads << std::endl;
#else
  std::cout << "threadpool_threads=disabled" << std::endl;
#endif

  Sst2Dataset train_data =
      load_sst2_token_tsv(FLAGS_train_path, FLAGS_seq_len, FLAGS_vocab_size);
  Sst2Dataset eval_data;
  if (eval_enabled) {
    eval_data =
        load_sst2_token_tsv(FLAGS_eval_path, FLAGS_seq_len, FLAGS_vocab_size);
    check_dataset_pair(train_data, eval_data);
  }
  if (train_data.samples.size() < static_cast<size_t>(FLAGS_batch_size)) {
    fail("Training split has fewer samples than --batch_size");
  }
  std::cout << "train_samples=" << train_data.samples.size()
            << " eval_samples="
            << (eval_enabled ? std::to_string(eval_data.samples.size()) : "disabled")
            << " seq_len=" << train_data.seq_len
            << " negative_label_token_id=" << train_data.negative_label_token_id
            << " positive_label_token_id=" << train_data.positive_label_token_id
            << std::endl;

  Module module(
      FLAGS_model_path,
      FLAGS_data_path,
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  check_error(module.load(), "Module load failed");
  check_error(module.load_forward(), "Forward method load failed");

  const size_t lora_b_numel = get_lora_b_numel(module);
  std::cout << "lora_b_numel=" << lora_b_numel << std::endl;

  const size_t batch_numel =
      static_cast<size_t>(FLAGS_batch_size) * static_cast<size_t>(FLAGS_seq_len);
  std::vector<int64_t> tokens_data(batch_numel, train_data.pad_token_id);
  std::vector<int64_t> labels_data(batch_numel, -100);
  std::vector<float> lora_b_master(lora_b_numel, 0.0f);
  std::vector<Half> lora_b_data(lora_b_numel, float_to_half(0.0f));
  std::vector<float> noise(lora_b_numel);

  auto tokens = from_blob(
      tokens_data.data(),
      {FLAGS_batch_size, FLAGS_seq_len},
      ScalarType::Long);
  auto labels = from_blob(
      labels_data.data(),
      {FLAGS_batch_size, FLAGS_seq_len},
      ScalarType::Long);
  auto lora_b = from_blob(
      lora_b_data.data(),
      {static_cast<executorch::aten::SizesType>(lora_b_data.size())},
      ScalarType::Half);

  std::vector<EvalLogEntry> eval_log;
  if (eval_enabled) {
    const EvalResult initial_eval = evaluate(
        module,
        eval_data,
        FLAGS_batch_size,
        FLAGS_seq_len,
        tokens_data,
        labels_data,
        tokens,
        labels,
        lora_b);
    record_eval("eval[initial]", initial_eval, false, 0.0, eval_log);
  }

  std::mt19937 noise_rng(static_cast<uint32_t>(FLAGS_seed));
  std::mt19937 data_rng(static_cast<uint32_t>(FLAGS_seed) + 1U);
  std::vector<size_t> train_indices(train_data.samples.size());
  std::iota(train_indices.begin(), train_indices.end(), 0);
  size_t data_cursor = 0;

  const int64_t tokens_per_forward =
      static_cast<int64_t>(FLAGS_batch_size) * FLAGS_seq_len;
  const float epsilon = static_cast<float>(FLAGS_epsilon);
  const float eta = static_cast<float>(FLAGS_eta);
  double total_step_ms = 0.0;
  double total_forward_ms = 0.0;
  double training_time_ms = 0.0;

  for (int32_t step = 0; step < FLAGS_steps; ++step) {
    const auto step_start = Clock::now();
    const auto batch_start = step_start;
    if (data_cursor + static_cast<size_t>(FLAGS_batch_size) >
        train_indices.size()) {
      std::shuffle(train_indices.begin(), train_indices.end(), data_rng);
      data_cursor = 0;
    }
    fill_train_batch(
        train_data,
        train_indices,
        data_cursor,
        FLAGS_batch_size,
        FLAGS_seq_len,
        tokens_data,
        labels_data);
    data_cursor += static_cast<size_t>(FLAGS_batch_size);
    const auto batch_end = Clock::now();

    const auto noise_start = Clock::now();
    fill_noise(noise, noise_rng);
    const auto noise_end = Clock::now();

    const auto add_pos_start = Clock::now();
    materialize_lora_b(lora_b_master, noise, epsilon, lora_b_data);
    const auto add_pos_end = Clock::now();
    std::cout << "start pos forward step=" << step << std::endl;
    const auto plus_start = Clock::now();
    const float loss_plus = run_forward(module, tokens, labels, lora_b);
    const auto plus_end = Clock::now();
    check_finite_loss(loss_plus, "loss_plus", step);
    std::cout << "end pos forward step=" << step
              << " time_ms=" << elapsed_ms(plus_start, plus_end)
              << " loss=" << loss_plus << std::endl;

    const auto add_neg_start = Clock::now();
    materialize_lora_b(lora_b_master, noise, -epsilon, lora_b_data);
    const auto add_neg_end = Clock::now();
    std::cout << "start neg forward step=" << step << std::endl;
    const auto minus_start = Clock::now();
    const float loss_minus = run_forward(module, tokens, labels, lora_b);
    const auto minus_end = Clock::now();
    check_finite_loss(loss_minus, "loss_minus", step);
    std::cout << "end neg forward step=" << step
              << " time_ms=" << elapsed_ms(minus_start, minus_end)
              << " loss=" << loss_minus << std::endl;

    const float grad_scale = (loss_plus - loss_minus) / (2.0f * epsilon);
    const auto update_start = Clock::now();
    zo_update_master(lora_b_master, noise, eta, grad_scale);
    const auto update_end = Clock::now();

    const auto step_end = Clock::now();
    const double forward_ms =
        elapsed_ms(plus_start, plus_end) + elapsed_ms(minus_start, minus_end);
    const double step_ms = elapsed_ms(step_start, step_end);
    total_forward_ms += forward_ms;
    total_step_ms += step_ms;
    training_time_ms += step_ms;

    std::cout << "step[" << step << "]"
              << " time_ms=" << step_ms
              << " loss_plus=" << loss_plus
              << " loss_minus=" << loss_minus
              << " g=" << grad_scale << std::endl;
    if (FLAGS_profile_step_timing) {
      std::cout << "step_timing[" << step << "]"
                << " batch_ms=" << elapsed_ms(batch_start, batch_end)
                << " noise_ms=" << elapsed_ms(noise_start, noise_end)
                << " add_pos_ms=" << elapsed_ms(add_pos_start, add_pos_end)
                << " forward_pos_ms=" << elapsed_ms(plus_start, plus_end)
                << " add_neg_ms=" << elapsed_ms(add_neg_start, add_neg_end)
                << " forward_neg_ms=" << elapsed_ms(minus_start, minus_end)
                << " update_ms=" << elapsed_ms(update_start, update_end)
                << std::endl;
    }

    if (eval_enabled && (step + 1) % FLAGS_eval_step == 0) {
      materialize_lora_b_master(lora_b_master, lora_b_data);
      const EvalResult eval = evaluate(
          module,
          eval_data,
          FLAGS_batch_size,
          FLAGS_seq_len,
          tokens_data,
          labels_data,
          tokens,
          labels,
          lora_b);
      const std::string tag = "eval[step=" + std::to_string(step + 1) + "]";
      record_eval(tag, eval, true, training_time_ms, eval_log);
    }
  }

  const double avg_step_time_ms = total_step_ms / FLAGS_steps;
  const double speed_pp =
      (static_cast<double>(tokens_per_forward) * FLAGS_steps * 2.0 * 1000.0) /
      total_forward_ms;
  const long peak_rss_kb = read_peak_rss_kb();

  std::cout << "speed_pp=" << speed_pp << " tokens/s" << std::endl;
  std::cout << "avg_step_time_ms=" << avg_step_time_ms << std::endl;
  print_memory_mb("peak_rss_mib", peak_rss_kb);
  std::cout << std::endl;

  return 0;
}
