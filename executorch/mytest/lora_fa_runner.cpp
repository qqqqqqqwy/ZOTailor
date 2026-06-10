#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
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
DEFINE_int32(batch_size, 4, "Batch size.");
DEFINE_int32(seq_len, 32, "Sequence length.");
DEFINE_int32(steps, 5, "Number of zeroth-order finetuning steps.");
DEFINE_int32(vocab_size, 32000, "Dummy token vocabulary size.");
DEFINE_int32(seed, 1234, "Random seed.");
DEFINE_double(epsilon, 1e-2, "Zeroth-order perturbation size.");
DEFINE_double(eta, 5e-5, "Learning rate.");
DEFINE_int32(cpu_threads, -1, "CPU thread count, <=0 auto-detects logical CPU count.");

namespace {

using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
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

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

long read_peak_rss_kb() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmHWM:") {
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

size_t numel_from_sizes(executorch::runtime::Span<const int32_t> sizes) {
  size_t numel = 1;
  for (int32_t size : sizes) {
    numel *= static_cast<size_t>(size);
  }
  return numel;
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
  if (loss_tensor.numel() != 1 || loss_tensor.scalar_type() != ScalarType::Float) {
    fail("Expected scalar fp32 loss output");
  }
  return loss_tensor.const_data_ptr<float>()[0];
}

float half_to_float(Half value) {
  return c10::detail::fp16_ieee_to_fp32_value(value.x);
}

Half float_to_half(float value) {
  return Half(c10::detail::fp16_ieee_from_fp32_value(value), Half::from_bits());
}

void make_dummy_tokens_and_labels(
    std::vector<int64_t>& tokens,
    std::vector<int64_t>& labels,
    int32_t batch_size,
    int32_t seq_len,
    int32_t vocab_size,
    std::mt19937& rng) {
  std::uniform_int_distribution<int64_t> token_dist(0, vocab_size - 1);
  tokens.resize(static_cast<size_t>(batch_size) * seq_len);
  labels.resize(tokens.size());

  for (int32_t b = 0; b < batch_size; ++b) {
    for (int32_t t = 0; t < seq_len; ++t) {
      tokens[static_cast<size_t>(b) * seq_len + t] = token_dist(rng);
    }
    for (int32_t t = 0; t < seq_len - 1; ++t) {
      labels[static_cast<size_t>(b) * seq_len + t] =
          tokens[static_cast<size_t>(b) * seq_len + t + 1];
    }
    labels[static_cast<size_t>(b) * seq_len + seq_len - 1] = -100;
  }
}

void fill_noise(std::vector<float>& noise, std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (float& value : noise) {
    value = normal(rng);
  }
}

void add_scaled_noise(
    std::vector<Half>& b,
    const std::vector<float>& noise,
    float scale) {
  for (size_t i = 0; i < b.size(); ++i) {
    const float value = half_to_float(b[i]) + scale * noise[i];
    b[i] = float_to_half(value);
  }
}

void zo_update(
    std::vector<Half>& b,
    const std::vector<float>& noise,
    float epsilon,
    float eta,
    float grad_scale) {
  for (size_t i = 0; i < b.size(); ++i) {
    const float value =
        half_to_float(b[i]) + epsilon * noise[i] - eta * grad_scale * noise[i];
    b[i] = float_to_half(value);
  }
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_batch_size <= 0 || FLAGS_seq_len <= 1 || FLAGS_steps <= 0 ||
      FLAGS_vocab_size <= 1) {
    fail("Invalid batch_size, seq_len, steps, or vocab_size");
  }

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

  Module module(
      FLAGS_model_path,
      FLAGS_data_path,
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  check_error(module.load(), "Module load failed");
  check_error(module.load_forward(), "Forward method load failed");

  const size_t lora_b_numel = get_lora_b_numel(module);
  std::cout << "lora_b_numel=" << lora_b_numel << std::endl;

  std::mt19937 rng(static_cast<uint32_t>(FLAGS_seed));
  std::vector<int64_t> tokens_data;
  std::vector<int64_t> labels_data;
  make_dummy_tokens_and_labels(
      tokens_data,
      labels_data,
      FLAGS_batch_size,
      FLAGS_seq_len,
      FLAGS_vocab_size,
      rng);

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

  const int64_t tokens_per_forward =
      static_cast<int64_t>(FLAGS_batch_size) * FLAGS_seq_len;
  const float epsilon = static_cast<float>(FLAGS_epsilon);
  const float eta = static_cast<float>(FLAGS_eta);
  double total_step_ms = 0.0;
  double total_forward_ms = 0.0;

  for (int32_t step = 0; step < FLAGS_steps; ++step) {
    const auto step_start = Clock::now();
    fill_noise(noise, rng);

    add_scaled_noise(lora_b_data, noise, epsilon);
    const auto plus_start = Clock::now();
    const float loss_plus = run_forward(module, tokens, labels, lora_b);
    const auto plus_end = Clock::now();

    add_scaled_noise(lora_b_data, noise, -2.0f * epsilon);
    const auto minus_start = Clock::now();
    const float loss_minus = run_forward(module, tokens, labels, lora_b);
    const auto minus_end = Clock::now();

    const float grad_scale = (loss_plus - loss_minus) / (2.0f * epsilon);
    zo_update(lora_b_data, noise, epsilon, eta, grad_scale);

    const auto step_end = Clock::now();
    const double forward_ms =
        elapsed_ms(plus_start, plus_end) + elapsed_ms(minus_start, minus_end);
    const double step_ms = elapsed_ms(step_start, step_end);
    total_forward_ms += forward_ms;
    total_step_ms += step_ms;

    std::cout << "step[" << step << "]"
              << " time_ms=" << step_ms
              << " loss_plus=" << loss_plus
              << " loss_minus=" << loss_minus
              << " g=" << grad_scale << std::endl;
  }

  const double avg_step_time_ms = total_step_ms / FLAGS_steps;
  const double speed_pp =
      (static_cast<double>(tokens_per_forward) * FLAGS_steps * 2.0 * 1000.0) /
      total_forward_ms;
  const long peak_rss_kb = read_peak_rss_kb();

  std::cout << "speed_pp=" << speed_pp << " tokens/s" << std::endl;
  std::cout << "avg_step_time_ms=" << avg_step_time_ms << std::endl;
  if (peak_rss_kb >= 0) {
    std::cout << "peak_rss_mb=" << (static_cast<double>(peak_rss_kb) / 1024.0)
              << std::endl;
  } else {
    std::cout << "peak_rss_mb=unavailable" << std::endl;
  }

  return 0;
}
