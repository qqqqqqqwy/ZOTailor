#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <future>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

int zoo_lora_fa_benchmark_main(int argc, char ** argv);

#define main zoo_lora_fa_benchmark_main
#include "zo-lora-fa.cpp"
#undef main

namespace {

static const char * SST2_DEFAULT_TRAIN_PATH = "/data/local/tmp/zotailor/mine/train.tsv";
static const char * SST2_DEFAULT_EVAL_PATH  = "/data/local/tmp/zotailor/mine/dev.tsv";

static bool g_sst2_htp_q4_fused_lora_active = false;

enum class sst2_lora_exec_mode {
    RUNTIME,
    FUSED_HTP,
    MERGE_REPACK,
};

struct sst2_options {
    zo_mode              mode          = zo_mode::CPU;
    sst2_lora_exec_mode  lora_exec     = sst2_lora_exec_mode::RUNTIME;
    bool                 lora_exec_set = false;
    bool                 pipeline      = false;
    bool                 antithetic    = false;
    int                  steps         = 2000;
    int                  eval_step     = 50;
    int                  batch_size    = 4;
    int                  seq_len       = 64;
    int                  max_train     = 1000;
    int                  max_eval      = 1000;
    float                epsilon       = 1e-2f;
    float                lr            = 5e-5f;
    std::string          train_path    = SST2_DEFAULT_TRAIN_PATH;
    std::string          eval_path     = SST2_DEFAULT_EVAL_PATH;
    bool                 debug_padding = false;
    bool                 debug_logits  = false;
    bool                 debug_mnn     = false;
    int                  diag_layerwise = 0;
    std::vector<int>     diag_layerwise_layers{0, 1, 21};
    int                  diag_layerwise_values = 8;
    int                  diag_lora_perturb = 0;
    int                  diag_lora_a = 0;
    int                  diag_lora_branch = 0;
    std::vector<std::string> diag_lora_targets{
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.ffn_gate.weight",
        "blk.21.ffn_up.weight",
    };
    int                  diag_lora_values = 8;
    std::string          diag_lora_pattern = "noise";
    bool                 diag_lora_propagation = false;
    bool                 diag_attn_internals = false;
    std::vector<int>     diag_attn_heads{0};
    int                  diag_attn_values = 16;
};

struct sst2_sample {
    std::vector<llama_token> input_ids;
    int                      label = 0;
    llama_token              target_token = LLAMA_TOKEN_NULL;
};

struct sst2_batch_data {
    std::vector<const sst2_sample *> samples;
    std::vector<size_t>              sample_indices;
    std::vector<int32_t>             final_indices;
    int32_t                          n_tokens = 0;
};

struct sst2_forward_result {
    float   loss    = 0.0f;
    int64_t t_pp_us = 0;
    int32_t n_tokens = 0;
};

struct sst2_paired_forward_result {
    float   loss_plus  = 0.0f;
    float   loss_minus = 0.0f;
    int64_t t_pp_us    = 0;
    int32_t n_tokens   = 0;
};

struct sst2_logits_snapshot {
    int32_t n_vocab = 0;
    std::vector<float> logits;
    std::vector<llama_token> labels;
};

struct sst2_debug_state {
    bool                             enabled = false;
    std::string                      tag;
    const llama_vocab *              vocab = nullptr;
    const std::vector<llama_token> * class_tokens = nullptr;
    int                              sample_limit = 1;
    int                              printed_samples = 0;
};

struct sst2_layerwise_debug_state {
    bool enabled = false;
    bool active = false;
    std::string tag;
    int forward_id = 0;
    int sample_ordinal = -1;
    size_t data_index = (size_t) -1;
    int label = -1;
    llama_token target_token = LLAMA_TOKEN_NULL;
    int final_pos = -1;
    int token_count = -1;
    int max_values = 8;
    std::vector<int> layers{0, 1, 21};
};

struct sst2_lora_branch_debug_state {
    bool enabled = false;
    bool active = false;
    std::string tag;
    std::string selected_key;
    int forward_id = 0;
    int sample_ordinal = -1;
    size_t data_index = (size_t) -1;
    int label = -1;
    llama_token target_token = LLAMA_TOKEN_NULL;
    int final_pos = -1;
    int token_count = -1;
    int max_values = 8;
    int printed_ops = 0;
    bool propagation = false;
    bool attn_internals = false;
    std::vector<int> attn_heads{0};
    int attn_values = 16;
};

struct sst2_eval_result {
    float   accuracy = 0.0f;
    int32_t correct  = 0;
    int32_t total    = 0;
    int64_t t_eval_us = 0;
};

struct json_log_writer {
    std::ofstream file;
    bool          first = true;

    explicit json_log_writer(const char * path) : file(path, std::ios::out | std::ios::trunc) {
        if (!file.is_open()) {
            throw std::runtime_error(std::string("failed to open json log: ") + path);
        }
        file << "[\n";
    }

    void begin_object() {
        if (!first) {
            file << ",\n";
        }
        first = false;
        file << "  {";
    }

    void end_object() {
        file << "}";
        file.flush();
    }

    void close() {
        if (file.is_open()) {
            file << "\n]\n";
            file.close();
        }
    }

    ~json_log_writer() {
        close();
    }
};

static void append_eval_log(json_log_writer & log, float accuracy, double training_time_s, double pss_mb) {
    log.begin_object();
    log.file << "\"accuracy\":" << accuracy
             << ",\"training_time_s\":" << training_time_s
             << ",\"pss_mb\":" << pss_mb;
    log.end_object();
}

static void append_train_log(json_log_writer & log, int step, float loss, double time_ms) {
    log.begin_object();
    log.file << "\"step\":" << step
             << ",\"loss\":" << loss
             << ",\"time\":" << time_ms;
    log.end_object();
}

struct merge_repack_timing {
    int64_t merge_us = 0;
    int64_t set_us   = 0;

    int64_t total_us() const {
        return merge_us + set_us;
    }
};

struct merge_repack_weight {
    std::string         name;
    ggml_tensor *       base = nullptr;
    ggml_type           type = GGML_TYPE_COUNT;
    int64_t             k = 0;
    int64_t             n = 0;
    int64_t             r = 0;
    size_t              b_index = 0;
    float               scale = 1.0f;
    std::vector<float>  base_original;
    std::vector<float>  lora_a;
    std::vector<float>  merged;
    std::vector<uint8_t> io_buf;
};

struct sst2_perf_stats {
    int64_t total_pp_us                    = 0;
    int64_t total_pp_tokens                = 0;
    int64_t total_step_us                  = 0;
    int64_t total_prepare_noise_us         = 0;
    int64_t total_prepare_plus_us          = 0;
    int64_t total_prepare_minus_us         = 0;
    int64_t total_prepare_update_us        = 0;
    int64_t total_prepare_next_noise_us    = 0;
    int64_t total_push_plus_us             = 0;
    int64_t total_push_minus_us            = 0;
    int64_t total_push_antithetic_b_us     = 0;
    int64_t total_merge_plus_us            = 0;
    int64_t total_merge_minus_us           = 0;
    int64_t total_merge_update_us          = 0;
    int64_t total_wait_minus_ready_us      = 0;
    int64_t total_wait_next_noise_ready_us = 0;
    int64_t total_wait_plus_loss_ready_us  = 0;
    int64_t total_copy_plus_logits_us      = 0;
    int64_t total_forwards                 = 0;
};

static const char * sst2_lora_exec_mode_name(sst2_lora_exec_mode mode) {
    switch (mode) {
        case sst2_lora_exec_mode::RUNTIME:      return "runtime";
        case sst2_lora_exec_mode::FUSED_HTP:    return "fused-htp";
        case sst2_lora_exec_mode::MERGE_REPACK: return "merge-repack";
    }
    return "unknown";
}

static std::string sst2_trim(std::string s);

static bool sst2_parse_int_list(const std::string & text, std::vector<int> & values) {
    values.clear();
    std::string stripped = sst2_trim(text);
    if (stripped.empty() || stripped == "all" || stripped == "*") {
        return true;
    }
    std::stringstream ss(stripped);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = sst2_trim(item);
        if (item.empty()) {
            return false;
        }
        char * end = nullptr;
        const long parsed = std::strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
            return false;
        }
        values.push_back((int) parsed);
    }
    return true;
}

static std::string sst2_join_int_list(const std::vector<int> & values) {
    std::ostringstream os;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << values[i];
    }
    return os.str();
}

static bool sst2_parse_string_list(const std::string & text, std::vector<std::string> & values) {
    values.clear();
    std::string stripped = sst2_trim(text);
    if (stripped.empty() || stripped == "all" || stripped == "*") {
        return true;
    }
    std::stringstream ss(stripped);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = sst2_trim(item);
        if (item.empty()) {
            return false;
        }
        values.push_back(item);
    }
    return true;
}

static void sst2_print_usage(int /*argc*/, char ** argv) {
    LOG("\n");
    LOG("usage:\n");
    LOG("  %s [common llama.cpp args] [zoo SST2 args]\n", argv[0]);
    LOG("\n");
    LOG("zoo SST2 args:\n");
    LOG("  --mode <cpu|coop>                  execution mode (default: cpu)\n");
    LOG("  --lora-exec <runtime|fused-htp|merge-repack>\n");
    LOG("  --pipeline <true|false>            overlap CPU LoRA prep with NPU forward (default: false)\n");
    LOG("  --antithetic <true|false>          pair +/- perturbations in one fused HTP forward (default: false)\n");
    LOG("  --train-data <path>                SST2 train TSV (default: %s)\n", SST2_DEFAULT_TRAIN_PATH);
    LOG("  --eval-data <path>                 SST2 dev TSV (default: %s)\n", SST2_DEFAULT_EVAL_PATH);
    LOG("  --max-train <N>                    max train samples (default: 1000)\n");
    LOG("  --max-eval <N>                     max eval samples (default: 1000)\n");
    LOG("  --steps <N>                        ZO training steps (default: 2000)\n");
    LOG("  --eval-step <N>                    validation interval, -1 disables validation (default: 50)\n");
    LOG("  --batch-size <N>                   training batch size (default: 4)\n");
    LOG("  --seq-len <N>                      max prompt tokens per sample (default: 64)\n");
    LOG("  --epsilon <F>                      perturbation scale (default: 1e-2)\n");
    LOG("  --lr <F>                           learning rate (default: 5e-5)\n");
    LOG("  --debug_padding <true|false>       revert to OLD batch build without padding; CPU mode enables this automatically (default: false)\n");
    LOG("  --debug_logits <true|false>        revert to OLD logits=true on all real tokens (default: false)\n");
    LOG("  --debug_MNN <true|false>           print token/logit diagnostics comparable with MNN (default: false)\n");
    LOG("  --diag-layerwise <N>               dump selected layer activations for N warmup samples, then exit\n");
    LOG("  --diag-layerwise-layers <list>     comma-separated layers, or all (default: 0,1,21)\n");
    LOG("  --diag-layerwise-values <K>        values to print from each selected vector (default: 8)\n");
    LOG("  --diag-lora-a <N>                  dump selected LoRA-A tensors and x@A for N warmup samples, then exit\n");
    LOG("  --diag-lora-branch <N>             dump B=0/+eps/-eps LoRA branch internals for N warmup samples, then exit\n");
    LOG("  --diag-lora-perturb <N>            run B=0/+eps/-eps single-target LoRA-B diagnostics, then exit\n");
    LOG("  --diag-lora-targets <list>         comma-separated canonical targets, or all\n");
    LOG("  --diag-lora-values <K>             values to print from selected B tensors (default: 8)\n");
    LOG("  --diag-lora-pattern <kind>         noise|ramp|constant (default: noise)\n");
    LOG("  --diag-lora-propagation <true|false>\n");
    LOG("  --diag-attn-internals <true|false> dump QK/softmax/V attention internals under --diag-lora-branch\n");
    LOG("  --diag-attn-heads <list>           comma-separated attention heads (default: 0)\n");
    LOG("  --diag-attn-values <K>             values to print from attention rows (default: 16)\n");
    LOG("\n");
}

static bool sst2_prescan_mode(int argc, char ** argv, sst2_options & opts) {
    zo_options tmp;
    if (!prescan_mode(argc, argv, tmp)) {
        return false;
    }
    opts.mode = tmp.mode;
    return true;
}

static bool sst2_parse_custom_args(int argc, char ** argv, sst2_options & opts, std::vector<char *> & filtered_argv) {
    filtered_argv.clear();
    filtered_argv.reserve(argc);
    filtered_argv.push_back(argv[0]);

    const bool strip_device_args = opts.mode == zo_mode::CPU || (!zo_hexagon_compiled() && opts.mode == zo_mode::COOP);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;

        auto take_next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        try {
            if (split_arg(arg, "--mode", value)) {
                if (value.empty()) {
                    value = take_next("--mode");
                }
                if (value == "cpu") {
                    opts.mode = zo_mode::CPU;
                } else if (value == "coop") {
                    opts.mode = zo_mode::COOP;
                } else {
                    LOG_ERR("%s: invalid mode: %s\n", __func__, value.c_str());
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--lora-exec", value)) {
                if (value.empty()) {
                    value = take_next("--lora-exec");
                }
                if (value == "runtime") {
                    opts.lora_exec = sst2_lora_exec_mode::RUNTIME;
                } else if (value == "fused-htp") {
                    opts.lora_exec = sst2_lora_exec_mode::FUSED_HTP;
                } else if (value == "merge-repack") {
                    opts.lora_exec = sst2_lora_exec_mode::MERGE_REPACK;
                } else {
                    LOG_ERR("%s: invalid lora exec mode: %s\n", __func__, value.c_str());
                    return false;
                }
                opts.lora_exec_set = true;
                continue;
            }

            if (split_arg(arg, "--pipeline", value)) {
                if (value.empty()) {
                    value = take_next("--pipeline");
                }
                if (!parse_bool_arg(value, opts.pipeline, "--pipeline")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--antithetic", value)) {
                if (value.empty()) {
                    value = take_next("--antithetic");
                }
                if (!parse_bool_arg(value, opts.antithetic, "--antithetic")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--train-data", value)) {
                opts.train_path = value.empty() ? take_next("--train-data") : value;
                continue;
            }

            if (split_arg(arg, "--eval-data", value)) {
                opts.eval_path = value.empty() ? take_next("--eval-data") : value;
                continue;
            }

            if (split_arg(arg, "--steps", value)) {
                if (value.empty()) {
                    value = take_next("--steps");
                }
                if (!parse_int_arg(value, opts.steps, "--steps")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--eval-step", value) || split_arg(arg, "--eval_interval", value)) {
                if (value.empty()) {
                    value = take_next("--eval-step");
                }
                if (!parse_int_arg(value, opts.eval_step, "--eval-step")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--batch-size", value)) {
                if (value.empty()) {
                    value = take_next("--batch-size");
                }
                if (!parse_int_arg(value, opts.batch_size, "--batch-size")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--seq-len", value)) {
                if (value.empty()) {
                    value = take_next("--seq-len");
                }
                if (!parse_int_arg(value, opts.seq_len, "--seq-len")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--max-train", value)) {
                if (value.empty()) {
                    value = take_next("--max-train");
                }
                if (!parse_int_arg(value, opts.max_train, "--max-train")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--max-eval", value)) {
                if (value.empty()) {
                    value = take_next("--max-eval");
                }
                if (!parse_int_arg(value, opts.max_eval, "--max-eval")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--epsilon", value)) {
                if (value.empty()) {
                    value = take_next("--epsilon");
                }
                if (!parse_float_arg(value, opts.epsilon, "--epsilon")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--lr", value)) {
                if (value.empty()) {
                    value = take_next("--lr");
                }
                if (!parse_float_arg(value, opts.lr, "--lr")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--debug_padding", value)) {
                if (value.empty()) {
                    value = take_next("--debug_padding");
                }
                if (!parse_bool_arg(value, opts.debug_padding, "--debug_padding")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--debug_logits", value)) {
                if (value.empty()) {
                    value = take_next("--debug_logits");
                }
                if (!parse_bool_arg(value, opts.debug_logits, "--debug_logits")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--debug_MNN", value) || split_arg(arg, "--debug-mnn", value)) {
                if (value.empty()) {
                    value = take_next("--debug_MNN");
                }
                if (!parse_bool_arg(value, opts.debug_mnn, "--debug_MNN")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-layerwise", value)) {
                if (value.empty()) {
                    value = take_next("--diag-layerwise");
                }
                if (!parse_int_arg(value, opts.diag_layerwise, "--diag-layerwise")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-layerwise-layers", value)) {
                if (value.empty()) {
                    value = take_next("--diag-layerwise-layers");
                }
                if (!sst2_parse_int_list(value, opts.diag_layerwise_layers)) {
                    LOG_ERR("%s: --diag-layerwise-layers must be a comma-separated list of non-negative integers or all\n", __func__);
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-layerwise-values", value)) {
                if (value.empty()) {
                    value = take_next("--diag-layerwise-values");
                }
                if (!parse_int_arg(value, opts.diag_layerwise_values, "--diag-layerwise-values")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-lora-perturb", value)) {
                if (value.empty()) {
                    value = take_next("--diag-lora-perturb");
                }
                if (!parse_int_arg(value, opts.diag_lora_perturb, "--diag-lora-perturb")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-lora-a", value)) {
                if (value.empty()) {
                    value = take_next("--diag-lora-a");
                }
                if (!parse_int_arg(value, opts.diag_lora_a, "--diag-lora-a")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-lora-branch", value)) {
                if (value.empty()) {
                    value = take_next("--diag-lora-branch");
                }
                if (!parse_int_arg(value, opts.diag_lora_branch, "--diag-lora-branch")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-lora-propagation", value)) {
                if (value.empty()) {
                    value = take_next("--diag-lora-propagation");
                }
                if (!parse_bool_arg(value, opts.diag_lora_propagation, "--diag-lora-propagation")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-attn-internals", value)) {
                if (value.empty()) {
                    value = take_next("--diag-attn-internals");
                }
                if (!parse_bool_arg(value, opts.diag_attn_internals, "--diag-attn-internals")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-attn-heads", value)) {
                if (value.empty()) {
                    value = take_next("--diag-attn-heads");
                }
                if (!sst2_parse_int_list(value, opts.diag_attn_heads) || opts.diag_attn_heads.empty()) {
                    LOG_ERR("%s: --diag-attn-heads must be a non-empty comma-separated list of non-negative integers\n", __func__);
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-attn-values", value)) {
                if (value.empty()) {
                    value = take_next("--diag-attn-values");
                }
                if (!parse_int_arg(value, opts.diag_attn_values, "--diag-attn-values")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-lora-targets", value)) {
                if (value.empty()) {
                    value = take_next("--diag-lora-targets");
                }
                if (!sst2_parse_string_list(value, opts.diag_lora_targets)) {
                    LOG_ERR("%s: --diag-lora-targets must be a comma-separated canonical target list or all\n", __func__);
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-lora-values", value)) {
                if (value.empty()) {
                    value = take_next("--diag-lora-values");
                }
                if (!parse_int_arg(value, opts.diag_lora_values, "--diag-lora-values")) {
                    return false;
                }
                continue;
            }

            if (split_arg(arg, "--diag-lora-pattern", value)) {
                if (value.empty()) {
                    value = take_next("--diag-lora-pattern");
                }
                std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                    return (char) std::tolower(c);
                });
                opts.diag_lora_pattern = value;
                continue;
            }

            if (strip_device_args) {
                if (arg == "-dev" || arg == "--device") {
                    (void) take_next(arg.c_str());
                    continue;
                }
                if (arg.compare(0, std::strlen("--device="), "--device=") == 0) {
                    continue;
                }
            }

            filtered_argv.push_back(argv[i]);
        } catch (const std::exception & e) {
            LOG_ERR("%s: %s\n", __func__, e.what());
            return false;
        }
    }

    return true;
}

static void sst2_validate_options(const sst2_options & opts) {
    if (opts.steps <= 0) {
        throw std::runtime_error("steps must be > 0");
    }
    if (opts.eval_step == 0 || opts.eval_step < -1) {
        throw std::runtime_error("eval-step must be > 0 or -1 to disable validation");
    }
    if (opts.batch_size <= 0) {
        throw std::runtime_error("batch-size must be > 0");
    }
    if (opts.seq_len < 4) {
        throw std::runtime_error("seq-len must be >= 4");
    }
    if (opts.max_train <= 0 || opts.max_eval <= 0) {
        throw std::runtime_error("max-train and max-eval must be > 0");
    }
    if (!(opts.epsilon > 0.0f)) {
        throw std::runtime_error("epsilon must be > 0");
    }
    if (!(opts.lr > 0.0f)) {
        throw std::runtime_error("lr must be > 0");
    }
    if (opts.diag_layerwise < 0) {
        throw std::runtime_error("diag-layerwise must be >= 0");
    }
    if (opts.diag_layerwise_values <= 0) {
        throw std::runtime_error("diag-layerwise-values must be > 0");
    }
    if (opts.diag_lora_perturb < 0) {
        throw std::runtime_error("diag-lora-perturb must be >= 0");
    }
    if (opts.diag_lora_a < 0) {
        throw std::runtime_error("diag-lora-a must be >= 0");
    }
    if (opts.diag_lora_branch < 0) {
        throw std::runtime_error("diag-lora-branch must be >= 0");
    }
    if (opts.diag_lora_values <= 0) {
        throw std::runtime_error("diag-lora-values must be > 0");
    }
    if (opts.diag_lora_pattern != "noise" && opts.diag_lora_pattern != "ramp" &&
            opts.diag_lora_pattern != "constant") {
        throw std::runtime_error("diag-lora-pattern must be one of: noise, ramp, constant");
    }
    if (opts.diag_lora_propagation && opts.diag_lora_branch <= 0) {
        throw std::runtime_error("--diag-lora-propagation requires --diag-lora-branch > 0");
    }
    if (opts.diag_attn_internals && opts.diag_lora_branch <= 0) {
        throw std::runtime_error("--diag-attn-internals requires --diag-lora-branch > 0");
    }
    if (opts.diag_attn_internals && opts.diag_attn_heads.empty()) {
        throw std::runtime_error("--diag-attn-heads must be non-empty when --diag-attn-internals is true");
    }
    if (opts.diag_attn_values <= 0) {
        throw std::runtime_error("diag-attn-values must be > 0");
    }
    if (opts.pipeline && opts.mode != zo_mode::COOP) {
        throw std::runtime_error("--pipeline true requires --mode coop");
    }
    if (opts.antithetic && (opts.mode != zo_mode::COOP || opts.lora_exec != sst2_lora_exec_mode::FUSED_HTP)) {
        throw std::runtime_error("--antithetic true requires --mode coop --lora-exec fused-htp");
    }
    if (opts.lora_exec == sst2_lora_exec_mode::FUSED_HTP && opts.mode != zo_mode::COOP) {
        throw std::runtime_error("--lora-exec fused-htp requires --mode coop");
    }
    if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK && opts.pipeline) {
        throw std::runtime_error("--lora-exec merge-repack currently supports --pipeline false only");
    }
    if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK && opts.antithetic) {
        throw std::runtime_error("--lora-exec merge-repack does not support --antithetic true");
    }
}

static std::vector<llama_token> sst2_tokenize(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special) {
    int n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, false);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }
    std::vector<llama_token> tokens((size_t) n_tokens);
    const int ret = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), add_special, false);
    if (ret < 0) {
        throw std::runtime_error("tokenization failed for: " + text.substr(0, 80));
    }
    tokens.resize((size_t) ret);
    return tokens;
}

static std::string sst2_trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\r' || s.back() == '\n' || s.back() == '\t')) {
        s.pop_back();
    }
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\r' || s[start] == '\n' || s[start] == '\t')) {
        ++start;
    }
    if (start > 0) {
        s.erase(0, start);
    }
    return s;
}

static bool sst2_parse_label(const std::string & s, int & label) {
    try {
        size_t end = 0;
        const int v = std::stoi(s, &end);
        if (v != 0 && v != 1) {
            return false;
        }
        label = v;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

static std::vector<sst2_sample> sst2_load_tsv(
        const char * path,
        const llama_vocab * vocab,
        int max_samples,
        int max_len,
        const std::vector<std::vector<llama_token>> & label_suffixes) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error(std::string("cannot open SST2 TSV: ") + path);
    }

    std::vector<sst2_sample> samples;
    samples.reserve((size_t) max_samples);

    if (label_suffixes.size() != 2) {
        throw std::runtime_error("SST2 expects exactly two label suffixes");
    }
    for (size_t cls = 0; cls < label_suffixes.size(); ++cls) {
        const auto & label_suffix = label_suffixes[cls];
        const int prompt_suffix_tokens = (int) label_suffix.size() - 1;
        if (label_suffix.empty() || prompt_suffix_tokens >= max_len) {
            throw std::runtime_error(
                    "invalid SST2 label suffix for seq-len, class=" + std::to_string(cls));
        }
    }

    std::string line;
    bool first = true;
    while (std::getline(fin, line) && (int) samples.size() < max_samples) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        const size_t tab = line.rfind('\t');
        if (tab == std::string::npos) {
            continue;
        }

        std::string sentence = sst2_trim(line.substr(0, tab));
        std::string label_s  = sst2_trim(line.substr(tab + 1));
        int label = 0;
        if (!sst2_parse_label(label_s, label)) {
            if (first) {
                first = false;
                continue;
            }
            continue;
        }
        first = false;

        const std::vector<llama_token> & label_suffix = label_suffixes[(size_t) label];
        const int prompt_suffix_tokens = (int) label_suffix.size() - 1;
        std::vector<llama_token> sentence_tokens = sst2_tokenize(vocab, sentence, true);
        const int max_sentence_tokens = std::max(1, max_len - prompt_suffix_tokens);
        if ((int) sentence_tokens.size() > max_sentence_tokens) {
            sentence_tokens.resize((size_t) max_sentence_tokens);
        }

        sst2_sample sample;
        sample.input_ids = std::move(sentence_tokens);
        sample.input_ids.insert(sample.input_ids.end(), label_suffix.begin(), label_suffix.end() - 1);
        sample.label = label;
        sample.target_token = label_suffix.back();
        if (!sample.input_ids.empty() && (int) sample.input_ids.size() <= max_len) {
            samples.emplace_back(std::move(sample));
        }
    }

    if (samples.empty()) {
        throw std::runtime_error(std::string("no valid SST2 samples loaded from: ") + path);
    }

    return samples;
}

static sst2_batch_data sst2_make_train_batch(
        const std::vector<sst2_sample> & train,
        int batch_size,
        int step) {
    sst2_batch_data data;
    data.samples.reserve((size_t) batch_size);
    data.sample_indices.reserve((size_t) batch_size);
    std::mt19937 rng(derive_step_seed(DEFAULT_SEED, step, 0x44415441U));
    std::uniform_int_distribution<size_t> dist(0, train.size() - 1);
    for (int i = 0; i < batch_size; ++i) {
        const size_t idx = dist(rng);
        data.samples.push_back(&train[idx]);
        data.sample_indices.push_back(idx);
    }
    return data;
}

static sst2_batch_data sst2_make_eval_batch(
        const std::vector<sst2_sample> & eval,
        size_t start,
        int batch_size) {
    sst2_batch_data data;
    const size_t end = std::min(eval.size(), start + (size_t) batch_size);
    data.samples.reserve(end - start);
    data.sample_indices.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
        data.samples.push_back(&eval[i]);
        data.sample_indices.push_back(i);
    }
    return data;
}

static std::vector<llama_token> sst2_strip_tokenizer_prefix(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens = sst2_tokenize(vocab, text, true);
    const std::vector<llama_token> prefix = sst2_tokenize(vocab, "", true);
    if (!prefix.empty() && tokens.size() >= prefix.size() &&
            std::equal(prefix.begin(), prefix.end(), tokens.begin())) {
        tokens.erase(tokens.begin(), tokens.begin() + (std::ptrdiff_t) prefix.size());
    }
    return tokens;
}

static std::string sst2_token_piece(const llama_vocab * vocab, llama_token token) {
    if (vocab == nullptr || token < 0) {
        return std::string();
    }
    std::string piece = common_token_to_piece(vocab, token, true);
    for (char & c : piece) {
        if (c == '\n' || c == '\r' || c == '\t') {
            c = ' ';
        }
    }
    return piece;
}

static void sst2_print_token_list(
        const char * label,
        const std::vector<llama_token> & tokens,
        const llama_vocab * vocab,
        size_t limit = 32) {
    std::ostringstream os;
    const size_t n = std::min(tokens.size(), limit);
    os << label << " count=" << tokens.size() << " ids=[";
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            os << ",";
        }
        os << tokens[i];
    }
    if (tokens.size() > n) {
        os << ",...";
    }
    os << "]";
    if (!tokens.empty() && vocab != nullptr) {
        os << " decoded=\"";
        for (size_t i = 0; i < n; ++i) {
            os << sst2_token_piece(vocab, tokens[i]);
        }
        if (tokens.size() > n) {
            os << "...";
        }
        os << "\"";
    }
    LOG("%s\n", os.str().c_str());
}

static void sst2_print_float_stats(const std::string & prefix, const float * data, size_t size, size_t limit = 8) {
    if (data == nullptr || size == 0) {
        LOG("%s size=%zu data=<null-or-empty>\n", prefix.c_str(), size);
        return;
    }
    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    float max_abs = 0.0f;
    size_t finite_count = 0;
    for (size_t i = 0; i < size; ++i) {
        const float value = data[i];
        if (!std::isfinite(value)) {
            continue;
        }
        const float abs_value = std::fabs(value);
        sum += (double) value;
        sum_abs += (double) abs_value;
        sum_sq += (double) value * (double) value;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        max_abs = std::max(max_abs, abs_value);
        ++finite_count;
    }
    const double denom = finite_count > 0 ? (double) finite_count : 1.0;
    std::ostringstream os;
    os << prefix
       << " size=" << size
       << " finite=" << finite_count
       << " mean=" << (sum / denom)
       << " mean_abs=" << (sum_abs / denom)
       << " rms=" << std::sqrt(sum_sq / denom)
       << " min=" << (finite_count > 0 ? min_value : 0.0f)
       << " max=" << (finite_count > 0 ? max_value : 0.0f)
       << " max_abs=" << max_abs
       << " first=[";
    const size_t n = std::min(size, limit);
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            os << ",";
        }
        os << data[i];
    }
    if (size > n) {
        os << ",...";
    }
    os << "]";
    LOG("%s\n", os.str().c_str());
}

static void sst2_print_activation_sample_debug(
        const char * framework,
        const char * tag,
        int sample_ordinal,
        size_t data_index,
        const sst2_sample & sample) {
    std::ostringstream os;
    os << "activation_sample_debug"
       << " framework=" << framework
       << " tag=" << tag
       << " sample_ordinal=" << sample_ordinal
       << " data_index=" << data_index
       << " label=" << sample.label
       << " target=" << sample.target_token
       << " final_pos=" << (sample.input_ids.empty() ? -1 : (int) sample.input_ids.size() - 1)
       << " token_count=" << sample.input_ids.size()
       << " tokens=[";
    for (size_t i = 0; i < sample.input_ids.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << sample.input_ids[i];
    }
    os << "]";
    LOG("%s\n", os.str().c_str());
}

static bool sst2_layerwise_layer_selected(const sst2_layerwise_debug_state & state, int layer) {
    return state.layers.empty() ||
           std::find(state.layers.begin(), state.layers.end(), layer) != state.layers.end();
}

static bool sst2_parse_node_layer_suffix(const std::string & name, const std::string & prefix, int & layer) {
    if (name.rfind(prefix, 0) != 0) {
        return false;
    }
    const size_t pos = prefix.size();
    if (pos >= name.size() || !std::isdigit((unsigned char) name[pos])) {
        return false;
    }
    char * end = nullptr;
    const long parsed = std::strtol(name.c_str() + pos, &end, 10);
    if (end == name.c_str() + pos || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    layer = (int) parsed;
    return true;
}

static bool sst2_match_layerwise_node(
        const sst2_layerwise_debug_state & state,
        const std::string & name,
        int & layer,
        std::string & point) {
    if (name == "result_norm") {
        layer = -1;
        point = "final_norm";
        return true;
    }
    if (name == "result_output") {
        layer = -1;
        point = "lm_head";
        return true;
    }

    struct target {
        const char * prefix;
        const char * point;
    };
    static const target targets[] = {
        { "attn_norm-", "attn_norm" },
        { "Qcur-",      "q_raw" },
        { "Kcur-",      "k_raw" },
        { "Vcur-",      "v_raw" },
        { "attn_out-",  "attn_out" },
        { "ffn_inp-",   "attn_resid" },
        { "ffn_norm-",  "ffn_norm" },
        { "ffn_gate-",  "gate" },
        { "ffn_up-",    "up" },
        { "ffn_out-",   "down" },
        { "l_out-",     "layer_out" },
    };
    for (const target & t : targets) {
        int local_layer = -1;
        if (sst2_parse_node_layer_suffix(name, t.prefix, local_layer) &&
                sst2_layerwise_layer_selected(state, local_layer)) {
            layer = local_layer;
            point = t.point;
            return true;
        }
    }
    return false;
}

struct sst2_tensor_value_stats {
    int64_t count = 0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    float min_value = 0.0f;
    float max_value = 0.0f;
    std::vector<float> first_values;
};

static void sst2_update_tensor_stats(sst2_tensor_value_stats & stats, float value, int max_values) {
    if (stats.count == 0) {
        stats.min_value = value;
        stats.max_value = value;
    } else {
        stats.min_value = std::min(stats.min_value, value);
        stats.max_value = std::max(stats.max_value, value);
    }
    const float abs_value = std::fabs(value);
    stats.sum_abs += (double) abs_value;
    stats.sum_sq += (double) value * (double) value;
    stats.max_abs = std::max(stats.max_abs, abs_value);
    if ((int) stats.first_values.size() < max_values) {
        stats.first_values.push_back(value);
    }
    ++stats.count;
}

static void sst2_append_stats_fields(std::ostringstream & os, const char * prefix, const sst2_tensor_value_stats & stats) {
    if (stats.count <= 0) {
        os << " " << prefix << "_count=0"
           << " " << prefix << "_mean_abs=nan"
           << " " << prefix << "_rms=nan"
           << " " << prefix << "_max_abs=nan"
           << " " << prefix << "_min=nan"
           << " " << prefix << "_max=nan";
        return;
    }
    const double denom = (double) stats.count;
    os << " " << prefix << "_count=" << stats.count
       << " " << prefix << "_mean_abs=" << (stats.sum_abs / denom)
       << " " << prefix << "_rms=" << std::sqrt(stats.sum_sq / denom)
       << " " << prefix << "_max_abs=" << stats.max_abs
       << " " << prefix << "_min=" << stats.min_value
       << " " << prefix << "_max=" << stats.max_value;
}

static void sst2_append_first_values(std::ostringstream & os, const char * name, const std::vector<float> & values) {
    os << " " << name << "=[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
}

static int sst2_choose_ggml_token_dim(const ggml_tensor * tensor, int token_count) {
    if (tensor == nullptr || token_count <= 0) {
        return -1;
    }
    const int preferred[] = { 1, 2, 3, 0 };
    for (int dim : preferred) {
        if (dim >= 0 && dim < GGML_MAX_DIMS && tensor->ne[dim] == token_count) {
            return dim;
        }
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim] == token_count) {
            return dim;
        }
    }
    return -1;
}

static bool sst2_read_ggml_scalar(const uint8_t * data, size_t data_size, enum ggml_type type, size_t offset, float & value) {
    if (type == GGML_TYPE_F32) {
        if (offset + sizeof(float) > data_size) {
            return false;
        }
        std::memcpy(&value, data + offset, sizeof(float));
        return true;
    }
    if (type == GGML_TYPE_F16) {
        if (offset + sizeof(ggml_fp16_t) > data_size) {
            return false;
        }
        ggml_fp16_t raw;
        std::memcpy(&raw, data + offset, sizeof(raw));
        value = ggml_fp16_to_fp32(raw);
        return true;
    }
    if (type == GGML_TYPE_BF16) {
        if (offset + sizeof(ggml_bf16_t) > data_size) {
            return false;
        }
        ggml_bf16_t raw;
        std::memcpy(&raw, data + offset, sizeof(raw));
        value = ggml_bf16_to_fp32(raw);
        return true;
    }
    return false;
}

static std::string sst2_ggml_shape_string(const ggml_tensor * tensor) {
    std::ostringstream os;
    os << "[";
    if (tensor != nullptr) {
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (i > 0) {
                os << ",";
            }
            os << tensor->ne[i];
        }
    }
    os << "]";
    return os.str();
}

static void sst2_print_layerwise_tensor(
        const sst2_layerwise_debug_state & state,
        const ggml_tensor * tensor,
        int layer,
        const std::string & point) {
    const std::string name = tensor != nullptr ? ggml_get_name(tensor) : "<null>";
    std::ostringstream os;
    os << "activation_debug"
       << " framework=llama"
       << " tag=" << state.tag
       << " forward_id=" << state.forward_id
       << " sample_ordinal=" << state.sample_ordinal
       << " data_index=" << state.data_index
       << " label=" << state.label
       << " target=" << state.target_token
       << " final_pos=" << state.final_pos
       << " layer=" << layer
       << " point=" << point
       << " name=\"" << name << "\"";

    if (tensor == nullptr) {
        os << " tensor=null";
        LOG("%s\n", os.str().c_str());
        return;
    }

    const size_t n_bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> readback;
    const uint8_t * data = nullptr;
    size_t data_size = n_bytes;
    if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer) && tensor->data != nullptr) {
        data = (const uint8_t *) tensor->data;
    } else {
        readback.resize(n_bytes);
        if (n_bytes > 0) {
            ggml_backend_tensor_get(tensor, readback.data(), 0, n_bytes);
        }
        data = readback.data();
    }

    os << " shape=" << sst2_ggml_shape_string(tensor)
       << " dtype=" << ggml_type_name(tensor->type)
       << " elements=" << ggml_nelements(tensor)
       << " seq_len=" << state.token_count;

    if (data == nullptr || n_bytes == 0) {
        os << " stats=empty";
        LOG("%s\n", os.str().c_str());
        return;
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
        os << " stats=unsupported_dtype";
        LOG("%s\n", os.str().c_str());
        return;
    }

    sst2_tensor_value_stats full_stats;
    sst2_tensor_value_stats vec_stats;
    const int selected_dim = sst2_choose_ggml_token_dim(tensor, state.token_count);
    int selected_token = selected_dim >= 0 ? state.final_pos : -1;
    if (selected_dim >= 0 && (selected_token < 0 || selected_token >= tensor->ne[selected_dim])) {
        selected_token = 0;
    }
    const int64_t n_elements = ggml_nelements(tensor);
    for (int64_t i = 0; i < n_elements; ++i) {
        int64_t rest = i;
        size_t offset = 0;
        int selected_coord = -1;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            const int64_t ne = std::max<int64_t>(tensor->ne[dim], 1);
            const int64_t coord = rest % ne;
            rest /= ne;
            offset += (size_t) coord * tensor->nb[dim];
            if (dim == selected_dim) {
                selected_coord = (int) coord;
            }
        }
        float value = 0.0f;
        if (!sst2_read_ggml_scalar(data, data_size, tensor->type, offset, value)) {
            continue;
        }
        sst2_update_tensor_stats(full_stats, value, state.max_values);
        if (selected_dim < 0 || selected_coord == selected_token) {
            sst2_update_tensor_stats(vec_stats, value, state.max_values);
        }
    }

    os << " selected_dim=" << selected_dim
       << " selected_token=" << selected_token;
    sst2_append_stats_fields(os, "full", full_stats);
    sst2_append_stats_fields(os, "vec", vec_stats);
    sst2_append_first_values(os, "vec_first", vec_stats.first_values);
    LOG("%s\n", os.str().c_str());
}

static bool sst2_layerwise_eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = (sst2_layerwise_debug_state *) user_data;
    if (state == nullptr || !state->enabled || !state->active || tensor == nullptr) {
        return ask ? false : true;
    }
    int layer = -1;
    std::string point;
    const std::string name = ggml_get_name(tensor);
    if (!sst2_match_layerwise_node(*state, name, layer, point)) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    sst2_print_layerwise_tensor(*state, tensor, layer, point);
    return true;
}

static bool sst2_parse_lora_target_layer_suffix(const std::string & canonical_key, int & layer, std::string & suffix) {
    static const std::string prefix = "blk.";
    if (canonical_key.rfind(prefix, 0) != 0) {
        return false;
    }
    const size_t pos = prefix.size();
    if (pos >= canonical_key.size() || !std::isdigit((unsigned char) canonical_key[pos])) {
        return false;
    }
    char * end = nullptr;
    const long parsed = std::strtol(canonical_key.c_str() + pos, &end, 10);
    if (end == canonical_key.c_str() + pos || *end != '.' ||
            parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    layer = (int) parsed;
    suffix = end + 1;
    return !suffix.empty();
}

static bool sst2_is_attention_lora_suffix(const std::string & suffix) {
    return suffix == "attn_q.weight" ||
           suffix == "attn_k.weight" ||
           suffix == "attn_v.weight";
}

static bool sst2_match_lora_branch_node(
        const sst2_lora_branch_debug_state & state,
        const std::string & name,
        std::string & point) {
    struct target {
        const char * prefix;
        const char * point;
    };
    static const target targets[] = {
        {"lora_base:",       "add_base"},
        {"lora_xA_raw:",     "matmul_A_raw"},
        {"lora_xA_scaled:",  "matmul_A_scaled"},
        {"lora_xAB_scaled:", "matmul_B_scaled"},
        {"lora_add_out:",    "add_out"},
    };
    for (const auto & t : targets) {
        if (name.rfind(t.prefix, 0) == 0 && name.find(state.selected_key, std::strlen(t.prefix)) != std::string::npos) {
            point = t.point;
            return true;
        }
    }
    if (state.propagation || state.attn_internals) {
        int target_layer = -1;
        std::string suffix;
        if (!sst2_parse_lora_target_layer_suffix(state.selected_key, target_layer, suffix) ||
                !sst2_is_attention_lora_suffix(suffix)) {
            return false;
        }
        int node_layer = -1;
        if (state.attn_internals) {
            if (sst2_parse_node_layer_suffix(name, "kq-", node_layer) && node_layer == target_layer) {
                point = "qk_raw";
                return true;
            }
            if (sst2_parse_node_layer_suffix(name, "kq_soft_max-", node_layer) && node_layer == target_layer) {
                point = "attn_prob";
                return true;
            }
            if (sst2_parse_node_layer_suffix(name, "kqv-", node_layer) && node_layer == target_layer) {
                point = "qkv_context_packed";
                return true;
            }
            if (sst2_parse_node_layer_suffix(name, "attn_out-", node_layer) && node_layer == target_layer) {
                point = "attn_context_out";
                return true;
            }
        }
        if (sst2_parse_node_layer_suffix(name, "Qcur-", node_layer) && node_layer == target_layer) {
            point = "q_rope";
            return true;
        }
        if (sst2_parse_node_layer_suffix(name, "Kcur-", node_layer) && node_layer == target_layer) {
            point = "k_rope";
            return true;
        }
        if (sst2_parse_node_layer_suffix(name, "Vcur-", node_layer) && node_layer == target_layer) {
            point = "v_block_reshape";
            return true;
        }
        if (sst2_parse_node_layer_suffix(name, "attn_out-", node_layer) && node_layer == target_layer) {
            point = "attn_context";
            return true;
        }
    }
    return false;
}

static void sst2_print_lora_branch_tensor(
        const sst2_lora_branch_debug_state & state,
        const ggml_tensor * tensor,
        const std::string & point) {
    const std::string name = tensor != nullptr ? ggml_get_name(tensor) : "<null>";
    std::ostringstream os;
    os << "diag_lora_branch_debug"
       << " framework=llama"
       << " tag=" << state.tag
       << " forward_id=" << state.forward_id
       << " sample_ordinal=" << state.sample_ordinal
       << " data_index=" << state.data_index
       << " label=" << state.label
       << " target=" << state.target_token
       << " final_pos=" << state.final_pos
       << " canonical_key=" << state.selected_key
       << " point=" << point
       << " name=\"" << name << "\"";

    if (tensor == nullptr) {
        os << " tensor=null";
        LOG("%s\n", os.str().c_str());
        return;
    }

    const size_t n_bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> readback;
    const uint8_t * data = nullptr;
    size_t data_size = n_bytes;
    if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer) && tensor->data != nullptr) {
        data = (const uint8_t *) tensor->data;
    } else {
        readback.resize(n_bytes);
        if (n_bytes > 0) {
            ggml_backend_tensor_get(tensor, readback.data(), 0, n_bytes);
        }
        data = readback.data();
    }

    os << " shape=" << sst2_ggml_shape_string(tensor)
       << " dtype=" << ggml_type_name(tensor->type)
       << " elements=" << ggml_nelements(tensor)
       << " seq_len=" << state.token_count;

    if (data == nullptr || n_bytes == 0) {
        os << " stats=empty";
        LOG("%s\n", os.str().c_str());
        return;
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
        os << " stats=unsupported_dtype";
        LOG("%s\n", os.str().c_str());
        return;
    }

    sst2_tensor_value_stats full_stats;
    sst2_tensor_value_stats vec_stats;
    const int selected_dim = sst2_choose_ggml_token_dim(tensor, state.token_count);
    int selected_token = selected_dim >= 0 ? state.final_pos : -1;
    if (selected_dim >= 0 && (selected_token < 0 || selected_token >= tensor->ne[selected_dim])) {
        selected_token = 0;
    }
    const int64_t n_elements = ggml_nelements(tensor);
    for (int64_t i = 0; i < n_elements; ++i) {
        int64_t rest = i;
        size_t offset = 0;
        int selected_coord = -1;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            const int64_t ne = std::max<int64_t>(tensor->ne[dim], 1);
            const int64_t coord = rest % ne;
            rest /= ne;
            offset += (size_t) coord * tensor->nb[dim];
            if (dim == selected_dim) {
                selected_coord = (int) coord;
            }
        }
        float value = 0.0f;
        if (!sst2_read_ggml_scalar(data, data_size, tensor->type, offset, value)) {
            continue;
        }
        sst2_update_tensor_stats(full_stats, value, state.max_values);
        if (selected_dim < 0 || selected_coord == selected_token) {
            sst2_update_tensor_stats(vec_stats, value, state.max_values);
        }
    }

    os << " selected_dim=" << selected_dim
       << " selected_token=" << selected_token;
    sst2_append_stats_fields(os, "full", full_stats);
    sst2_append_stats_fields(os, "vec", vec_stats);
    sst2_append_first_values(os, "vec_first", vec_stats.first_values);
    LOG("%s\n", os.str().c_str());
}

static bool sst2_is_attn_internal_point(const std::string & point) {
    return point == "qk_raw" ||
           point == "attn_prob" ||
           point == "qkv_context_packed" ||
           point == "attn_context_out";
}

static bool sst2_read_ggml_coords(
        const ggml_tensor * tensor,
        const uint8_t * data,
        size_t data_size,
        const int64_t coords[GGML_MAX_DIMS],
        float & value) {
    size_t offset = 0;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (coords[dim] < 0 || coords[dim] >= tensor->ne[dim]) {
            return false;
        }
        offset += (size_t) coords[dim] * tensor->nb[dim];
    }
    return sst2_read_ggml_scalar(data, data_size, tensor->type, offset, value);
}

static int sst2_find_token_dim_except(const ggml_tensor * tensor, int token_count, int excluded_dim) {
    if (tensor == nullptr || token_count <= 0) {
        return -1;
    }
    const int preferred[] = { 1, 2, 3, 0 };
    for (int dim : preferred) {
        if (dim != excluded_dim && dim >= 0 && dim < GGML_MAX_DIMS && tensor->ne[dim] == token_count) {
            return dim;
        }
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (dim != excluded_dim && tensor->ne[dim] == token_count) {
            return dim;
        }
    }
    return -1;
}

static int sst2_find_head_dim_except(const ggml_tensor * tensor, int head, int excluded_a, int excluded_b) {
    if (tensor == nullptr) {
        return -1;
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (dim == excluded_a || dim == excluded_b) {
            continue;
        }
        if (tensor->ne[dim] > head) {
            return dim;
        }
    }
    return -1;
}

static void sst2_print_attn_internal_tensor_for_head(
        const sst2_lora_branch_debug_state & state,
        const ggml_tensor * tensor,
        const std::string & point,
        int head) {
    const std::string name = tensor != nullptr ? ggml_get_name(tensor) : "<null>";
    int layer = -1;
    std::string suffix;
    (void) sst2_parse_lora_target_layer_suffix(state.selected_key, layer, suffix);

    std::ostringstream os;
    os << "attn_internal_debug"
       << " framework=llama"
       << " tag=" << state.tag
       << " forward_id=" << state.forward_id
       << " sample_ordinal=" << state.sample_ordinal
       << " data_index=" << state.data_index
       << " target=" << state.selected_key
       << " layer=" << layer
       << " head=" << head
       << " qpos=" << state.final_pos
       << " point=" << point
       << " name=\"" << name << "\"";

    if (tensor == nullptr) {
        os << " tensor=null";
        LOG("%s\n", os.str().c_str());
        return;
    }

    const size_t n_bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> readback;
    const uint8_t * data = nullptr;
    size_t data_size = n_bytes;
    if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer) && tensor->data != nullptr) {
        data = (const uint8_t *) tensor->data;
    } else {
        readback.resize(n_bytes);
        if (n_bytes > 0) {
            ggml_backend_tensor_get(tensor, readback.data(), 0, n_bytes);
        }
        data = readback.data();
    }

    os << " shape=" << sst2_ggml_shape_string(tensor)
       << " dtype=" << ggml_type_name(tensor->type)
       << " elements=" << ggml_nelements(tensor)
       << " seq_len=" << state.token_count;

    if (data == nullptr || n_bytes == 0) {
        os << " stats=empty";
        LOG("%s\n", os.str().c_str());
        return;
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
        os << " stats=unsupported_dtype";
        LOG("%s\n", os.str().c_str());
        return;
    }

    sst2_tensor_value_stats full_stats;
    const int64_t n_elements = ggml_nelements(tensor);
    for (int64_t i = 0; i < n_elements; ++i) {
        int64_t rest = i;
        size_t offset = 0;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            const int64_t ne = std::max<int64_t>(tensor->ne[dim], 1);
            const int64_t coord = rest % ne;
            rest /= ne;
            offset += (size_t) coord * tensor->nb[dim];
        }
        float value = 0.0f;
        if (sst2_read_ggml_scalar(data, data_size, tensor->type, offset, value)) {
            sst2_update_tensor_stats(full_stats, value, state.attn_values);
        }
    }

    sst2_tensor_value_stats vec_stats;
    bool extracted = false;
    if (point == "qk_raw" || point == "attn_prob") {
        const int kv_dim = 0;
        const int q_dim = sst2_find_token_dim_except(tensor, state.token_count, kv_dim);
        const int head_dim = sst2_find_head_dim_except(tensor, head, kv_dim, q_dim);
        if (q_dim >= 0 && head_dim >= 0 && state.final_pos >= 0 && state.final_pos < tensor->ne[q_dim]) {
            for (int64_t kv = 0; kv < tensor->ne[kv_dim]; ++kv) {
                int64_t coords[GGML_MAX_DIMS] = {0, 0, 0, 0};
                coords[kv_dim] = kv;
                coords[q_dim] = state.final_pos;
                coords[head_dim] = head;
                float value = 0.0f;
                if (sst2_read_ggml_coords(tensor, data, data_size, coords, value)) {
                    sst2_update_tensor_stats(vec_stats, value, state.attn_values);
                    extracted = true;
                }
            }
        }
        os << " kv_dim=" << kv_dim << " q_dim=" << q_dim << " head_dim=" << head_dim;
    } else if (point == "qkv_context_packed") {
        const int value_dim = 0;
        const int q_dim = sst2_find_token_dim_except(tensor, state.token_count, value_dim);
        const int head_dim = sst2_find_head_dim_except(tensor, head, value_dim, q_dim);
        if (q_dim >= 0 && head_dim >= 0 && state.final_pos >= 0 && state.final_pos < tensor->ne[q_dim]) {
            for (int64_t d = 0; d < tensor->ne[value_dim]; ++d) {
                int64_t coords[GGML_MAX_DIMS] = {0, 0, 0, 0};
                coords[value_dim] = d;
                coords[q_dim] = state.final_pos;
                coords[head_dim] = head;
                float value = 0.0f;
                if (sst2_read_ggml_coords(tensor, data, data_size, coords, value)) {
                    sst2_update_tensor_stats(vec_stats, value, state.attn_values);
                    extracted = true;
                }
            }
        }
        os << " value_dim=" << value_dim << " q_dim=" << q_dim << " head_dim=" << head_dim;
    } else {
        const int hidden_dim = 0;
        const int q_dim = sst2_find_token_dim_except(tensor, state.token_count, hidden_dim);
        int n_heads_hint = 32;
        if (!state.attn_heads.empty()) {
            n_heads_hint = std::max(n_heads_hint, *std::max_element(state.attn_heads.begin(), state.attn_heads.end()) + 1);
        }
        const int64_t head_dim_size = n_heads_hint > 0 ? tensor->ne[hidden_dim] / n_heads_hint : tensor->ne[hidden_dim];
        if (q_dim >= 0 && head_dim_size > 0 && state.final_pos >= 0 && state.final_pos < tensor->ne[q_dim]) {
            const int64_t begin = static_cast<int64_t>(head) * head_dim_size;
            const int64_t end = std::min<int64_t>(begin + head_dim_size, tensor->ne[hidden_dim]);
            for (int64_t d = begin; d < end; ++d) {
                int64_t coords[GGML_MAX_DIMS] = {0, 0, 0, 0};
                coords[hidden_dim] = d;
                coords[q_dim] = state.final_pos;
                float value = 0.0f;
                if (sst2_read_ggml_coords(tensor, data, data_size, coords, value)) {
                    sst2_update_tensor_stats(vec_stats, value, state.attn_values);
                    extracted = true;
                }
            }
        }
        os << " hidden_dim=" << hidden_dim << " q_dim=" << q_dim << " heads_hint=" << n_heads_hint;
    }

    if (!extracted) {
        vec_stats = full_stats;
        os << " vec_fallback=full";
    }
    sst2_append_stats_fields(os, "full", full_stats);
    sst2_append_stats_fields(os, "vec", vec_stats);
    sst2_append_first_values(os, "vec_first", vec_stats.first_values);
    LOG("%s\n", os.str().c_str());
}

static bool sst2_lora_branch_eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = (sst2_lora_branch_debug_state *) user_data;
    if (state == nullptr || !state->enabled || !state->active || tensor == nullptr) {
        return ask ? false : true;
    }
    std::string point;
    const std::string name = ggml_get_name(tensor);
    if (!sst2_match_lora_branch_node(*state, name, point)) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    if (sst2_is_attn_internal_point(point)) {
        for (int head : state->attn_heads) {
            sst2_print_attn_internal_tensor_for_head(*state, tensor, point, head);
            state->printed_ops++;
        }
        return true;
    }
    sst2_print_lora_branch_tensor(*state, tensor, point);
    state->printed_ops++;
    return true;
}

static void sst2_begin_lora_branch_debug(
        sst2_lora_branch_debug_state * state,
        const char * tag,
        const std::string & selected_key,
        int sample_ordinal,
        size_t data_index,
        const sst2_sample & sample) {
    if (state == nullptr || !state->enabled) {
        return;
    }
    state->active = true;
    state->tag = tag != nullptr ? tag : "";
    state->selected_key = selected_key;
    state->forward_id++;
    state->sample_ordinal = sample_ordinal;
    state->data_index = data_index;
    state->label = sample.label;
    state->target_token = sample.target_token;
    state->token_count = (int) sample.input_ids.size();
    state->final_pos = sample.input_ids.empty() ? -1 : (int) sample.input_ids.size() - 1;
    state->printed_ops = 0;
    LOG("diag_lora_branch_debug_begin framework=llama tag=%s forward_id=%d data_index=%zu sample_ordinal=%d canonical_key=%s propagation=%s attn_internals=%s seq_len=%d final_pos=%d label=%d target=%d\n",
            state->tag.c_str(),
            state->forward_id,
            state->data_index,
            state->sample_ordinal,
            state->selected_key.c_str(),
            state->propagation ? "true" : "false",
            state->attn_internals ? "true" : "false",
            state->token_count,
            state->final_pos,
            state->label,
            state->target_token);
}

static void sst2_end_lora_branch_debug(sst2_lora_branch_debug_state * state) {
    if (state == nullptr || !state->enabled || !state->active) {
        return;
    }
    LOG("diag_lora_branch_debug_end framework=llama tag=%s forward_id=%d printed_ops=%d\n",
            state->tag.c_str(), state->forward_id, state->printed_ops);
    state->active = false;
}

static bool sst2_should_print_b_detail(size_t index, size_t count, int detail_limit) {
    return (int) index < detail_limit || index + 1 == count;
}

static std::string sst2_lora_b_canonical_key(const std::string & name) {
    static const std::string suffix = ".lora_b";
    if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return name.substr(0, name.size() - suffix.size());
    }
    return name;
}

static const ggml_tensor * sst2_find_model_tensor(const llama_model * model, const std::string & name) {
    if (model == nullptr) {
        return nullptr;
    }
    for (const auto & entry : model->tensors_by_name) {
        if (entry.first == name) {
            return entry.second;
        }
    }
    return nullptr;
}

static bool sst2_model_has_htp_q4_lora_target(
        const llama_model * model,
        const std::vector<lora_b_tensor> & lora_b_tensors) {
    for (const auto & tensor : lora_b_tensors) {
        const std::string canonical_key = sst2_lora_b_canonical_key(tensor.name);
        const ggml_tensor * base = sst2_find_model_tensor(model, canonical_key);
        if (base != nullptr && base->type == GGML_TYPE_Q4_0 && lora_tensor_on_hexagon(base)) {
            return true;
        }
    }
    return false;
}

static const std::vector<float> * sst2_lora_b_values_for_source(const lora_b_tensor & tensor, const std::string & source) {
    if (source == "master") {
        return &tensor.master;
    }
    if (source == "work") {
        return &tensor.work;
    }
    if (source == "plus") {
        return &tensor.plus;
    }
    if (source == "minus") {
        return &tensor.minus;
    }
    if (source == "plus_rm") {
        return &tensor.plus_rm;
    }
    if (source == "minus_rm") {
        return &tensor.minus_rm;
    }
    return nullptr;
}

static bool sst2_lora_b_source_is_mnn_flat(const std::string & source) {
    return source == "plus_rm" || source == "minus_rm";
}

static size_t sst2_lora_b_ggml_index_for_mnn_flat(const lora_b_tensor & tensor, size_t mnn_flat_index) {
    const size_t rank = (size_t) tensor.tensor->ne[0];
    const size_t out  = (size_t) tensor.tensor->ne[1];
    const size_t r = mnn_flat_index / out;
    const size_t o = mnn_flat_index - r * out;
    return o * rank + r;
}

static size_t sst2_lora_b_mnn_index_for_ggml_flat(const lora_b_tensor & tensor, size_t ggml_flat_index) {
    const size_t rank = (size_t) tensor.tensor->ne[0];
    const size_t out  = (size_t) tensor.tensor->ne[1];
    const size_t o = ggml_flat_index / rank;
    const size_t r = ggml_flat_index - o * rank;
    return r * out + o;
}

static float sst2_lora_b_value_at_mnn_flat(
        const lora_b_tensor & tensor,
        const std::vector<float> & values,
        bool source_is_mnn_flat,
        size_t mnn_flat_index) {
    const size_t index = source_is_mnn_flat ? mnn_flat_index : sst2_lora_b_ggml_index_for_mnn_flat(tensor, mnn_flat_index);
    return index < values.size() ? values[index] : 0.0f;
}

static float sst2_lora_b_value_at_ggml_flat(
        const lora_b_tensor & tensor,
        const std::vector<float> & values,
        bool source_is_mnn_flat,
        size_t ggml_flat_index) {
    const size_t index = source_is_mnn_flat ? sst2_lora_b_mnn_index_for_ggml_flat(tensor, ggml_flat_index) : ggml_flat_index;
    return index < values.size() ? values[index] : 0.0f;
}

static uint32_t sst2_stable_string_hash32(const std::string & text) {
    uint32_t h = 2166136261U;
    for (unsigned char c : text) {
        h ^= (uint32_t) c;
        h *= 16777619U;
    }
    return h;
}

static uint32_t sst2_mix_index_hash32(uint32_t seed, size_t index) {
    uint64_t x = (uint64_t) seed;
    x += 0x9e3779b97f4a7c15ULL + (uint64_t) index * 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (uint32_t) x;
}

static float sst2_diag_lora_pattern_value(
        const std::string & canonical_key,
        const std::string & pattern,
        size_t mnn_flat_index,
        int64_t rank,
        int64_t out) {
    (void) rank;
    const size_t r = out > 0 ? mnn_flat_index / (size_t) out : 0;
    const size_t o = out > 0 ? mnn_flat_index - r * (size_t) out : mnn_flat_index;
    if (pattern == "constant") {
        return 1.0f;
    }
    if (pattern == "ramp") {
        return 0.1f * (float) (r + 1) + 0.00001f * (float) (o + 1);
    }
    const uint32_t stream = 0x4c4f5241U ^ sst2_stable_string_hash32(canonical_key);
    const uint32_t base = derive_step_seed(DEFAULT_SEED, 0, stream);
    const uint32_t bits = sst2_mix_index_hash32(base, mnn_flat_index);
    const float unit = ((float) (bits >> 8) + 0.5f) * (1.0f / 16777216.0f);
    return 2.0f * unit - 1.0f;
}

static int sst2_find_lora_b_by_canonical_key(const std::vector<lora_b_tensor> & tensors, const std::string & key) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (sst2_lora_b_canonical_key(tensors[i].name) == key) {
            return (int) i;
        }
    }
    return -1;
}

static void sst2_select_diag_lora_targets(
        const std::vector<lora_b_tensor> & tensors,
        const std::vector<std::string> & requested,
        const char * diag_name,
        std::vector<size_t> & selected_targets) {
    selected_targets.clear();
    if (requested.empty()) {
        selected_targets.reserve(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            selected_targets.push_back(i);
        }
    } else {
        for (const auto & key : requested) {
            const int index = sst2_find_lora_b_by_canonical_key(tensors, key);
            if (index < 0) {
                LOG("%s_missing_target framework=llama canonical_key=%s\n", diag_name, key.c_str());
                continue;
            }
            const size_t uindex = (size_t) index;
            if (std::find(selected_targets.begin(), selected_targets.end(), uindex) == selected_targets.end()) {
                selected_targets.push_back(uindex);
            }
        }
    }
    if (selected_targets.empty()) {
        throw std::runtime_error(std::string(diag_name) + " found no matching LoRA targets");
    }
}

static void sst2_print_lora_b_target_order_debug(
        const std::vector<lora_b_tensor> & tensors,
        int noise_step,
        int detail_limit = 12) {
    size_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & tensor = tensors[i];
        if (sst2_should_print_b_detail(i, tensors.size(), detail_limit)) {
            LOG("lora_b_target_order target_index=%zu tensor_name=%s canonical_key=%s shape_rank=%lld shape_out=%lld offset=%zu size=%lld noise_seed_step=%d noise_seed=%u\n",
                    i,
                    tensor.name.c_str(),
                    sst2_lora_b_canonical_key(tensor.name).c_str(),
                    (long long) tensor.tensor->ne[0],
                    (long long) tensor.tensor->ne[1],
                    offset,
                    (long long) tensor.n_elms,
                    noise_step,
                    (unsigned) derive_tensor_noise_seed(noise_step, i));
        }
        offset += (size_t) tensor.n_elms;
    }
}

static void sst2_print_lora_b_storage_debug(
        const std::vector<lora_b_tensor> & tensors,
        const char * tag,
        const char * source,
        int noise_step,
        int detail_limit = 6,
        size_t value_limit = 8) {
    const std::string source_name = source != nullptr ? source : "";
    const bool source_is_mnn_flat = sst2_lora_b_source_is_mnn_flat(source_name);
    const char * layout_name = source_is_mnn_flat ? "mnn" : "ggml";

    double total_sum_abs = 0.0;
    double total_sum_sq = 0.0;
    float total_max_abs = 0.0f;
    size_t total_count = 0;
    for (const auto & tensor : tensors) {
        const std::vector<float> * values = sst2_lora_b_values_for_source(tensor, source_name);
        if (values == nullptr) {
            continue;
        }
        for (float value : *values) {
            const float abs_value = std::fabs(value);
            total_sum_abs += (double) abs_value;
            total_sum_sq += (double) value * (double) value;
            total_max_abs = std::max(total_max_abs, abs_value);
            ++total_count;
        }
    }
    const double denom = total_count > 0 ? (double) total_count : 1.0;
    LOG("lora_b_debug_summary tag=%s source=%s layout=%s tensors=%zu values=%zu mean_abs=%.9g rms=%.9g max_abs=%.9g\n",
            tag != nullptr ? tag : "B",
            source_name.c_str(),
            layout_name,
            tensors.size(),
            total_count,
            total_sum_abs / denom,
            std::sqrt(total_sum_sq / denom),
            (double) total_max_abs);

    size_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & tensor = tensors[i];
        const std::vector<float> * values = sst2_lora_b_values_for_source(tensor, source_name);
        if (!sst2_should_print_b_detail(i, tensors.size(), detail_limit)) {
            offset += (size_t) tensor.n_elms;
            continue;
        }
        if (values == nullptr || values->size() != (size_t) tensor.n_elms) {
            LOG("lora_b_debug tag=%s source=%s target_index=%zu tensor_name=%s range=unavailable\n",
                    tag != nullptr ? tag : "B", source_name.c_str(), i, tensor.name.c_str());
            offset += (size_t) tensor.n_elms;
            continue;
        }

        double sum_abs = 0.0;
        double sum_sq = 0.0;
        float max_abs = 0.0f;
        for (float value : *values) {
            const float abs_value = std::fabs(value);
            sum_abs += (double) abs_value;
            sum_sq += (double) value * (double) value;
            max_abs = std::max(max_abs, abs_value);
        }
        const double tensor_denom = tensor.n_elms > 0 ? (double) tensor.n_elms : 1.0;

        std::ostringstream os;
        os << "lora_b_debug"
           << " tag=" << (tag != nullptr ? tag : "B")
           << " source=" << source_name
           << " layout=" << layout_name
           << " target_index=" << i
           << " tensor_name=" << tensor.name
           << " canonical_key=" << sst2_lora_b_canonical_key(tensor.name)
           << " shape_rank=" << tensor.tensor->ne[0]
           << " shape_out=" << tensor.tensor->ne[1]
           << " offset=" << offset
           << " size=" << tensor.n_elms
           << " noise_seed_step=" << noise_step
           << " noise_seed=" << derive_tensor_noise_seed(noise_step, i)
           << " mean_abs=" << (sum_abs / tensor_denom)
           << " rms=" << std::sqrt(sum_sq / tensor_denom)
           << " max_abs=" << max_abs
           << " first_storage=[";
        const size_t n = std::min((size_t) tensor.n_elms, value_limit);
        for (size_t j = 0; j < n; ++j) {
            if (j > 0) {
                os << ",";
            }
            os << (*values)[j];
        }
        os << "] first_mnn_flat=[";
        for (size_t j = 0; j < n; ++j) {
            if (j > 0) {
                os << ",";
            }
            os << sst2_lora_b_value_at_mnn_flat(tensor, *values, source_is_mnn_flat, j);
        }
        os << "] first_ggml_flat=[";
        for (size_t j = 0; j < n; ++j) {
            if (j > 0) {
                os << ",";
            }
            os << sst2_lora_b_value_at_ggml_flat(tensor, *values, source_is_mnn_flat, j);
        }
        os << "]";
        LOG("%s\n", os.str().c_str());

        offset += (size_t) tensor.n_elms;
    }
}

static void sst2_print_lora_b_target_debug(
        const lora_b_tensor & tensor,
        size_t target_index,
        const char * tag,
        const char * source,
        const std::string & pattern,
        float epsilon,
        int value_limit) {
    const std::string source_name = source != nullptr ? source : "";
    const std::vector<float> * values = sst2_lora_b_values_for_source(tensor, source_name);
    const bool source_is_mnn_flat = sst2_lora_b_source_is_mnn_flat(source_name);
    const char * layout_name = source_is_mnn_flat ? "mnn" : "ggml";
    if (values == nullptr || values->size() != (size_t) tensor.n_elms) {
        LOG("diag_lora_b_debug framework=llama tag=%s source=%s target_index=%zu tensor_name=%s range=unavailable\n",
                tag != nullptr ? tag : "B",
                source_name.c_str(),
                target_index,
                tensor.name.c_str());
        return;
    }

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (float value : *values) {
        const float abs_value = std::fabs(value);
        sum_abs += (double) abs_value;
        sum_sq += (double) value * (double) value;
        max_abs = std::max(max_abs, abs_value);
    }
    const double denom = tensor.n_elms > 0 ? (double) tensor.n_elms : 1.0;
    const size_t n = std::min((size_t) tensor.n_elms, (size_t) std::max(0, value_limit));

    std::ostringstream os;
    os << "diag_lora_b_debug"
       << " framework=llama"
       << " tag=" << (tag != nullptr ? tag : "B")
       << " source=" << source_name
       << " layout=" << layout_name
       << " pattern=" << pattern
       << " epsilon=" << epsilon
       << " target_index=" << target_index
       << " tensor_name=" << tensor.name
       << " canonical_key=" << sst2_lora_b_canonical_key(tensor.name)
       << " shape_rank=" << tensor.tensor->ne[0]
       << " shape_out=" << tensor.tensor->ne[1]
       << " size=" << tensor.n_elms
       << " mean_abs=" << (sum_abs / denom)
       << " rms=" << std::sqrt(sum_sq / denom)
       << " max_abs=" << max_abs
       << " first_mnn_flat=[";
    for (size_t j = 0; j < n; ++j) {
        if (j > 0) {
            os << ",";
        }
        os << sst2_lora_b_value_at_mnn_flat(tensor, *values, source_is_mnn_flat, j);
    }
    os << "] first_storage=[";
    for (size_t j = 0; j < n; ++j) {
        if (j > 0) {
            os << ",";
        }
        os << (*values)[j];
    }
    os << "] first_ggml_flat=[";
    for (size_t j = 0; j < n; ++j) {
        if (j > 0) {
            os << ",";
        }
        os << sst2_lora_b_value_at_ggml_flat(tensor, *values, source_is_mnn_flat, j);
    }
    os << "]";
    LOG("%s\n", os.str().c_str());
}

static bool sst2_read_tensor_values(const ggml_tensor * tensor, std::vector<float> & values) {
    if (tensor == nullptr) {
        return false;
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        return false;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    if (!bytes.empty()) {
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    }
    values.assign((size_t) ggml_nelements(tensor), 0.0f);
    deserialize_tensor_to_float(bytes, tensor->type, values);
    return true;
}

static void sst2_print_lora_a_debug(
        const llama_adapter_lora * adapter,
        const lora_b_tensor & b_tensor,
        size_t target_index,
        int sample_ordinal,
        size_t data_index,
        int value_limit) {
    const std::string canonical_key = sst2_lora_b_canonical_key(b_tensor.name);
    auto it = adapter->ab_map.find(canonical_key);
    if (it == adapter->ab_map.end() || it->second.a == nullptr || it->second.b == nullptr) {
        LOG("diag_lora_a_debug framework=llama sample_ordinal=%d data_index=%zu target_index=%zu canonical_key=%s status=missing_adapter_weight\n",
                sample_ordinal, data_index, target_index, canonical_key.c_str());
        return;
    }

    const llama_adapter_lora_weight & lw = it->second;
    std::vector<float> values;
    if (!sst2_read_tensor_values(lw.a, values)) {
        LOG("diag_lora_a_debug framework=llama sample_ordinal=%d data_index=%zu target_index=%zu canonical_key=%s status=readback_failed\n",
                sample_ordinal, data_index, target_index, canonical_key.c_str());
        return;
    }

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (float value : values) {
        const float abs_value = std::fabs(value);
        sum_abs += (double) abs_value;
        sum_sq += (double) value * (double) value;
        max_abs = std::max(max_abs, abs_value);
    }
    const double denom = values.empty() ? 1.0 : (double) values.size();
    const int64_t input_size = lw.a->ne[0];
    const int64_t rank = lw.a->ne[1];
    const float runtime_scale = lw.get_scale(adapter->alpha, 1.0f);
    const size_t n = std::min(values.size(), (size_t) std::max(0, value_limit));

    std::ostringstream os;
    os << "diag_lora_a_debug"
       << " framework=llama"
       << " sample_ordinal=" << sample_ordinal
       << " data_index=" << data_index
       << " target_index=" << target_index
       << " tensor_name=" << ggml_get_name(lw.a)
       << " canonical_key=" << canonical_key
       << " input_size=" << input_size
       << " output_size=" << lw.b->ne[1]
       << " rank=" << rank
       << " alpha=" << adapter->alpha
       << " runtime_scale=" << runtime_scale
       << " shape=" << sst2_ggml_shape_string(lw.a)
       << " dtype=" << ggml_type_name(lw.a->type)
       << " mean_abs=" << (sum_abs / denom)
       << " rms=" << std::sqrt(sum_sq / denom)
       << " max_abs=" << max_abs
       << " first_ggml_flat=[";
    for (size_t j = 0; j < n; ++j) {
        if (j > 0) {
            os << ",";
        }
        os << values[j];
    }
    os << "] first_mnn_scaled_flat=[";
    for (size_t j = 0; j < n; ++j) {
        if (j > 0) {
            os << ",";
        }
        const size_t in = rank > 0 ? j / (size_t) rank : j;
        const size_t r = rank > 0 ? j - in * (size_t) rank : 0;
        const size_t ggml_index = r * (size_t) input_size + in;
        os << (ggml_index < values.size() ? values[ggml_index] * runtime_scale : 0.0f);
    }
    os << "]";
    LOG("%s\n", os.str().c_str());
}

static void sst2_print_tokenizer_diagnostics(
        const llama_vocab * vocab,
        const std::vector<llama_token> & class_tokens,
        const std::vector<sst2_sample> & train_data) {
    sst2_print_token_list("tokenizer_prefix_empty", sst2_tokenize(vocab, "", true), vocab);
    sst2_print_token_list("suffix_raw_It_was", sst2_tokenize(vocab, "It was", true), vocab);
    sst2_print_token_list("suffix_stripped_It_was", sst2_strip_tokenizer_prefix(vocab, "It was"), vocab);
    sst2_print_token_list("suffix_stripped_leading_space_It_was", sst2_strip_tokenizer_prefix(vocab, " It was"), vocab);
    sst2_print_token_list("negative_word_stripped", sst2_strip_tokenizer_prefix(vocab, "terrible"), vocab);
    sst2_print_token_list("negative_word_leading_space_stripped", sst2_strip_tokenizer_prefix(vocab, " terrible"), vocab);
    sst2_print_token_list("negative_fallback_stripped", sst2_strip_tokenizer_prefix(vocab, "It was terrible"), vocab);
    sst2_print_token_list("positive_word_stripped", sst2_strip_tokenizer_prefix(vocab, "great"), vocab);
    sst2_print_token_list("positive_word_leading_space_stripped", sst2_strip_tokenizer_prefix(vocab, " great"), vocab);
    sst2_print_token_list("positive_fallback_stripped", sst2_strip_tokenizer_prefix(vocab, "It was great"), vocab);
    LOG("class_token_negative_decode=\"%s\" class_token_positive_decode=\"%s\"\n",
            sst2_token_piece(vocab, class_tokens[0]).c_str(),
            sst2_token_piece(vocab, class_tokens[1]).c_str());
    if (!train_data.empty()) {
        const sst2_sample & sample = train_data[0];
        LOG("first_train_sample label=%d target_token=%d target_decode=\"%s\"\n",
                sample.label,
                sample.target_token,
                sst2_token_piece(vocab, sample.target_token).c_str());
        sst2_print_token_list("first_train_tokens", sample.input_ids, vocab);
    }
}

static void sst2_print_sample_debug(const sst2_debug_state & debug, const sst2_batch_data & data, size_t batch_pos) {
    const sst2_sample * sample = data.samples[batch_pos];
    const size_t data_index = batch_pos < data.sample_indices.size() ? data.sample_indices[batch_pos] : (size_t) -1;
    const llama_token last_token = sample->input_ids.empty() ? LLAMA_TOKEN_NULL : sample->input_ids.back();
    LOG("debug_llama_sample tag=%s batch_pos=%zu data_index=%zu token_count=%zu label=%d target_token=%d target_decode=\"%s\" last_token=%d last_decode=\"%s\"\n",
            debug.tag.c_str(),
            batch_pos,
            data_index,
            sample->input_ids.size(),
            sample->label,
            sample->target_token,
            sst2_token_piece(debug.vocab, sample->target_token).c_str(),
            last_token,
            sst2_token_piece(debug.vocab, last_token).c_str());
    sst2_print_token_list("debug_llama_tokens", sample->input_ids, debug.vocab, 64);
}

static void sst2_print_logit_debug(
        const sst2_debug_state & debug,
        const float * logits,
        int32_t n_vocab,
        const sst2_batch_data & data,
        size_t batch_pos,
        size_t final_offset) {
    const sst2_sample * sample = data.samples[batch_pos];
    const int32_t final_index = data.final_indices[final_offset + batch_pos];
    const llama_token target = sample->target_token;
    float max_value = -std::numeric_limits<float>::infinity();
    int32_t max_token = 0;
    for (int32_t v = 0; v < n_vocab; ++v) {
        if (logits[v] > max_value) {
            max_value = logits[v];
            max_token = v;
        }
    }
    double sum = 0.0;
    for (int32_t v = 0; v < n_vocab; ++v) {
        sum += std::exp((double) logits[v] - (double) max_value);
    }
    const double logsumexp = std::log(sum) + (double) max_value;
    const double loss = logsumexp - (double) logits[target];

    std::ostringstream os;
    os << "logit_debug tag=" << debug.tag
       << " final_index=" << final_index
       << " vocab=" << n_vocab
       << " target=" << target
       << " target_decode=\"" << sst2_token_piece(debug.vocab, target) << "\""
       << " target_logit=" << logits[target]
       << " max_token=" << max_token
       << " max_decode=\"" << sst2_token_piece(debug.vocab, max_token) << "\""
       << " max_logit=" << max_value
       << " logsumexp=" << logsumexp
       << " loss=" << loss;
    if (debug.class_tokens != nullptr && debug.class_tokens->size() >= 2 &&
            (*debug.class_tokens)[0] >= 0 && (*debug.class_tokens)[0] < n_vocab &&
            (*debug.class_tokens)[1] >= 0 && (*debug.class_tokens)[1] < n_vocab) {
        const float neg_logit = logits[(*debug.class_tokens)[0]];
        const float pos_logit = logits[(*debug.class_tokens)[1]];
        const float class_max = std::max(neg_logit, pos_logit);
        const double class_logsumexp = std::log(std::exp((double) neg_logit - (double) class_max) +
                                                std::exp((double) pos_logit - (double) class_max)) +
                                       (double) class_max;
        const double binary_loss = class_logsumexp - (double) logits[target];
        os << " neg_logit=" << neg_logit
           << " pos_logit=" << pos_logit
           << " class_margin_pos_minus_neg=" << (pos_logit - neg_logit)
           << " binary_verbalizer_loss=" << binary_loss;
    }
    LOG("%s\n", os.str().c_str());
    sst2_print_float_stats(std::string("logit_row_stats tag=") + debug.tag, logits, (size_t) n_vocab);

    std::vector<int32_t> top;
    top.reserve(5);
    for (int32_t v = 0; v < n_vocab; ++v) {
        auto pos = std::lower_bound(top.begin(), top.end(), v, [&](int32_t lhs, int32_t rhs) {
            return logits[lhs] > logits[rhs];
        });
        top.insert(pos, v);
        if (top.size() > 5) {
            top.pop_back();
        }
    }
    for (size_t i = 0; i < top.size(); ++i) {
        const int32_t token = top[i];
        LOG("logit_top%zu token=%d logit=%g decode=\"%s\"\n",
                i + 1,
                token,
                (double) logits[token],
                sst2_token_piece(debug.vocab, token).c_str());
    }
}

static void sst2_build_decode_batch(
        llama_batch & batch,
        sst2_batch_data & data,
        bool paired,
        llama_token pad_token,
        bool debug_padding,
        bool debug_logits,
        bool htp_active) {
    common_batch_clear(batch);
    data.final_indices.clear();
    data.n_tokens = 0;

    size_t max_len = 0;
    if (!debug_padding) {
        for (const sst2_sample * sample : data.samples) {
            if (sample->input_ids.empty()) {
                throw std::runtime_error("empty SST2 tokenized sample");
            }
            max_len = std::max(max_len, sample->input_ids.size());
        }
        if (max_len == 0) {
            throw std::runtime_error("empty SST2 batch");
        }
        // HTP consumes rows in 32-row tiles.  Keep M 32-aligned to avoid partial
        // tile output corruption.  Q4_0 HMX additionally has a faster pipelined
        // path for M >= 128, so paired Q4 LoRA forwards use the smallest extra
        // padding that reaches that threshold.
        if (htp_active) {
            const size_t B = data.samples.size();
            const size_t total_factor = (paired ? (size_t) 2 : (size_t) 1) * B;
            if (paired && g_sst2_htp_q4_fused_lora_active) {
                const size_t min_len_for_q4_pipeline = (128 + total_factor - 1) / total_factor;
                max_len = std::max(max_len, min_len_for_q4_pipeline);
            }
            const size_t g = std::gcd(total_factor, (size_t) 32);
            const size_t align = g > 0 ? (size_t) 32 / g : (size_t) 1;
            max_len = ((max_len + align - 1) / align) * align;
        }
    }

    auto add_one_side = [&](int seq_offset) {
        for (size_t s = 0; s < data.samples.size(); ++s) {
            const auto & ids = data.samples[s]->input_ids;
            if (ids.empty()) {
                throw std::runtime_error("empty SST2 tokenized sample");
            }
            int32_t final_index = -1;
            for (int32_t t = 0; t < (int32_t) ids.size(); ++t) {
                final_index = batch.n_tokens;
                const bool logits_flag = debug_logits ? true : (t + 1 == (int32_t) ids.size());
                common_batch_add(batch, ids[(size_t) t], t, { (llama_seq_id) (seq_offset + (int) s) }, logits_flag);
            }
            if (!debug_padding) {
                for (int32_t t = (int32_t) ids.size(); t < (int32_t) max_len; ++t) {
                    common_batch_add(batch, pad_token, t, { (llama_seq_id) (seq_offset + (int) s) }, false);
                }
            }
            data.final_indices.push_back(final_index);
        }
    };

    add_one_side(0);
    if (paired) {
        add_one_side((int) data.samples.size());
    }
    data.n_tokens = batch.n_tokens;
}

static int64_t sst2_run_decode(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        sst2_batch_data & data,
        bool paired,
        llama_token pad_token,
        bool debug_padding,
        bool debug_logits,
        bool htp_active) {
    sst2_build_decode_batch(batch, data, paired, pad_token, debug_padding, debug_logits, htp_active);
    llama_memory_clear(mem, false);

    const int64_t t_start = ggml_time_us();
    if (!decode_helper(ctx, batch, n_batch)) {
        throw std::runtime_error("llama_decode failed during SST2 forward");
    }
    llama_synchronize(ctx);
    return ggml_time_us() - t_start;
}

static float sst2_loss_from_current_logits(
        llama_context * ctx,
        int32_t n_vocab,
        const sst2_batch_data & data,
        size_t final_offset,
        sst2_debug_state * debug = nullptr) {
    double loss_sum = 0.0;
    for (size_t i = 0; i < data.samples.size(); ++i) {
        const int32_t idx = data.final_indices[final_offset + i];
        const float * logits = llama_get_logits_ith(ctx, idx);
        if (logits == nullptr) {
            throw std::runtime_error("failed to fetch SST2 logits row");
        }
        const bool do_debug = debug != nullptr && debug->enabled && debug->printed_samples < debug->sample_limit;
        if (do_debug) {
            sst2_print_sample_debug(*debug, data, i);
            sst2_print_logit_debug(*debug, logits, n_vocab, data, i, final_offset);
            ++debug->printed_samples;
        }
        loss_sum += logits_cross_entropy(logits, n_vocab, data.samples[i]->target_token);
    }
    return (float) (loss_sum / (double) data.samples.size());
}

static sst2_logits_snapshot sst2_copy_loss_logits_from_current(
        llama_context * ctx,
        int32_t n_vocab,
        const sst2_batch_data & data,
        size_t final_offset) {
    sst2_logits_snapshot snap;
    snap.n_vocab = n_vocab;
    snap.logits.resize(data.samples.size() * (size_t) n_vocab);
    snap.labels.resize(data.samples.size());
    for (size_t i = 0; i < data.samples.size(); ++i) {
        const int32_t idx = data.final_indices[final_offset + i];
        const float * logits = llama_get_logits_ith(ctx, idx);
        if (logits == nullptr) {
            throw std::runtime_error("failed to fetch SST2 logits row for snapshot");
        }
        std::memcpy(snap.logits.data() + i * (size_t) n_vocab, logits, (size_t) n_vocab * sizeof(float));
        snap.labels[i] = data.samples[i]->target_token;
    }
    return snap;
}

static float sst2_loss_from_snapshot(const sst2_logits_snapshot & snap) {
    double loss_sum = 0.0;
    for (size_t i = 0; i < snap.labels.size(); ++i) {
        const float * logits = snap.logits.data() + i * (size_t) snap.n_vocab;
        loss_sum += logits_cross_entropy(logits, snap.n_vocab, snap.labels[i]);
    }
    return (float) (loss_sum / (double) snap.labels.size());
}

static sst2_forward_result sst2_run_forward(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        sst2_batch_data data,
        llama_token pad_token,
        bool debug_padding,
        bool debug_logits,
        bool htp_active,
        sst2_debug_state * debug = nullptr) {
    sst2_forward_result res;
    res.t_pp_us = sst2_run_decode(ctx, batch, mem, n_batch, data, false, pad_token, debug_padding, debug_logits, htp_active);
    res.n_tokens = data.n_tokens;
    res.loss = sst2_loss_from_current_logits(ctx, n_vocab, data, 0, debug);
    return res;
}

static void sst2_run_layerwise_check(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        const sst2_batch_data & warmup_data,
        llama_token pad_token,
        bool debug_padding,
        bool debug_logits,
        bool htp_active,
        const llama_vocab * vocab,
        const std::vector<llama_token> & class_tokens,
        const sst2_options & opts,
        sst2_layerwise_debug_state & layerwise_state) {
    const int sample_count = std::min(opts.diag_layerwise, (int) warmup_data.samples.size());
    LOG("diag_layerwise framework=llama batch_size=%zu samples=%d values=%d layers=",
            warmup_data.samples.size(), sample_count, opts.diag_layerwise_values);
    if (opts.diag_layerwise_layers.empty()) {
        LOG("all\n");
    } else {
        for (size_t i = 0; i < opts.diag_layerwise_layers.size(); ++i) {
            LOG("%s%d", i > 0 ? "," : "", opts.diag_layerwise_layers[i]);
        }
        LOG("\n");
    }

    for (int i = 0; i < sample_count; ++i) {
        if (warmup_data.samples[(size_t) i] == nullptr) {
            throw std::runtime_error("diag-layerwise warmup sample is null");
        }
        const sst2_sample & sample = *warmup_data.samples[(size_t) i];
        const size_t data_index = i < (int) warmup_data.sample_indices.size() ?
                warmup_data.sample_indices[(size_t) i] : (size_t) -1;

        sst2_batch_data single;
        single.samples.push_back(&sample);
        single.sample_indices.push_back(data_index);

        sst2_print_activation_sample_debug("llama", "warmup_B0", i, data_index, sample);

        layerwise_state.active = true;
        layerwise_state.tag = "warmup_B0";
        layerwise_state.forward_id++;
        layerwise_state.sample_ordinal = i;
        layerwise_state.data_index = data_index;
        layerwise_state.label = sample.label;
        layerwise_state.target_token = sample.target_token;
        layerwise_state.token_count = (int) sample.input_ids.size();
        layerwise_state.final_pos = sample.input_ids.empty() ? -1 : (int) sample.input_ids.size() - 1;
        LOG("activation_debug_begin framework=llama tag=%s forward_id=%d data_index=%zu sample_ordinal=%d seq_len=%d final_pos=%d label=%d target=%d\n",
                layerwise_state.tag.c_str(),
                layerwise_state.forward_id,
                layerwise_state.data_index,
                layerwise_state.sample_ordinal,
                layerwise_state.token_count,
                layerwise_state.final_pos,
                layerwise_state.label,
                layerwise_state.target_token);

        sst2_debug_state debug;
        debug.enabled = true;
        debug.tag = "diag_layerwise";
        debug.vocab = vocab;
        debug.class_tokens = &class_tokens;
        debug.sample_limit = 1;
        debug.printed_samples = 0;

        try {
            const sst2_forward_result result = sst2_run_forward(ctx, batch, mem, n_batch, n_vocab, single,
                    pad_token, debug_padding, debug_logits, htp_active, &debug);
            LOG("diag_layerwise_result framework=llama sample_ordinal=%d data_index=%zu loss=%g t_pp_us=%lld\n",
                    i, data_index, (double) result.loss, (long long) result.t_pp_us);
        } catch (...) {
            layerwise_state.active = false;
            throw;
        }

        LOG("activation_debug_end framework=llama tag=%s forward_id=%d printed_ops=unknown\n",
                layerwise_state.tag.c_str(), layerwise_state.forward_id);
        layerwise_state.active = false;
    }
}

static sst2_paired_forward_result sst2_run_paired_forward(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        sst2_batch_data data,
        llama_token pad_token,
        bool debug_padding,
        bool debug_logits,
        bool htp_active,
        sst2_debug_state * plus_debug = nullptr,
        sst2_debug_state * minus_debug = nullptr) {
    sst2_paired_forward_result res;
    res.t_pp_us = sst2_run_decode(ctx, batch, mem, n_batch, data, true, pad_token, debug_padding, debug_logits, htp_active);
    res.n_tokens = data.n_tokens;
    const size_t n = data.samples.size();
    res.loss_plus  = sst2_loss_from_current_logits(ctx, n_vocab, data, 0, plus_debug);
    res.loss_minus = sst2_loss_from_current_logits(ctx, n_vocab, data, n, minus_debug);
    return res;
}

static sst2_eval_result sst2_evaluate(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        const std::vector<sst2_sample> & eval,
        int batch_size,
        const std::vector<llama_token> & class_tokens,
        llama_token pad_token,
        bool debug_padding,
        bool debug_logits,
        bool htp_active) {
    GGML_UNUSED(n_vocab);
    sst2_eval_result res;
    const int64_t t_start = ggml_time_us();
    for (size_t start = 0; start < eval.size(); start += (size_t) batch_size) {
        sst2_batch_data data = sst2_make_eval_batch(eval, start, batch_size);
        (void) sst2_run_decode(ctx, batch, mem, n_batch, data, false, pad_token, debug_padding, debug_logits, htp_active);
        for (size_t i = 0; i < data.samples.size(); ++i) {
            const float * logits = llama_get_logits_ith(ctx, data.final_indices[i]);
            if (logits == nullptr) {
                throw std::runtime_error("failed to fetch SST2 eval logits row");
            }
            const int pred = logits[class_tokens[1]] > logits[class_tokens[0]] ? 1 : 0;
            if (pred == data.samples[i]->label) {
                ++res.correct;
            }
            ++res.total;
        }
    }
    res.t_eval_us = ggml_time_us() - t_start;
    res.accuracy = res.total > 0 ? (float) res.correct / (float) res.total : 0.0f;
    return res;
}

static int64_t read_current_pss_kb() {
    std::ifstream fin("/proc/self/smaps_rollup");
    if (!fin.is_open()) {
        return -1;
    }
    std::string key;
    int64_t value = 0;
    std::string unit;
    while (fin >> key >> value >> unit) {
        if (key == "Pss:") {
            return value;
        }
        std::string rest;
        std::getline(fin, rest);
    }
    return -1;
}

struct pss_peak_sampler {
    std::atomic<bool>    stop { false };
    std::atomic<int64_t> peak_kb { -1 };
    std::thread          worker;

    void observe_once() {
        const int64_t pss = read_current_pss_kb();
        if (pss < 0) {
            return;
        }

        int64_t cur = peak_kb.load(std::memory_order_relaxed);
        while (pss > cur && !peak_kb.compare_exchange_weak(cur, pss, std::memory_order_relaxed)) {
        }
    }

    void start() {
        observe_once();
        worker = std::thread([this]() {
            while (!stop.load(std::memory_order_relaxed)) {
                observe_once();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }

    int64_t peak() {
        observe_once();
        return peak_kb.load(std::memory_order_relaxed);
    }

    void shutdown() {
        stop.store(true, std::memory_order_relaxed);
        if (worker.joinable()) {
            worker.join();
        }
    }

    ~pss_peak_sampler() {
        shutdown();
    }
};

static std::vector<merge_repack_weight> collect_merge_repack_weights(
        llama_adapter_lora * adapter,
        const std::vector<lora_b_tensor> & lora_b_tensors) {
    std::unordered_map<ggml_tensor *, size_t> b_to_index;
    for (size_t i = 0; i < lora_b_tensors.size(); ++i) {
        b_to_index[lora_b_tensors[i].tensor] = i;
    }

    std::vector<merge_repack_weight> weights;
    weights.reserve(adapter->ab_map.size());
    for (auto & it : adapter->ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & lw = it.second;
        if (lw.a == nullptr || lw.b == nullptr) {
            continue;
        }
        if (name.size() >= std::strlen("token_embd.weight") &&
                name.compare(name.size() - std::strlen("token_embd.weight"), std::strlen("token_embd.weight"), "token_embd.weight") == 0) {
            throw std::runtime_error("merge-repack baseline does not support token embedding LoRA yet: " + name);
        }
        const auto pos = b_to_index.find(lw.b);
        if (pos == b_to_index.end()) {
            throw std::runtime_error("internal error: missing LoRA B shadow tensor for " + name);
        }

        ggml_tensor * base = const_cast<ggml_tensor *>(adapter->model->get_tensor(name.c_str()));
        if (base == nullptr) {
            throw std::runtime_error("base tensor not found for merge-repack: " + name);
        }
        if (base->type != GGML_TYPE_F16 && base->type != GGML_TYPE_F32) {
            throw std::runtime_error("merge-repack supports F16/F32 base tensors only: " + name);
        }
        if (lw.a->type != GGML_TYPE_F16 && lw.a->type != GGML_TYPE_F32) {
            throw std::runtime_error("merge-repack supports F16/F32 LoRA-A tensors only: " + name);
        }
        if (base->ne[0] != lw.a->ne[0] || base->ne[1] != lw.b->ne[1] || lw.a->ne[1] != lw.b->ne[0]) {
            throw std::runtime_error("merge-repack LoRA shape mismatch for " + name);
        }

        merge_repack_weight entry;
        entry.name = name;
        entry.base = base;
        entry.type = base->type;
        entry.k = base->ne[0];
        entry.n = base->ne[1];
        entry.r = lw.b->ne[0];
        entry.b_index = pos->second;
        entry.scale = lw.get_scale(adapter->alpha, 1.0f);

        const size_t base_nbytes = ggml_nbytes(base);
        entry.io_buf.resize(base_nbytes);
        entry.base_original.resize((size_t) ggml_nelements(base));
        entry.merged.resize(entry.base_original.size());
        ggml_backend_tensor_get(base, entry.io_buf.data(), 0, base_nbytes);
        deserialize_tensor_to_float(entry.io_buf, entry.type, entry.base_original);

        std::vector<uint8_t> a_buf(ggml_nbytes(lw.a));
        entry.lora_a.resize((size_t) ggml_nelements(lw.a));
        ggml_backend_tensor_get(lw.a, a_buf.data(), 0, a_buf.size());
        deserialize_tensor_to_float(a_buf, lw.a->type, entry.lora_a);

        weights.emplace_back(std::move(entry));
    }

    if (weights.empty()) {
        throw std::runtime_error("no merge-repack weights collected");
    }
    return weights;
}

static merge_repack_timing merge_repack_from(
        std::vector<merge_repack_weight> & merge_weights,
        const std::vector<lora_b_tensor> & lora_b_tensors,
        lora_push_source source) {
    merge_repack_timing timing;
    for (auto & w : merge_weights) {
        const auto & b = push_source_buffer(lora_b_tensors[w.b_index], source);
        const int64_t t_merge_start = ggml_time_us();
        w.merged = w.base_original;
        for (int64_t out = 0; out < w.n; ++out) {
            for (int64_t ir = 0; ir < w.r; ++ir) {
                const float bv = w.scale * b[(size_t) out * (size_t) w.r + (size_t) ir];
                if (bv == 0.0f) {
                    continue;
                }
                const size_t a_base = (size_t) ir * (size_t) w.k;
                const size_t w_base = (size_t) out * (size_t) w.k;
                for (int64_t in = 0; in < w.k; ++in) {
                    w.merged[w_base + (size_t) in] += w.lora_a[a_base + (size_t) in] * bv;
                }
            }
        }
        timing.merge_us += ggml_time_us() - t_merge_start;

        serialize_float_to_tensor(w.merged, w.type, w.io_buf);
        const int64_t t_set_start = ggml_time_us();
        ggml_backend_tensor_set(w.base, w.io_buf.data(), 0, w.io_buf.size());
        timing.set_us += ggml_time_us() - t_set_start;
    }
    return timing;
}

static void prepare_eval_lora_state(
        const sst2_options & opts,
        std::vector<lora_b_tensor> & lora_b_tensors,
        std::vector<merge_repack_weight> & merge_weights) {
    if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK) {
        (void) merge_repack_from(merge_weights, lora_b_tensors, lora_push_source::MASTER);
        return;
    }

    if (opts.antithetic) {
        set_env_var("GGML_HEXAGON_ANTITHETIC_LORA", "0");
    }
    (void) push_b_tensors_from(lora_b_tensors, lora_push_source::MASTER, "eval-master");
}

static void restore_train_lora_env(const sst2_options & opts) {
    if (opts.mode == zo_mode::COOP) {
        set_env_var("GGML_HEXAGON_ANTITHETIC_LORA", opts.antithetic ? "1" : "0");
    }
}

static void sst2_zero_lora_work(std::vector<lora_b_tensor> & tensors) {
    for (auto & tensor : tensors) {
        std::fill(tensor.work.begin(), tensor.work.end(), 0.0f);
    }
}

static void sst2_fill_single_target_lora_work(
        std::vector<lora_b_tensor> & tensors,
        size_t target_index,
        float signed_epsilon,
        const sst2_options & opts) {
    sst2_zero_lora_work(tensors);
    if (target_index >= tensors.size()) {
        return;
    }
    lora_b_tensor & tensor = tensors[target_index];
    const std::string canonical_key = sst2_lora_b_canonical_key(tensor.name);
    const int64_t rank = tensor.tensor->ne[0];
    const int64_t out  = tensor.tensor->ne[1];
    for (size_t mnn_flat = 0; mnn_flat < (size_t) tensor.n_elms; ++mnn_flat) {
        const float z = sst2_diag_lora_pattern_value(canonical_key, opts.diag_lora_pattern,
                                                     mnn_flat, rank, out);
        const size_t ggml_flat = sst2_lora_b_ggml_index_for_mnn_flat(tensor, mnn_flat);
        tensor.work[ggml_flat] = signed_epsilon * z;
    }
}

static void sst2_apply_lora_work_for_diag(
        const sst2_options & opts,
        std::vector<lora_b_tensor> & lora_b_tensors,
        std::vector<merge_repack_weight> & merge_weights,
        const char * label) {
    if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK) {
        (void) merge_repack_from(merge_weights, lora_b_tensors, lora_push_source::WORK);
    } else {
        (void) push_b_tensors_from(lora_b_tensors, lora_push_source::WORK, label);
    }
}

static void sst2_run_lora_a_check(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        const sst2_batch_data & warmup_data,
        llama_token pad_token,
        bool debug_padding,
        bool debug_logits,
        bool htp_active,
        const llama_vocab * vocab,
        const std::vector<llama_token> & class_tokens,
        const sst2_options & opts,
        llama_adapter_lora * adapter,
        std::vector<lora_b_tensor> & lora_b_tensors,
        std::vector<merge_repack_weight> & merge_weights,
        sst2_lora_branch_debug_state & branch_state) {
    if (opts.antithetic) {
        throw std::runtime_error("diag-lora-a does not support --antithetic true");
    }
    const int sample_count = std::min(opts.diag_lora_a, (int) warmup_data.samples.size());
    std::vector<size_t> selected_targets;
    sst2_select_diag_lora_targets(lora_b_tensors, opts.diag_lora_targets, "diag_lora_a", selected_targets);

    LOG("diag_lora_a framework=llama batch_size=%zu samples=%d targets=%zu values=%d\n",
            warmup_data.samples.size(),
            sample_count,
            selected_targets.size(),
            opts.diag_lora_values);

    auto make_debug = [&](const char * tag) {
        sst2_debug_state debug;
        debug.enabled = true;
        debug.tag = tag;
        debug.vocab = vocab;
        debug.class_tokens = &class_tokens;
        debug.sample_limit = 1;
        debug.printed_samples = 0;
        return debug;
    };

    sst2_zero_lora_work(lora_b_tensors);
    sst2_apply_lora_work_for_diag(opts, lora_b_tensors, merge_weights, "diag-lora-a-B0");

    for (int s = 0; s < sample_count; ++s) {
        if (warmup_data.samples[(size_t) s] == nullptr) {
            throw std::runtime_error("diag-lora-a warmup sample is null");
        }
        const sst2_sample & sample = *warmup_data.samples[(size_t) s];
        const size_t data_index = s < (int) warmup_data.sample_indices.size() ?
                warmup_data.sample_indices[(size_t) s] : (size_t) -1;
        sst2_print_activation_sample_debug("llama", "diag_lora_a", s, data_index, sample);

        sst2_batch_data single;
        single.samples.push_back(&sample);
        single.sample_indices.push_back(data_index);

        for (size_t selected_ordinal = 0; selected_ordinal < selected_targets.size(); ++selected_ordinal) {
            const size_t target_index = selected_targets[selected_ordinal];
            const lora_b_tensor & tensor = lora_b_tensors[target_index];
            const std::string canonical_key = sst2_lora_b_canonical_key(tensor.name);
            LOG("diag_lora_a_target framework=llama sample_ordinal=%d data_index=%zu selected_ordinal=%zu target_index=%zu tensor_name=%s canonical_key=%s shape_rank=%lld shape_out=%lld size=%lld\n",
                    s,
                    data_index,
                    selected_ordinal,
                    target_index,
                    tensor.name.c_str(),
                    canonical_key.c_str(),
                    (long long) tensor.tensor->ne[0],
                    (long long) tensor.tensor->ne[1],
                    (long long) tensor.n_elms);
            sst2_print_lora_a_debug(adapter, tensor, target_index, s, data_index, opts.diag_lora_values);

            sst2_debug_state debug = make_debug("diag_lora_A");
            sst2_begin_lora_branch_debug(&branch_state, "diag_lora_A", canonical_key, s, data_index, sample);
            sst2_forward_result result;
            try {
                result = sst2_run_forward(ctx, batch, mem, n_batch, n_vocab, single,
                        pad_token, debug_padding, debug_logits, htp_active, &debug);
            } catch (...) {
                sst2_end_lora_branch_debug(&branch_state);
                throw;
            }
            sst2_end_lora_branch_debug(&branch_state);
            LOG("diag_lora_a_eval framework=llama sample_ordinal=%d data_index=%zu selected_ordinal=%zu target_index=%zu canonical_key=%s loss=%g t_pp_us=%lld\n",
                    s,
                    data_index,
                    selected_ordinal,
                    target_index,
                    canonical_key.c_str(),
                    (double) result.loss,
                    (long long) result.t_pp_us);
        }
    }

    sst2_zero_lora_work(lora_b_tensors);
    sst2_apply_lora_work_for_diag(opts, lora_b_tensors, merge_weights, "diag-lora-a-restore-B0");
}

static void sst2_run_lora_perturb_check(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        const sst2_batch_data & warmup_data,
        llama_token pad_token,
        bool debug_padding,
        bool debug_logits,
        bool htp_active,
        const llama_vocab * vocab,
        const std::vector<llama_token> & class_tokens,
        const sst2_options & opts,
        std::vector<lora_b_tensor> & lora_b_tensors,
        std::vector<merge_repack_weight> & merge_weights,
        sst2_lora_branch_debug_state * branch_state = nullptr,
        bool branch_mode = false) {
    if (opts.antithetic) {
        throw std::runtime_error(branch_mode ? "diag-lora-branch does not support --antithetic true" :
                                               "diag-lora-perturb does not support --antithetic true");
    }
    const char * diag_name = branch_mode ? "diag_lora_branch" : "diag_lora_perturb";
    const int requested_samples = branch_mode ? opts.diag_lora_branch : opts.diag_lora_perturb;
    const int sample_count = std::min(requested_samples, (int) warmup_data.samples.size());
    std::vector<size_t> selected_targets;
    sst2_select_diag_lora_targets(lora_b_tensors, opts.diag_lora_targets, diag_name, selected_targets);

    LOG("%s framework=llama batch_size=%zu samples=%d targets=%zu pattern=%s epsilon=%g values=%d attn_internals=%s attn_heads=%s attn_values=%d\n",
            diag_name,
            warmup_data.samples.size(),
            sample_count,
            selected_targets.size(),
            opts.diag_lora_pattern.c_str(),
            (double) opts.epsilon,
            opts.diag_lora_values,
            opts.diag_attn_internals ? "true" : "false",
            sst2_join_int_list(opts.diag_attn_heads).c_str(),
            opts.diag_attn_values);

    auto make_debug = [&](const char * tag) {
        sst2_debug_state debug;
        debug.enabled = true;
        debug.tag = tag;
        debug.vocab = vocab;
        debug.class_tokens = &class_tokens;
        debug.sample_limit = 1;
        debug.printed_samples = 0;
        return debug;
    };

    auto run_current_work = [&](sst2_batch_data single,
                                const lora_b_tensor & tensor,
                                size_t target_index,
                                int sample_ordinal,
                                size_t data_index,
                                const sst2_sample & sample,
                                const char * tag) -> sst2_forward_result {
        sst2_print_lora_b_target_debug(tensor, target_index, tag, "work",
                                       opts.diag_lora_pattern, opts.epsilon,
                                       opts.diag_lora_values);
        sst2_apply_lora_work_for_diag(opts, lora_b_tensors, merge_weights, tag);
        sst2_debug_state debug = make_debug(tag);
        if (branch_mode) {
            sst2_begin_lora_branch_debug(branch_state, tag, sst2_lora_b_canonical_key(tensor.name),
                    sample_ordinal, data_index, sample);
        }
        sst2_forward_result result;
        try {
            result = sst2_run_forward(ctx, batch, mem, n_batch, n_vocab, single,
                    pad_token, debug_padding, debug_logits, htp_active, &debug);
        } catch (...) {
            if (branch_mode) {
                sst2_end_lora_branch_debug(branch_state);
            }
            throw;
        }
        if (branch_mode) {
            sst2_end_lora_branch_debug(branch_state);
        }
        LOG("%s_eval framework=llama tag=%s target_index=%zu canonical_key=%s loss=%g t_pp_us=%lld\n",
                diag_name,
                tag,
                target_index,
                sst2_lora_b_canonical_key(tensor.name).c_str(),
                (double) result.loss,
                (long long) result.t_pp_us);
        return result;
    };

    for (int s = 0; s < sample_count; ++s) {
        if (warmup_data.samples[(size_t) s] == nullptr) {
            throw std::runtime_error("diag-lora-perturb warmup sample is null");
        }
        const sst2_sample & sample = *warmup_data.samples[(size_t) s];
        const size_t data_index = s < (int) warmup_data.sample_indices.size() ?
                warmup_data.sample_indices[(size_t) s] : (size_t) -1;
        sst2_print_activation_sample_debug("llama", diag_name, s, data_index, sample);

        for (size_t selected_ordinal = 0; selected_ordinal < selected_targets.size(); ++selected_ordinal) {
            const size_t target_index = selected_targets[selected_ordinal];
            lora_b_tensor & tensor = lora_b_tensors[target_index];
            const std::string canonical_key = sst2_lora_b_canonical_key(tensor.name);
            LOG("%s_target framework=llama sample_ordinal=%d data_index=%zu selected_ordinal=%zu target_index=%zu tensor_name=%s canonical_key=%s shape_rank=%lld shape_out=%lld size=%lld\n",
                    diag_name,
                    s,
                    data_index,
                    selected_ordinal,
                    target_index,
                    tensor.name.c_str(),
                    canonical_key.c_str(),
                    (long long) tensor.tensor->ne[0],
                    (long long) tensor.tensor->ne[1],
                    (long long) tensor.n_elms);

            sst2_batch_data single;
            single.samples.push_back(&sample);
            single.sample_indices.push_back(data_index);

            sst2_zero_lora_work(lora_b_tensors);
            const sst2_forward_result b0 = run_current_work(single, tensor, target_index, s, data_index, sample, "diag_lora_B0");

            sst2_fill_single_target_lora_work(lora_b_tensors, target_index, opts.epsilon, opts);
            const sst2_forward_result plus = run_current_work(single, tensor, target_index, s, data_index, sample, "diag_lora_plus");

            sst2_fill_single_target_lora_work(lora_b_tensors, target_index, -opts.epsilon, opts);
            const sst2_forward_result minus = run_current_work(single, tensor, target_index, s, data_index, sample, "diag_lora_minus");

            const double g = ((double) plus.loss - (double) minus.loss) / (2.0 * (double) opts.epsilon);
            LOG("%s_summary framework=llama sample_ordinal=%d data_index=%zu target_index=%zu canonical_key=%s pattern=%s epsilon=%g loss_b0=%g loss_plus=%g loss_minus=%g plus_minus_delta=%g g=%g\n",
                    diag_name,
                    s,
                    data_index,
                    target_index,
                    canonical_key.c_str(),
                    opts.diag_lora_pattern.c_str(),
                    (double) opts.epsilon,
                    (double) b0.loss,
                    (double) plus.loss,
                    (double) minus.loss,
                    (double) plus.loss - (double) minus.loss,
                    g);
        }
    }

    sst2_zero_lora_work(lora_b_tensors);
    sst2_apply_lora_work_for_diag(opts, lora_b_tensors, merge_weights, "diag-lora-restore-B0");
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx       = 256;
    params.n_batch     = 256;
    params.n_ubatch    = 256;
    params.n_parallel  = 4;
    params.n_sequences = 4;

    sst2_options opts;
    sst2_layerwise_debug_state layerwise_debug;
    sst2_lora_branch_debug_state lora_branch_debug;

    common_init();

    if (!sst2_prescan_mode(argc, argv, opts)) {
        return 1;
    }

    std::vector<char *> filtered_argv;
    if (!sst2_parse_custom_args(argc, argv, opts, filtered_argv)) {
        sst2_print_usage(argc, argv);
        return 1;
    }

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_BENCH, sst2_print_usage)) {
        return 1;
    }

    if (opts.diag_layerwise > 0) {
        layerwise_debug.enabled = true;
        layerwise_debug.layers = opts.diag_layerwise_layers;
        layerwise_debug.max_values = opts.diag_layerwise_values;
        params.cb_eval = sst2_layerwise_eval_callback;
        params.cb_eval_user_data = &layerwise_debug;
    } else if (opts.diag_lora_a > 0 || opts.diag_lora_branch > 0) {
        lora_branch_debug.enabled = true;
        lora_branch_debug.max_values = opts.diag_lora_values;
        lora_branch_debug.propagation = opts.diag_lora_branch > 0 && opts.diag_lora_propagation;
        lora_branch_debug.attn_internals = opts.diag_lora_branch > 0 && opts.diag_attn_internals;
        lora_branch_debug.attn_heads = opts.diag_attn_heads;
        lora_branch_debug.attn_values = opts.diag_attn_values;
        params.cb_eval = sst2_lora_branch_eval_callback;
        params.cb_eval_user_data = &lora_branch_debug;
    }

    bool backend_inited = false;
    bool batch_inited = false;
    llama_batch batch = {};
    pss_peak_sampler pss_sampler;

    try {
        if (!opts.lora_exec_set) {
            opts.lora_exec = opts.mode == zo_mode::COOP ? sst2_lora_exec_mode::FUSED_HTP : sst2_lora_exec_mode::RUNTIME;
        }

        sst2_validate_options(opts);
        if (opts.diag_attn_internals) {
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            LOG("diag_attn_internals forcing llama.cpp flash attention off for kq/kq_soft_max/kqv callbacks\n");
        }

        if (params.model.path.empty()) {
            params.model.path = DEFAULT_MODEL_PATH;
        }
        if (params.lora_adapters.empty()) {
            params.lora_adapters.push_back({ DEFAULT_LORA_PATH, 1.0f, "", "", nullptr });
        }
        if (params.lora_adapters.size() != 1) {
            throw std::runtime_error("this example supports exactly one LoRA adapter");
        }

        const int effective_batch_size = opts.antithetic ? 2 * opts.batch_size : opts.batch_size;
        const int effective_total_tokens = effective_batch_size * opts.seq_len;
        params.n_parallel  = effective_batch_size;
        params.n_sequences = effective_batch_size;
        params.n_ctx       = std::max(params.n_ctx, effective_total_tokens);
        params.n_batch     = std::max(params.n_batch, effective_total_tokens);
        params.n_ubatch    = std::max(params.n_ubatch, effective_total_tokens);

        if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK) {
            params.lora_init_without_apply = true;
        }
        if (opts.diag_lora_a > 0 || opts.diag_lora_branch > 0) {
            set_env_var("GGML_LORA_DEBUG_BRANCH", "1");
        } else {
            set_env_var("GGML_LORA_DEBUG_BRANCH", "0");
        }

        if (opts.mode == zo_mode::CPU) {
            params.devices.clear();
            params.n_gpu_layers  = 0;
            params.main_gpu      = -1;
            params.fit_params    = false;
            params.no_op_offload = true;
            params.no_kv_offload = true;
            params.split_mode    = LLAMA_SPLIT_MODE_NONE;
            set_env_var("GGML_HEXAGON_FUSED_LORA", "0");
            set_env_var("GGML_HEXAGON_ANTITHETIC_LORA", "0");
        } else {
            if (!zo_hexagon_compiled()) {
                throw std::runtime_error("coop mode requires a GGML_USE_HEXAGON build");
            }
            if (params.devices.empty() || params.devices[0] == nullptr) {
                throw std::runtime_error("coop mode requires an explicit non-CPU device, e.g. --device HTP0");
            }
            set_env_var("GGML_HEXAGON_FUSED_LORA", opts.lora_exec == sst2_lora_exec_mode::FUSED_HTP ? "auto" : "0");
            set_env_var("GGML_HEXAGON_ANTITHETIC_LORA", opts.antithetic ? "1" : "0");
        }

        const bool cpu_unpadded_batch = opts.mode == zo_mode::CPU;
        const bool effective_debug_padding = opts.debug_padding || cpu_unpadded_batch;

        LOG("\n");
        LOG("zoo-sst2-zo-lora configuration:\n");
        LOG("  mode       = %s\n", zo_mode_name(opts.mode));
        LOG("  lora_exec  = %s\n", sst2_lora_exec_mode_name(opts.lora_exec));
        LOG("  pipeline   = %s\n", opts.pipeline ? "true" : "false");
        LOG("  antithetic = %s\n", opts.antithetic ? "true" : "false");
        LOG("  model      = %s\n", params.model.path.c_str());
        LOG("  lora       = %s\n", params.lora_adapters[0].path.c_str());
        LOG("  train_data = %s\n", opts.train_path.c_str());
        LOG("  eval_data  = %s\n", opts.eval_step == -1 ? "<disabled>" : opts.eval_path.c_str());
        LOG("  max_train  = %d\n", opts.max_train);
        LOG("  max_eval   = %d\n", opts.max_eval);
        LOG("  batch_size = %d\n", opts.batch_size);
        LOG("  seq_len    = %d (max prompt tokens)\n", opts.seq_len);
        LOG("  steps      = %d\n", opts.steps);
        LOG("  eval_step  = %d%s\n", opts.eval_step, opts.eval_step == -1 ? " (disabled)" : "");
        LOG("  epsilon    = %.6g\n", opts.epsilon);
        LOG("  lr         = %.6g\n", opts.lr);
        LOG("  debug_padding = %s\n", opts.debug_padding ? "true" : "false");
        LOG("  cpu_unpadded_batch = %s\n", cpu_unpadded_batch ? "true" : "false");
        LOG("  effective_debug_padding = %s\n", effective_debug_padding ? "true" : "false");
        LOG("  debug_logits  = %s\n", opts.debug_logits  ? "true" : "false");
        LOG("  debug_MNN     = %s\n", opts.debug_mnn     ? "true" : "false");
        LOG("  diag_layerwise = %d\n", opts.diag_layerwise);
        LOG("  diag_layerwise_values = %d\n", opts.diag_layerwise_values);
        LOG("  diag_lora_perturb = %d\n", opts.diag_lora_perturb);
        LOG("  diag_lora_a = %d\n", opts.diag_lora_a);
        LOG("  diag_lora_branch = %d\n", opts.diag_lora_branch);
        LOG("  diag_lora_propagation = %s\n", opts.diag_lora_propagation ? "true" : "false");
        LOG("  diag_attn_internals = %s\n", opts.diag_attn_internals ? "true" : "false");
        LOG("  diag_attn_heads = %s\n", sst2_join_int_list(opts.diag_attn_heads).c_str());
        LOG("  diag_attn_values = %d\n", opts.diag_attn_values);
        LOG("  diag_lora_values = %d\n", opts.diag_lora_values);
        LOG("  diag_lora_pattern = %s\n", opts.diag_lora_pattern.c_str());
        LOG("  seed       = %u\n", DEFAULT_SEED);
        LOG("  n_ctx      = %d\n", params.n_ctx);
        LOG("  n_batch    = %d\n", params.n_batch);
        LOG("  n_ubatch   = %d\n", params.n_ubatch);
        LOG("\n");

        pss_sampler.start();

        llama_backend_init();
        backend_inited = true;
        llama_numa_init(params.numa);

        auto init = common_init_from_params(params);
        if (!init) {
            throw std::runtime_error("common_init_from_params failed");
        }

        llama_model * model = init->model();
        llama_context * ctx = init->context();
        if (model == nullptr || ctx == nullptr) {
            throw std::runtime_error("failed to initialize model/context");
        }

        auto & loras = init->lora();
        if (loras.empty() || loras.front() == nullptr) {
            throw std::runtime_error("no LoRA adapter was loaded");
        }

        llama_adapter_lora * adapter = loras.front().get();
        auto lora_b_tensors = collect_lora_b_tensors(adapter);
        g_sst2_htp_q4_fused_lora_active =
                opts.mode == zo_mode::COOP &&
                opts.lora_exec == sst2_lora_exec_mode::FUSED_HTP &&
                sst2_model_has_htp_q4_lora_target(model, lora_b_tensors);
        log_lora_summary(lora_b_tensors);
        log_lora_placement_summary(
                lora_b_tensors,
                opts.mode == zo_mode::COOP && opts.lora_exec == sst2_lora_exec_mode::FUSED_HTP,
                opts.antithetic);
        LOG("htp_q4_fused_lora = %s\n", g_sst2_htp_q4_fused_lora_active ? "true" : "false");
        const bool any_diag_lora = opts.diag_lora_perturb > 0 || opts.diag_lora_a > 0 || opts.diag_lora_branch > 0;
        if (opts.debug_mnn || any_diag_lora) {
            sst2_print_lora_b_target_order_debug(lora_b_tensors, 0);
            sst2_print_lora_b_storage_debug(lora_b_tensors, "initial_master_B0", "master", 0);
        }
        if (opts.antithetic) {
            for (const auto & tensor : lora_b_tensors) {
                if (!lora_antithetic_uses_rm(tensor) && !lora_antithetic_uses_plain(tensor)) {
                    throw std::runtime_error("--antithetic true requires usable LoRA B+/B- tensors for " + tensor.name);
                }
                if (tensor.tensor->ne[0] % 8 != 0 || tensor.tensor->ne[0] > 32) {
                    throw std::runtime_error("--antithetic true currently supports rank 8/16/24/32 LoRA-B tensors only");
                }
            }
        }

        std::vector<merge_repack_weight> merge_weights;
        if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK) {
            merge_weights = collect_merge_repack_weights(adapter, lora_b_tensors);
            LOG("merge-repack weights: %zu adapted base tensors\n", merge_weights.size());
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (n_vocab <= 0) {
            throw std::runtime_error("invalid vocabulary size");
        }

        std::vector<std::vector<llama_token>> label_suffixes(2);
        label_suffixes[0] = sst2_tokenize(vocab, " It was terrible", false);
        label_suffixes[1] = sst2_tokenize(vocab, " It was great", false);
        std::vector<llama_token> class_tokens(2);
        for (size_t cls = 0; cls < label_suffixes.size(); ++cls) {
            if (label_suffixes[cls].empty()) {
                throw std::runtime_error("failed to tokenize SST2 label suffix");
            }
            class_tokens[cls] = label_suffixes[cls].back();
        }
        LOG("SST2 verbalizer tokens: negative=%d positive=%d suffix_lens=%zu,%zu\n",
                class_tokens[0], class_tokens[1],
                label_suffixes[0].size(), label_suffixes[1].size());

        llama_token pad_token = llama_vocab_pad(vocab);
        if (pad_token == LLAMA_TOKEN_NULL) {
            pad_token = llama_vocab_eos(vocab);
        }
        if (pad_token == LLAMA_TOKEN_NULL) {
            pad_token = llama_vocab_bos(vocab);
        }
        if (pad_token == LLAMA_TOKEN_NULL) {
            pad_token = 0;
        }
        LOG("SST2 batch padding token: %d\n", pad_token);

        const auto train_data = sst2_load_tsv(opts.train_path.c_str(), vocab, opts.max_train, opts.seq_len, label_suffixes);
        std::vector<sst2_sample> eval_data;
        if (opts.eval_step != -1) {
            eval_data = sst2_load_tsv(opts.eval_path.c_str(), vocab, opts.max_eval, opts.seq_len, label_suffixes);
        }
        LOG("SST2 samples: train=%zu eval=%zu\n", train_data.size(), eval_data.size());
        if (opts.debug_mnn || opts.diag_layerwise > 0 || any_diag_lora) {
            sst2_print_tokenizer_diagnostics(vocab, class_tokens, train_data);
        }

        llama_memory_t mem = llama_get_memory(ctx);
        batch = llama_batch_init(effective_total_tokens, 0, effective_batch_size);
        batch_inited = true;

        auto make_debug_state = [&](const char * tag, bool enabled) {
            sst2_debug_state debug;
            debug.enabled = enabled;
            debug.tag = tag;
            debug.vocab = vocab;
            debug.class_tokens = &class_tokens;
            debug.sample_limit = opts.debug_mnn ? std::min(opts.batch_size, 2) : 1;
            return debug;
        };

        if (opts.diag_layerwise > 0) {
            sst2_batch_data warmup_data = sst2_make_train_batch(train_data, opts.batch_size, 0);
            if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK) {
                (void) merge_repack_from(merge_weights, lora_b_tensors, lora_push_source::MASTER);
            } else {
                (void) push_b_tensors_from(lora_b_tensors, lora_push_source::MASTER, "diag-layerwise-master");
            }
            sst2_run_layerwise_check(ctx, batch, mem, params.n_batch, n_vocab, warmup_data, pad_token,
                    effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, vocab, class_tokens,
                    opts, layerwise_debug);
            if (batch_inited) {
                llama_batch_free(batch);
                batch_inited = false;
            }
            init.reset();
            if (backend_inited) {
                llama_backend_free();
                backend_inited = false;
            }
            return 0;
        }

        if (opts.diag_lora_a > 0) {
            sst2_batch_data warmup_data = sst2_make_train_batch(train_data, opts.batch_size, 0);
            sst2_run_lora_a_check(ctx, batch, mem, params.n_batch, n_vocab, warmup_data, pad_token,
                    effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, vocab, class_tokens,
                    opts, adapter, lora_b_tensors, merge_weights, lora_branch_debug);
            if (batch_inited) {
                llama_batch_free(batch);
                batch_inited = false;
            }
            init.reset();
            if (backend_inited) {
                llama_backend_free();
                backend_inited = false;
            }
            return 0;
        }

        if (opts.diag_lora_branch > 0) {
            sst2_batch_data warmup_data = sst2_make_train_batch(train_data, opts.batch_size, 0);
            sst2_run_lora_perturb_check(ctx, batch, mem, params.n_batch, n_vocab, warmup_data, pad_token,
                    effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, vocab, class_tokens,
                    opts, lora_b_tensors, merge_weights, &lora_branch_debug, true);
            if (batch_inited) {
                llama_batch_free(batch);
                batch_inited = false;
            }
            init.reset();
            if (backend_inited) {
                llama_backend_free();
                backend_inited = false;
            }
            return 0;
        }

        if (opts.diag_lora_perturb > 0) {
            sst2_batch_data warmup_data = sst2_make_train_batch(train_data, opts.batch_size, 0);
            sst2_run_lora_perturb_check(ctx, batch, mem, params.n_batch, n_vocab, warmup_data, pad_token,
                    effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, vocab, class_tokens,
                    opts, lora_b_tensors, merge_weights);
            if (batch_inited) {
                llama_batch_free(batch);
                batch_inited = false;
            }
            init.reset();
            if (backend_inited) {
                llama_backend_free();
                backend_inited = false;
            }
            return 0;
        }

        LOG("running untimed warmup prefill ...\n");
        {
            sst2_batch_data warmup_data = sst2_make_train_batch(train_data, opts.batch_size, 0);
            sst2_debug_state warmup_debug = make_debug_state("warmup_B0", opts.debug_mnn);
            if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK) {
                (void) merge_repack_from(merge_weights, lora_b_tensors, lora_push_source::MASTER);
                (void) sst2_run_forward(ctx, batch, mem, params.n_batch, n_vocab, warmup_data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &warmup_debug);
            } else if (opts.antithetic) {
                sst2_debug_state warmup_minus_debug = make_debug_state("warmup_B0_minus", opts.debug_mnn);
                for (auto & tensor : lora_b_tensors) {
                    set_antithetic_rm_from_master(tensor);
                }
                (void) push_antithetic_b_tensors(lora_b_tensors, "warmup-master");
                (void) sst2_run_paired_forward(ctx, batch, mem, params.n_batch, n_vocab, warmup_data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &warmup_debug, &warmup_minus_debug);
            } else {
                (void) push_b_tensors_from(lora_b_tensors, lora_push_source::MASTER, "warmup-master");
                (void) sst2_run_forward(ctx, batch, mem, params.n_batch, n_vocab, warmup_data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &warmup_debug);
            }
        }

        json_log_writer train_log("./train_log.json");
        LOG("training log: ./train_log.json (time uses milliseconds, matching terminal step=... ms)\n");

        std::unique_ptr<json_log_writer> eval_log;
        if (opts.eval_step != -1) {
            eval_log.reset(new json_log_writer("./eval_log.json"));
            LOG("evaluation log: ./eval_log.json (pss_mb uses MiB: 1 MiB = 1024 * 1024 bytes)\n");
        }

        auto run_eval = [&](int step, int64_t train_time_us) {
            prepare_eval_lora_state(opts, lora_b_tensors, merge_weights);
            const int64_t peak_pss_kb = pss_sampler.peak();
            const sst2_eval_result eval = sst2_evaluate(
                    ctx, batch, mem, params.n_batch, n_vocab, eval_data, opts.batch_size, class_tokens, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU);
            const double training_time_s = train_time_us / 1e6;
            const double pss_mb = peak_pss_kb >= 0 ? (double) peak_pss_kb / 1024.0 : -1.0;
            LOG("[eval] step=%d accuracy=%.4f (%d/%d) train_time=%.3f s eval_time=%.3f s peak_pss=%.3f MiB\n",
                    step, (double) eval.accuracy, eval.correct, eval.total,
                    training_time_s, eval.t_eval_us / 1e6, pss_mb);
            append_eval_log(*eval_log, eval.accuracy, training_time_s, pss_mb);
            restore_train_lora_env(opts);
        };

        if (opts.eval_step != -1) {
            LOG("\ninitial evaluation before training:\n");
            run_eval(0, 0);
        } else {
            LOG("\nevaluation disabled by --eval-step -1\n");
        }

        sst2_perf_stats stats;
        int64_t train_time_us = 0;

        auto maybe_eval = [&](int completed_step) {
            if (opts.eval_step == -1) {
                return;
            }
            if (completed_step % opts.eval_step == 0 || completed_step == opts.steps) {
                run_eval(completed_step, train_time_us);
            }
        };

        if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK) {
            for (int step = 0; step < opts.steps; ++step) {
                sst2_batch_data data = sst2_make_train_batch(train_data, opts.batch_size, step);
                const int64_t t_step_start = ggml_time_us();

                const int64_t t_prepare_plus_us = make_work_from_master_regen_noise(lora_b_tensors, step, +opts.epsilon);
                if (opts.debug_mnn && step == 0) {
                    sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_plus_work", "work", step);
                }
                const merge_repack_timing plus_merge = merge_repack_from(merge_weights, lora_b_tensors, lora_push_source::WORK);
                sst2_debug_state plus_debug = make_debug_state("step1_plus", opts.debug_mnn && step == 0);
                const sst2_forward_result plus = sst2_run_forward(ctx, batch, mem, params.n_batch, n_vocab, data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &plus_debug);

                const int64_t t_prepare_minus_us = make_work_from_master_regen_noise(lora_b_tensors, step, -opts.epsilon);
                if (opts.debug_mnn && step == 0) {
                    sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_minus_work", "work", step);
                }
                const merge_repack_timing minus_merge = merge_repack_from(merge_weights, lora_b_tensors, lora_push_source::WORK);
                sst2_debug_state minus_debug = make_debug_state("step1_minus", opts.debug_mnn && step == 0);
                const sst2_forward_result minus = sst2_run_forward(ctx, batch, mem, params.n_batch, n_vocab, data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &minus_debug);

                const float g = (plus.loss - minus.loss) / (2.0f * opts.epsilon);
                const int64_t t_prepare_update_us = apply_update_to_master_regen_noise(lora_b_tensors, step, -opts.lr * g);
                if (opts.debug_mnn && step == 0) {
                    sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_master_after_update", "master", step);
                }

                const int64_t t_step_us = ggml_time_us() - t_step_start;
                train_time_us += t_step_us;

                stats.total_pp_us += plus.t_pp_us + minus.t_pp_us;
                stats.total_pp_tokens += plus.n_tokens + minus.n_tokens;
                stats.total_step_us += t_step_us;
                stats.total_prepare_plus_us += t_prepare_plus_us;
                stats.total_prepare_minus_us += t_prepare_minus_us;
                stats.total_prepare_update_us += t_prepare_update_us;
                stats.total_merge_plus_us += plus_merge.total_us();
                stats.total_merge_minus_us += minus_merge.total_us();
                stats.total_forwards += 2;

                const float loss_delta = plus.loss - minus.loss;
                append_train_log(train_log, step + 1, 0.5f * (plus.loss + minus.loss), t_step_us / 1000.0);
                LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                    "t_pp_plus=%.3f ms t_pp_minus=%.3f ms | step=%.3f ms\n",
                        step + 1, opts.steps, plus.loss, minus.loss, (double) loss_delta, (double) g,
                        plus.t_pp_us / 1000.0, minus.t_pp_us / 1000.0, t_step_us / 1000.0);
                LOG("  merge-repack: plus_prepare=%.3f ms plus_merge=%.3f ms plus_repack=%.3f ms | "
                    "minus_prepare=%.3f ms minus_merge=%.3f ms minus_repack=%.3f ms | update=%.3f ms\n",
                        t_prepare_plus_us / 1000.0, plus_merge.merge_us / 1000.0, plus_merge.set_us / 1000.0,
                        t_prepare_minus_us / 1000.0, minus_merge.merge_us / 1000.0, minus_merge.set_us / 1000.0,
                        t_prepare_update_us / 1000.0);
                maybe_eval(step + 1);
            }
        } else if (!opts.pipeline) {
            if (!opts.antithetic) {
                for (int step = 0; step < opts.steps; ++step) {
                    sst2_batch_data data = sst2_make_train_batch(train_data, opts.batch_size, step);
                    const int64_t t_step_start = ggml_time_us();

                    const int64_t t_prepare_plus_us = make_work_from_master_regen_noise(lora_b_tensors, step, +opts.epsilon);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_plus_work", "work", step);
                    }
                    const int64_t t_push_plus_us = push_b_tensors(lora_b_tensors);
                    sst2_debug_state plus_debug = make_debug_state("step1_plus", opts.debug_mnn && step == 0);
                    const sst2_forward_result plus = sst2_run_forward(ctx, batch, mem, params.n_batch, n_vocab, data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &plus_debug);

                    const int64_t t_prepare_minus_us = make_work_from_master_regen_noise(lora_b_tensors, step, -opts.epsilon);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_minus_work", "work", step);
                    }
                    const int64_t t_push_minus_us = push_b_tensors(lora_b_tensors);
                    sst2_debug_state minus_debug = make_debug_state("step1_minus", opts.debug_mnn && step == 0);
                    const sst2_forward_result minus = sst2_run_forward(ctx, batch, mem, params.n_batch, n_vocab, data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &minus_debug);

                    const float g = (plus.loss - minus.loss) / (2.0f * opts.epsilon);
                    const int64_t t_prepare_update_us = apply_update_to_master_regen_noise(lora_b_tensors, step, -opts.lr * g);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_master_after_update", "master", step);
                    }

                    const int64_t t_step_us = ggml_time_us() - t_step_start;
                    train_time_us += t_step_us;

                    stats.total_pp_us += plus.t_pp_us + minus.t_pp_us;
                    stats.total_pp_tokens += plus.n_tokens + minus.n_tokens;
                    stats.total_step_us += t_step_us;
                    stats.total_prepare_plus_us += t_prepare_plus_us;
                    stats.total_prepare_minus_us += t_prepare_minus_us;
                    stats.total_prepare_update_us += t_prepare_update_us;
                    stats.total_push_plus_us += t_push_plus_us;
                    stats.total_push_minus_us += t_push_minus_us;
                    stats.total_forwards += 2;

                    const float loss_delta = plus.loss - minus.loss;
                    append_train_log(train_log, step + 1, 0.5f * (plus.loss + minus.loss), t_step_us / 1000.0);
                    LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                        "t_pp_plus=%.3f ms t_pp_minus=%.3f ms | step=%.3f ms\n",
                            step + 1, opts.steps, plus.loss, minus.loss, (double) loss_delta, (double) g,
                            plus.t_pp_us / 1000.0, minus.t_pp_us / 1000.0, t_step_us / 1000.0);
                    LOG("  serial prep: plus=%.3f ms minus=%.3f ms update=%.3f ms | push: plus=%.3f ms minus=%.3f ms\n",
                            t_prepare_plus_us / 1000.0, t_prepare_minus_us / 1000.0, t_prepare_update_us / 1000.0,
                            t_push_plus_us / 1000.0, t_push_minus_us / 1000.0);
                    maybe_eval(step + 1);
                }
            } else {
                for (int step = 0; step < opts.steps; ++step) {
                    sst2_batch_data data = sst2_make_train_batch(train_data, opts.batch_size, step);
                    const int64_t t_step_start = ggml_time_us();

                    const int64_t t_prepare_b_pair_us = make_antithetic_slots_from_master_regen_noise(lora_b_tensors, step, opts.epsilon);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_plus_work", "plus_rm", step);
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_minus_work", "minus_rm", step);
                    }
                    const int64_t t_push_b_pair_us = push_antithetic_b_tensors(lora_b_tensors, "regen-b-pair");
                    sst2_debug_state plus_debug = make_debug_state("step1_plus", opts.debug_mnn && step == 0);
                    sst2_debug_state minus_debug = make_debug_state("step1_minus", opts.debug_mnn && step == 0);
                    const sst2_paired_forward_result paired = sst2_run_paired_forward(ctx, batch, mem, params.n_batch, n_vocab, data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &plus_debug, &minus_debug);

                    const float g = (paired.loss_plus - paired.loss_minus) / (2.0f * opts.epsilon);
                    const int64_t t_prepare_update_us = apply_update_to_master_regen_noise(lora_b_tensors, step, -opts.lr * g);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_master_after_update", "master", step);
                    }
                    const int64_t t_step_us = ggml_time_us() - t_step_start;
                    train_time_us += t_step_us;

                    stats.total_pp_us += paired.t_pp_us;
                    stats.total_pp_tokens += paired.n_tokens;
                    stats.total_step_us += t_step_us;
                    stats.total_prepare_plus_us += t_prepare_b_pair_us;
                    stats.total_prepare_update_us += t_prepare_update_us;
                    stats.total_push_antithetic_b_us += t_push_b_pair_us;
                    stats.total_forwards += 1;

                    const float loss_delta = paired.loss_plus - paired.loss_minus;
                    append_train_log(train_log, step + 1, 0.5f * (paired.loss_plus + paired.loss_minus), t_step_us / 1000.0);
                    LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                        "t_pp_paired=%.3f ms | step=%.3f ms\n",
                            step + 1, opts.steps, paired.loss_plus, paired.loss_minus, (double) loss_delta, (double) g,
                            paired.t_pp_us / 1000.0, t_step_us / 1000.0);
                    LOG("  antithetic serial prep: b_pair=%.3f ms update=%.3f ms | push_b_pair=%.3f ms\n",
                            t_prepare_b_pair_us / 1000.0, t_prepare_update_us / 1000.0, t_push_b_pair_us / 1000.0);
                    maybe_eval(step + 1);
                }
            }
        } else {
            if (!opts.antithetic) {
                const int64_t t_initial_noise_us = sample_noise_for_step(lora_b_tensors, lora_noise_slot::A, 0);
                stats.total_prepare_noise_us += t_initial_noise_us;
                stats.total_step_us += t_initial_noise_us;
                train_time_us += t_initial_noise_us;
                LOG("pipeline initial noise prepare: %.3f ms\n", t_initial_noise_us / 1000.0);

                for (int step = 0; step < opts.steps; ++step) {
                    sst2_batch_data data = sst2_make_train_batch(train_data, opts.batch_size, step);
                    const lora_noise_slot cur_noise  = (step % 2 == 0) ? lora_noise_slot::A : lora_noise_slot::B;
                    const lora_noise_slot next_noise = (step % 2 == 0) ? lora_noise_slot::B : lora_noise_slot::A;
                    const int64_t t_step_start = ggml_time_us();

                    const int64_t t_prepare_plus_us = make_slot_from_master(lora_b_tensors, lora_work_slot::PLUS, cur_noise, +opts.epsilon);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_plus_work", "plus", step);
                    }
                    const int64_t t_push_plus_us = push_b_tensors_from(lora_b_tensors, lora_push_source::PLUS, "plus");

                    auto minus_future = std::async(std::launch::async, [&lora_b_tensors, cur_noise, epsilon = opts.epsilon]() {
                        return make_slot_from_master(lora_b_tensors, lora_work_slot::MINUS, cur_noise, -epsilon);
                    });

                    sst2_forward_result plus;
                    sst2_debug_state plus_debug = make_debug_state("step1_plus", opts.debug_mnn && step == 0);
                    plus.t_pp_us = sst2_run_decode(ctx, batch, mem, params.n_batch, data, false, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU);
                    plus.n_tokens = data.n_tokens;
                    if (plus_debug.enabled) {
                        (void) sst2_loss_from_current_logits(ctx, n_vocab, data, 0, &plus_debug);
                    }

                    const int64_t t_copy_plus_logits_start = ggml_time_us();
                    auto plus_logits = sst2_copy_loss_logits_from_current(ctx, n_vocab, data, 0);
                    const int64_t t_copy_plus_logits_us = ggml_time_us() - t_copy_plus_logits_start;
                    auto plus_loss_future = std::async(std::launch::async, [snap = std::move(plus_logits)]() {
                        return sst2_loss_from_snapshot(snap);
                    });

                    const int64_t t_wait_minus_start = ggml_time_us();
                    const int64_t t_prepare_minus_us = minus_future.get();
                    const int64_t t_wait_minus_ready_us = ggml_time_us() - t_wait_minus_start;
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_minus_work", "minus", step);
                    }
                    const int64_t t_push_minus_us = push_b_tensors_from(lora_b_tensors, lora_push_source::MINUS, "minus");

                    const bool has_next_step = step + 1 < opts.steps;
                    std::future<int64_t> next_noise_future;
                    if (has_next_step) {
                        next_noise_future = std::async(std::launch::async, [&lora_b_tensors, next_noise, next_step = step + 1]() {
                            return sample_noise_for_step(lora_b_tensors, next_noise, next_step);
                        });
                    }

                    sst2_debug_state minus_debug = make_debug_state("step1_minus", opts.debug_mnn && step == 0);
                    const sst2_forward_result minus = sst2_run_forward(ctx, batch, mem, params.n_batch, n_vocab, data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &minus_debug);

                    const int64_t t_wait_plus_loss_start = ggml_time_us();
                    plus.loss = plus_loss_future.get();
                    const int64_t t_wait_plus_loss_ready_us = ggml_time_us() - t_wait_plus_loss_start;
                    const float g = (plus.loss - minus.loss) / (2.0f * opts.epsilon);

                    apply_update_to_master_from_slot(lora_b_tensors, cur_noise, -opts.lr * g);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_master_after_update", "master", step);
                    }

                    int64_t t_prepare_next_noise_us = 0;
                    int64_t t_wait_next_noise_ready_us = 0;
                    if (has_next_step) {
                        const int64_t t_wait_next_start = ggml_time_us();
                        t_prepare_next_noise_us = next_noise_future.get();
                        t_wait_next_noise_ready_us = ggml_time_us() - t_wait_next_start;
                        stats.total_prepare_noise_us += t_prepare_next_noise_us;
                    }

                    const int64_t t_step_us = ggml_time_us() - t_step_start;
                    train_time_us += t_step_us;

                    stats.total_pp_us += plus.t_pp_us + minus.t_pp_us;
                    stats.total_pp_tokens += plus.n_tokens + minus.n_tokens;
                    stats.total_step_us += t_step_us;
                    stats.total_prepare_plus_us += t_prepare_plus_us;
                    stats.total_prepare_minus_us += t_prepare_minus_us;
                    stats.total_prepare_next_noise_us += t_prepare_next_noise_us;
                    stats.total_wait_minus_ready_us += t_wait_minus_ready_us;
                    stats.total_wait_next_noise_ready_us += t_wait_next_noise_ready_us;
                    stats.total_wait_plus_loss_ready_us += t_wait_plus_loss_ready_us;
                    stats.total_copy_plus_logits_us += t_copy_plus_logits_us;
                    stats.total_push_plus_us += t_push_plus_us;
                    stats.total_push_minus_us += t_push_minus_us;
                    stats.total_forwards += 2;

                    const float loss_delta = plus.loss - minus.loss;
                    append_train_log(train_log, step + 1, 0.5f * (plus.loss + minus.loss), t_step_us / 1000.0);
                    LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                        "t_pp_plus=%.3f ms t_pp_minus=%.3f ms | step=%.3f ms\n",
                            step + 1, opts.steps, plus.loss, minus.loss, (double) loss_delta, (double) g,
                            plus.t_pp_us / 1000.0, minus.t_pp_us / 1000.0, t_step_us / 1000.0);
                    LOG("  pipeline prep: plus=%.3f ms minus=%.3f ms next_noise=%.3f ms | copy_plus_logits=%.3f ms wait_minus=%.3f ms wait_next_noise=%.3f ms wait_plus_loss=%.3f ms\n",
                            t_prepare_plus_us / 1000.0, t_prepare_minus_us / 1000.0, t_prepare_next_noise_us / 1000.0,
                            t_copy_plus_logits_us / 1000.0, t_wait_minus_ready_us / 1000.0,
                            t_wait_next_noise_ready_us / 1000.0, t_wait_plus_loss_ready_us / 1000.0);
                    maybe_eval(step + 1);
                }
            } else {
                const int64_t t_initial_noise_us = sample_noise_for_step_tensorwise(lora_b_tensors, lora_noise_slot::A, 0);
                stats.total_prepare_noise_us += t_initial_noise_us;
                stats.total_step_us += t_initial_noise_us;
                train_time_us += t_initial_noise_us;
                LOG("antithetic pipeline initial noise prepare: %.3f ms\n", t_initial_noise_us / 1000.0);

                for (int step = 0; step < opts.steps; ++step) {
                    sst2_batch_data data = sst2_make_train_batch(train_data, opts.batch_size, step);
                    const lora_noise_slot cur_noise  = (step % 2 == 0) ? lora_noise_slot::A : lora_noise_slot::B;
                    const lora_noise_slot next_noise = (step % 2 == 0) ? lora_noise_slot::B : lora_noise_slot::A;
                    const int64_t t_step_start = ggml_time_us();

                    const int64_t t_prepare_b_pair_us = make_antithetic_slots_from_master_slot(lora_b_tensors, cur_noise, opts.epsilon);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_plus_work", "plus_rm", step);
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_minus_work", "minus_rm", step);
                    }
                    const int64_t t_push_b_pair_us = push_antithetic_b_tensors(lora_b_tensors, "slot-b-pair");

                    const bool has_next_step = step + 1 < opts.steps;
                    std::future<int64_t> next_noise_future;
                    if (has_next_step) {
                        next_noise_future = std::async(std::launch::async, [&lora_b_tensors, next_noise, next_step = step + 1]() {
                            return sample_noise_for_step_tensorwise(lora_b_tensors, next_noise, next_step);
                        });
                    }

                    sst2_debug_state plus_debug = make_debug_state("step1_plus", opts.debug_mnn && step == 0);
                    sst2_debug_state minus_debug = make_debug_state("step1_minus", opts.debug_mnn && step == 0);
                    const sst2_paired_forward_result paired = sst2_run_paired_forward(ctx, batch, mem, params.n_batch, n_vocab, data, pad_token, effective_debug_padding, opts.debug_logits, opts.mode != zo_mode::CPU, &plus_debug, &minus_debug);
                    const float g = (paired.loss_plus - paired.loss_minus) / (2.0f * opts.epsilon);
                    apply_update_to_master_from_slot(lora_b_tensors, cur_noise, -opts.lr * g);
                    if (opts.debug_mnn && step == 0) {
                        sst2_print_lora_b_storage_debug(lora_b_tensors, "step1_master_after_update", "master", step);
                    }

                    int64_t t_prepare_next_noise_us = 0;
                    int64_t t_wait_next_noise_ready_us = 0;
                    if (has_next_step) {
                        const int64_t t_wait_next_start = ggml_time_us();
                        t_prepare_next_noise_us = next_noise_future.get();
                        t_wait_next_noise_ready_us = ggml_time_us() - t_wait_next_start;
                        stats.total_prepare_noise_us += t_prepare_next_noise_us;
                    }

                    const int64_t t_step_us = ggml_time_us() - t_step_start;
                    train_time_us += t_step_us;

                    stats.total_pp_us += paired.t_pp_us;
                    stats.total_pp_tokens += paired.n_tokens;
                    stats.total_step_us += t_step_us;
                    stats.total_prepare_plus_us += t_prepare_b_pair_us;
                    stats.total_prepare_next_noise_us += t_prepare_next_noise_us;
                    stats.total_wait_next_noise_ready_us += t_wait_next_noise_ready_us;
                    stats.total_push_antithetic_b_us += t_push_b_pair_us;
                    stats.total_forwards += 1;

                    const float loss_delta = paired.loss_plus - paired.loss_minus;
                    append_train_log(train_log, step + 1, 0.5f * (paired.loss_plus + paired.loss_minus), t_step_us / 1000.0);
                    LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                        "t_pp_paired=%.3f ms | step=%.3f ms\n",
                            step + 1, opts.steps, paired.loss_plus, paired.loss_minus, (double) loss_delta, (double) g,
                            paired.t_pp_us / 1000.0, t_step_us / 1000.0);
                    LOG("  antithetic pipeline prep: b_pair=%.3f ms next_noise=%.3f ms | wait_next_noise=%.3f ms push_b_pair=%.3f ms\n",
                            t_prepare_b_pair_us / 1000.0, t_prepare_next_noise_us / 1000.0,
                            t_wait_next_noise_ready_us / 1000.0, t_push_b_pair_us / 1000.0);
                    maybe_eval(step + 1);
                }
            }
        }

        const double avg_speed_pp = stats.total_pp_us > 0 ?
            (double) stats.total_pp_tokens / ((double) stats.total_pp_us / 1e6) : 0.0;
        const double avg_tokens_per_fwd = stats.total_forwards > 0 ?
            (double) stats.total_pp_tokens / (double) stats.total_forwards : 0.0;
        const double avg_step_ms = (double) stats.total_step_us / (double) opts.steps / 1000.0;

        LOG("\nsummary:\n");
        LOG("  dataset             = SST2\n");
        LOG("  pipeline            = %s\n", opts.pipeline ? "true" : "false");
        LOG("  antithetic          = %s\n", opts.antithetic ? "true" : "false");
        LOG("  lora_exec           = %s\n", sst2_lora_exec_mode_name(opts.lora_exec));
        LOG("  train_time          = %.3f s\n", train_time_us / 1e6);
        LOG("  speed_pp            = %.2f t/s\n", avg_speed_pp);
        LOG("  total_pp_tokens     = %lld\n", (long long) stats.total_pp_tokens);
        LOG("  avg_step_time       = %.3f ms\n", avg_step_ms);
        if (stats.total_forwards > 0) {
            LOG("  avg_t_pp_per_fwd    = %.3f ms\n", (double) stats.total_pp_us / (double) stats.total_forwards / 1000.0);
            LOG("  avg_tokens_per_fwd  = %.3f\n", avg_tokens_per_fwd);
        }
        if (opts.antithetic) {
            LOG("  avg_prepare_b_pair  = %.3f ms\n", (double) stats.total_prepare_plus_us / (double) opts.steps / 1000.0);
        } else {
            LOG("  avg_prepare_plus    = %.3f ms\n", (double) stats.total_prepare_plus_us / (double) opts.steps / 1000.0);
            LOG("  avg_prepare_minus   = %.3f ms\n", (double) stats.total_prepare_minus_us / (double) opts.steps / 1000.0);
        }
        LOG("  avg_prepare_update  = %.3f ms\n", (double) stats.total_prepare_update_us / (double) opts.steps / 1000.0);
        if (opts.pipeline) {
            const double denom = (double) std::max(1, opts.steps - 1);
            LOG("  avg_prepare_noise   = %.3f ms\n", (double) stats.total_prepare_noise_us / (double) opts.steps / 1000.0);
            LOG("  avg_prepare_next_noise = %.3f ms\n", (double) stats.total_prepare_next_noise_us / denom / 1000.0);
            LOG("  avg_copy_plus_logits = %.3f ms\n", (double) stats.total_copy_plus_logits_us / (double) opts.steps / 1000.0);
            LOG("  avg_wait_minus_ready = %.3f ms\n", (double) stats.total_wait_minus_ready_us / (double) opts.steps / 1000.0);
            LOG("  avg_wait_next_noise_ready = %.3f ms\n", (double) stats.total_wait_next_noise_ready_us / denom / 1000.0);
            LOG("  avg_wait_plus_loss_ready = %.3f ms\n", (double) stats.total_wait_plus_loss_ready_us / (double) opts.steps / 1000.0);
        }
        if (opts.lora_exec == sst2_lora_exec_mode::MERGE_REPACK) {
            LOG("  avg_merge_plus      = %.3f ms\n", (double) stats.total_merge_plus_us / (double) opts.steps / 1000.0);
            LOG("  avg_merge_minus     = %.3f ms\n", (double) stats.total_merge_minus_us / (double) opts.steps / 1000.0);
        } else if (opts.antithetic) {
            LOG("  avg_push_b_pair     = %.3f ms\n", (double) stats.total_push_antithetic_b_us / (double) opts.steps / 1000.0);
        } else {
            LOG("  avg_push_plus       = %.3f ms\n", (double) stats.total_push_plus_us / (double) opts.steps / 1000.0);
            LOG("  avg_push_minus      = %.3f ms\n", (double) stats.total_push_minus_us / (double) opts.steps / 1000.0);
        }
        LOG("  peak_pss_sampled    = %lld KiB\n", (long long) pss_sampler.peak());
        LOG("\n");

        if (batch_inited) {
            llama_batch_free(batch);
        }
        init.reset();
        if (backend_inited) {
            llama_backend_free();
        }
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("main: %s\n", e.what());
        if (batch_inited) {
            llama_batch_free(batch);
        }
        if (backend_inited) {
            llama_backend_free();
        }
        return 1;
    }
}
