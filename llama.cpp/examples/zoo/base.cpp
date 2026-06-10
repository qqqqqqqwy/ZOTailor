#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static const char * DEFAULT_MODEL_PATH = "/data/local/tmp/gguf/TinyLlama-1.1B-Chat-v1.0-F16.gguf";
static const uint32_t DEFAULT_SEED = 1337;

enum class base_mode {
    CPU,
    COOP,
};

struct base_options {
    base_mode mode       = base_mode::CPU;
    int       steps      = 5;
    int       batch_size = 4;
    int       seq_len    = 32;
};

struct forward_result {
    float   loss    = 0.0f;
    int64_t t_pp_us = 0;
};

static bool base_hexagon_compiled() {
#ifdef GGML_USE_HEXAGON
    return true;
#else
    return false;
#endif
}

static const char * base_mode_name(base_mode mode) {
    switch (mode) {
        case base_mode::CPU:  return "cpu";
        case base_mode::COOP: return "coop";
    }
    return "unknown";
}

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void disable_lora_env() {
    set_env_var("GGML_HEXAGON_FUSED_LORA", "0");
    set_env_var("GGML_HEXAGON_ANTITHETIC_LORA", "0");
}

static void print_usage(int /*argc*/, char ** argv) {
    LOG("\n");
    LOG("usage:\n");
    LOG("  %s [common llama.cpp args] [zoo args]\n", argv[0]);
    LOG("\n");
    LOG("zoo base args:\n");
    LOG("  --mode <cpu|coop>       execution mode (default: cpu)\n");
    LOG("  --steps <N>             measured prefill steps (default: 5)\n");
    LOG("  --batch-size <N>        number of sequences per step (default: 4)\n");
    LOG("  --seq-len <N>           tokens per sequence (default: 32)\n");
    LOG("\n");
    LOG("defaults:\n");
    LOG("  -m      %s\n", DEFAULT_MODEL_PATH);
    LOG("  --lora  disabled even if passed on the command line\n");
    LOG("\n");
    LOG("examples:\n");
    LOG("  %s --mode cpu --device none -m %s\n", argv[0], DEFAULT_MODEL_PATH);
    LOG("  %s --mode coop --device HTP0 -m %s -ngl 99 -fa on\n", argv[0], DEFAULT_MODEL_PATH);
    LOG("\n");
}

static bool parse_int_arg(const std::string & value, int & dst, const char * name) {
    try {
        dst = std::stoi(value);
        return true;
    } catch (const std::exception &) {
        LOG_ERR("%s: invalid integer for %s: %s\n", __func__, name, value.c_str());
        return false;
    }
}

static bool split_arg(const std::string & arg, const char * key, std::string & value) {
    const std::string prefix = std::string(key) + "=";
    if (arg == key) {
        value.clear();
        return true;
    }
    if (arg.compare(0, prefix.size(), prefix) == 0) {
        value = arg.substr(prefix.size());
        return true;
    }
    return false;
}

static bool prescan_mode(int argc, char ** argv, base_options & opts) {
    for (int i = 1; i < argc; ++i) {
        std::string value;
        if (!split_arg(argv[i], "--mode", value)) {
            continue;
        }

        if (value.empty()) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: missing value for --mode\n", __func__);
                return false;
            }
            value = argv[++i];
        }

        if (value == "cpu") {
            opts.mode = base_mode::CPU;
        } else if (value == "coop") {
            opts.mode = base_mode::COOP;
        } else {
            LOG_ERR("%s: invalid mode: %s\n", __func__, value.c_str());
            return false;
        }
    }

    return true;
}

static bool parse_custom_args(int argc, char ** argv, base_options & opts, std::vector<char *> & filtered_argv) {
    filtered_argv.clear();
    filtered_argv.reserve(argc);
    filtered_argv.push_back(argv[0]);

    const bool strip_device_args = opts.mode == base_mode::CPU || (!base_hexagon_compiled() && opts.mode == base_mode::COOP);

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
                    opts.mode = base_mode::CPU;
                } else if (value == "coop") {
                    opts.mode = base_mode::COOP;
                } else {
                    LOG_ERR("%s: invalid mode: %s\n", __func__, value.c_str());
                    return false;
                }
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

            if (split_arg(arg, "--pipeline", value) ||
                split_arg(arg, "--antithetic", value) ||
                split_arg(arg, "--lora-exec", value)) {
                LOG_ERR("%s: %s is not supported by llama-zoo-base\n", __func__, arg.c_str());
                return false;
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

static void validate_options(const base_options & opts) {
    if (opts.steps <= 0) {
        throw std::runtime_error("steps must be > 0");
    }
    if (opts.batch_size <= 0) {
        throw std::runtime_error("batch-size must be > 0");
    }
    if (opts.seq_len < 2) {
        throw std::runtime_error("seq-len must be >= 2 for causal loss");
    }
}

static uint32_t derive_step_seed(uint32_t base_seed, int step, uint32_t salt) {
    uint32_t x = base_seed ^ salt ^ (uint32_t) step * 0x9e3779b9U;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static std::vector<llama_token> make_dummy_tokens(int32_t n_vocab, int batch_size, int seq_len, std::mt19937 & rng) {
    std::uniform_int_distribution<int32_t> dist(0, n_vocab - 1);

    std::vector<llama_token> tokens((size_t) batch_size * (size_t) seq_len);
    for (llama_token & token : tokens) {
        token = dist(rng);
    }

    return tokens;
}

static std::vector<llama_token> make_dummy_tokens_for_step(int32_t n_vocab, int batch_size, int seq_len, int step) {
    std::mt19937 rng(derive_step_seed(DEFAULT_SEED, step, 0x544f4b53U));
    return make_dummy_tokens(n_vocab, batch_size, seq_len, rng);
}

static bool decode_helper(llama_context * ctx, llama_batch & batch, int32_t n_batch) {
    for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
        const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
        };

        const int ret = llama_decode(ctx, batch_view);
        if (ret != 0) {
            LOG_ERR("%s: llama_decode failed, ret = %d\n", __func__, ret);
            return false;
        }
    }

    return true;
}

static float logits_cross_entropy(const float * logits, int32_t n_vocab, llama_token label) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp((double) logits[i] - (double) max_logit);
    }

    const double logsumexp = std::log(sum) + (double) max_logit;
    return (float) (logsumexp - (double) logits[label]);
}

static int64_t run_prefill_decode(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len) {
    common_batch_clear(batch);

    for (int s = 0; s < batch_size; ++s) {
        for (int t = 0; t < seq_len; ++t) {
            const bool logits = t == seq_len - 2;
            common_batch_add(batch, tokens[(size_t) s * seq_len + t], t, { (llama_seq_id) s }, logits);
        }
    }

    llama_memory_clear(mem, false);

    const int64_t t_start = ggml_time_us();
    if (!decode_helper(ctx, batch, n_batch)) {
        throw std::runtime_error("llama_decode failed during prefill forward");
    }
    llama_synchronize(ctx);
    return ggml_time_us() - t_start;
}

static float compute_loss_from_current_logits(
        llama_context * ctx,
        int32_t n_vocab,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len) {
    double loss_sum = 0.0;

    for (int s = 0; s < batch_size; ++s) {
        const int t = seq_len - 2;
        const int batch_token_index = s * seq_len + t;
        const llama_token label = tokens[(size_t) s * seq_len + (t + 1)];
        const float * logits = llama_get_logits_ith(ctx, batch_token_index);
        if (logits == nullptr) {
            throw std::runtime_error("failed to fetch logits row");
        }

        loss_sum += logits_cross_entropy(logits, n_vocab, label);
    }

    return (float) (loss_sum / (double) batch_size);
}

static forward_result run_prefill_forward(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len) {
    forward_result res;
    res.t_pp_us = run_prefill_decode(ctx, batch, mem, n_batch, tokens, batch_size, seq_len);
    res.loss    = compute_loss_from_current_logits(ctx, n_vocab, tokens, batch_size, seq_len);
    return res;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx       = 128;
    params.n_batch     = 128;
    params.n_ubatch    = 128;
    params.n_parallel  = 4;
    params.n_sequences = 4;

    base_options opts;

    common_init();

    if (!prescan_mode(argc, argv, opts)) {
        return 1;
    }

    disable_lora_env();

    std::vector<char *> filtered_argv;
    if (!parse_custom_args(argc, argv, opts, filtered_argv)) {
        print_usage(argc, argv);
        return 1;
    }

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    bool backend_inited = false;
    bool batch_inited = false;
    llama_batch batch = {};

    try {
        validate_options(opts);

        if (params.model.path.empty()) {
            params.model.path = DEFAULT_MODEL_PATH;
        }
        params.lora_adapters.clear();

        const int effective_batch_size = opts.batch_size;
        const int effective_total_tokens = effective_batch_size * opts.seq_len;

        params.n_parallel  = effective_batch_size;
        params.n_sequences = effective_batch_size;
        params.n_ctx       = std::max(params.n_ctx, effective_total_tokens);
        params.n_batch     = std::max(params.n_batch, effective_total_tokens);
        params.n_ubatch    = std::max(params.n_ubatch, effective_total_tokens);

        if (opts.mode == base_mode::CPU) {
            params.devices.clear();
            params.n_gpu_layers  = 0;
            params.main_gpu      = -1;
            params.fit_params    = false;
            params.no_op_offload = true;
            params.no_kv_offload = true;
            params.split_mode    = LLAMA_SPLIT_MODE_NONE;
        } else {
            if (!base_hexagon_compiled()) {
                throw std::runtime_error("coop mode requires a GGML_USE_HEXAGON build");
            }
            if (params.devices.empty() || params.devices[0] == nullptr) {
                throw std::runtime_error("coop mode requires an explicit non-CPU device, e.g. --device HTP0");
            }
        }

        LOG("\n");
        LOG("zoo-base configuration:\n");
        LOG("  mode       = %s\n", base_mode_name(opts.mode));
        LOG("  model      = %s\n", params.model.path.c_str());
        LOG("  lora       = disabled\n");
        LOG("  batch_size = %d\n", opts.batch_size);
        LOG("  seq_len    = %d\n", opts.seq_len);
        LOG("  steps      = %d\n", opts.steps);
        LOG("  seed       = %u\n", DEFAULT_SEED);
        LOG("  n_ctx      = %d\n", params.n_ctx);
        LOG("  n_batch    = %d\n", params.n_batch);
        LOG("  n_ubatch   = %d\n", params.n_ubatch);
        LOG("\n");

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

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (n_vocab <= 0) {
            throw std::runtime_error("invalid vocabulary size");
        }

        llama_memory_t mem = llama_get_memory(ctx);
        const int total_tokens = opts.batch_size * opts.seq_len;
        batch = llama_batch_init(effective_total_tokens, 0, effective_batch_size);
        batch_inited = true;

        {
            std::mt19937 warmup_rng(DEFAULT_SEED ^ 0x9e3779b9U);
            const auto warmup_tokens = make_dummy_tokens(n_vocab, opts.batch_size, opts.seq_len, warmup_rng);
            LOG("running untimed warmup prefill ...\n");
            (void) run_prefill_forward(ctx, batch, mem, params.n_batch, n_vocab, warmup_tokens, opts.batch_size, opts.seq_len);
        }

        int64_t total_pp_us = 0;
        int64_t total_step_us = 0;

        for (int step = 0; step < opts.steps; ++step) {
            const auto tokens = make_dummy_tokens_for_step(n_vocab, opts.batch_size, opts.seq_len, step);
            const int64_t t_step_start = ggml_time_us();
            const forward_result fwd = run_prefill_forward(ctx, batch, mem, params.n_batch, n_vocab, tokens, opts.batch_size, opts.seq_len);
            const int64_t t_step_us = ggml_time_us() - t_step_start;

            total_pp_us += fwd.t_pp_us;
            total_step_us += t_step_us;

            const double speed_pp = (double) total_tokens / ((double) fwd.t_pp_us / 1e6);
            LOG("step %d/%d: loss=%.6f | t_pp=%.3f ms speed_pp=%.2f t/s | step=%.3f ms\n",
                    step + 1, opts.steps,
                    fwd.loss,
                    fwd.t_pp_us / 1000.0, speed_pp,
                    t_step_us / 1000.0);
        }

        const double avg_speed_pp = (double) (opts.steps * total_tokens) / ((double) total_pp_us / 1e6);
        const double avg_step_ms  = (double) total_step_us / (double) opts.steps / 1000.0;

        LOG("\n");
        LOG("summary:\n");
        LOG("  mode               = %s\n", base_mode_name(opts.mode));
        LOG("  lora               = disabled\n");
        LOG("  speed_pp           = %.2f t/s\n", avg_speed_pp);
        LOG("  avg_step_time      = %.3f ms\n", avg_step_ms);
        LOG("  avg_t_pp_per_fwd   = %.3f ms\n", (double) total_pp_us / (double) opts.steps / 1000.0);
        LOG("\n");

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
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        if (batch_inited) {
            llama_batch_free(batch);
        }
        if (backend_inited) {
            llama_backend_free();
        }
        return 1;
    }
}
