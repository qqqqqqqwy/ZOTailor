#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

int zoo_lora_fa_benchmark_main(int argc, char ** argv);

#define main zoo_lora_fa_benchmark_main
#include "zo-lora-fa.cpp"
#undef main

namespace {

struct base_lora_options {
    int batch_size = 4;
    int seq_len    = 32;
    int iters      = 10;
};

static void base_lora_print_usage(int /*argc*/, char ** argv) {
    LOG("\n");
    LOG("usage:\n");
    LOG("  %s -m <model.gguf> [--lora <adapter.gguf>] --device HTP0 -ngl 99 [options]\n", argv[0]);
    LOG("\n");
    LOG("base-lora benchmark options:\n");
    LOG("  --batch-size <N>   number of independent dummy sequences (default: 4)\n");
    LOG("  --seq-len <N>      dummy tokens per sequence (default: 32)\n");
    LOG("  --iters <N>        measured prefill iterations (default: 10)\n");
    LOG("\n");
    LOG("examples:\n");
    LOG("  %s -m ./model-q4_0.gguf --device HTP0 -ngl 99 --batch-size 4 --seq-len 32\n", argv[0]);
    LOG("  %s -m ./model-q4_0.gguf --lora ./lora.gguf --device HTP0 -ngl 99 --batch-size 4 --seq-len 32\n", argv[0]);
    LOG("\n");
}

static bool base_lora_parse_args(int argc, char ** argv, base_lora_options & opts, std::vector<char *> & filtered_argv) {
    filtered_argv.clear();
    filtered_argv.reserve(argc);
    filtered_argv.push_back(argv[0]);

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
            if (split_arg(arg, "--iters", value)) {
                if (value.empty()) {
                    value = take_next("--iters");
                }
                if (!parse_int_arg(value, opts.iters, "--iters")) {
                    return false;
                }
                continue;
            }
        } catch (const std::exception & e) {
            LOG_ERR("%s\n", e.what());
            return false;
        }

        filtered_argv.push_back(argv[i]);
    }

    if (opts.batch_size <= 0) {
        LOG_ERR("--batch-size must be > 0\n");
        return false;
    }
    if (opts.seq_len <= 0) {
        LOG_ERR("--seq-len must be > 0\n");
        return false;
    }
    if (opts.iters <= 0) {
        LOG_ERR("--iters must be > 0\n");
        return false;
    }

    return true;
}

static std::string base_lora_lora_path(const common_params & params) {
    if (params.lora_adapters.empty()) {
        return "<none>";
    }
    return params.lora_adapters.front().path;
}

static const char * base_lora_getenv_or_unset(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "<unset>";
}

static void base_lora_prepare_hexagon_env() {
    // These must be visible before common_params_parse(), because --device HTP0
    // can instantiate the Hexagon backend while parsing common arguments.
    set_env_var("GGML_HEXAGON_FUSED_LORA", "0");
    set_env_var("GGML_HEXAGON_ANTITHETIC_LORA", "0");

    const char * prompt_fa = std::getenv("PROMPT_FA");
    const char * hex_prompt_fa = std::getenv("GGML_HEXAGON_FLASH_ATTN_PROMPT");
    if (prompt_fa != nullptr && hex_prompt_fa == nullptr) {
        set_env_var("GGML_HEXAGON_FLASH_ATTN_PROMPT", prompt_fa);
    }
}

static int base_lora_aligned_seq_len(int batch_size, int seq_len) {
    const int g = std::gcd(batch_size, 32);
    const int align = g > 0 ? 32 / g : 1;
    return ((seq_len + align - 1) / align) * align;
}

static llama_token base_lora_pad_token(const llama_vocab * vocab) {
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
    return pad_token;
}

static int64_t base_lora_run_prefill_decode(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len,
        int padded_seq_len,
        llama_token pad_token) {
    common_batch_clear(batch);

    for (int s = 0; s < batch_size; ++s) {
        for (int t = 0; t < padded_seq_len; ++t) {
            const bool real_token = t < seq_len;
            const llama_token token = real_token ? tokens[(size_t) s * seq_len + t] : pad_token;
            const bool logits = real_token && (t + 1 == seq_len);
            common_batch_add(batch, token, t, { (llama_seq_id) s }, logits);
        }
    }

    llama_memory_clear(mem, false);

    const int64_t t_start = ggml_time_us();
    if (!decode_helper(ctx, batch, n_batch)) {
        throw std::runtime_error("llama_decode failed during base-lora prefill");
    }
    llama_synchronize(ctx);
    return ggml_time_us() - t_start;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    base_lora_options opts;
    std::vector<char *> filtered_argv;

    common_params params;
    params.n_ctx       = opts.batch_size * opts.seq_len;
    params.n_batch     = opts.batch_size * opts.seq_len;
    params.n_ubatch    = opts.batch_size * opts.seq_len;
    params.n_parallel  = opts.batch_size;
    params.n_sequences = opts.batch_size;
    params.warmup      = false;

    base_lora_prepare_hexagon_env();
    common_init();

    if (!base_lora_parse_args(argc, argv, opts, filtered_argv)) {
        base_lora_print_usage(argc, argv);
        return 1;
    }

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_BENCH, base_lora_print_usage)) {
        return 1;
    }

    const int padded_seq_len = base_lora_aligned_seq_len(opts.batch_size, opts.seq_len);
    const int total_tokens = opts.batch_size * opts.seq_len;
    const int decode_tokens = opts.batch_size * padded_seq_len;
    params.n_parallel  = opts.batch_size;
    params.n_sequences = opts.batch_size;
    params.n_ctx       = std::max(params.n_ctx, decode_tokens);
    params.n_batch     = std::max(params.n_batch, decode_tokens);
    params.n_ubatch    = std::max(params.n_ubatch, decode_tokens);
    params.warmup      = false;

    bool backend_inited = false;
    bool batch_inited = false;
    llama_batch batch = {};

    try {
        if (!zo_hexagon_compiled()) {
            throw std::runtime_error("llama-zoo-base-lora requires a GGML_USE_HEXAGON build");
        }
        if (params.model.path.empty()) {
            throw std::runtime_error("missing model path: use -m <model.gguf>");
        }
        if (params.devices.empty() || params.devices[0] == nullptr) {
            throw std::runtime_error("llama-zoo-base-lora requires an explicit HTP device, e.g. --device HTP0");
        }
        if (params.lora_adapters.size() > 1) {
            throw std::runtime_error("llama-zoo-base-lora supports at most one LoRA adapter");
        }

        const bool has_lora = !params.lora_adapters.empty();

        LOG("\n");
        LOG("llama-zoo-base-lora configuration:\n");
        LOG("  model      = %s\n", params.model.path.c_str());
        LOG("  lora       = %s\n", base_lora_lora_path(params).c_str());
        LOG("  mode       = %s\n", has_lora ? "runtime-lora" : "base-only");
        LOG("  batch_size = %d\n", opts.batch_size);
        LOG("  seq_len    = %d\n", opts.seq_len);
        LOG("  padded_seq_len = %d\n", padded_seq_len);
        LOG("  iters      = %d\n", opts.iters);
        LOG("  logical_tokens/iter = %d\n", total_tokens);
        LOG("  decode_tokens/iter  = %d\n", decode_tokens);
        LOG("  n_ctx      = %d\n", params.n_ctx);
        LOG("  n_batch    = %d\n", params.n_batch);
        LOG("  n_ubatch   = %d\n", params.n_ubatch);
        LOG("  flash_attn = %s\n", llama_flash_attn_type_name(params.flash_attn_type));
        LOG("  PROMPT_FA  = %s\n", base_lora_getenv_or_unset("PROMPT_FA"));
        LOG("  GGML_HEXAGON_FLASH_ATTN_PROMPT = %s\n", base_lora_getenv_or_unset("GGML_HEXAGON_FLASH_ATTN_PROMPT"));
        LOG("  GGML_HEXAGON_FUSED_LORA        = %s\n", base_lora_getenv_or_unset("GGML_HEXAGON_FUSED_LORA"));
        LOG("  fused_lora = false\n");
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

        if (has_lora) {
            auto & loras = init->lora();
            if (loras.empty() || loras.front() == nullptr) {
                throw std::runtime_error("LoRA adapter was requested but not loaded");
            }
            llama_adapter_lora * adapter = loras.front().get();
            auto lora_b_tensors = collect_lora_b_tensors(adapter);
            log_lora_summary(lora_b_tensors);
            log_lora_placement_summary(lora_b_tensors, false, false);
        } else {
            LOG("LoRA adapter: disabled; benchmarking base model only\n");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (n_vocab <= 0) {
            throw std::runtime_error("invalid vocabulary size");
        }
        const llama_token pad_token = base_lora_pad_token(vocab);
        LOG("base-lora padding token: %d\n", pad_token);

        llama_memory_t mem = llama_get_memory(ctx);
        batch = llama_batch_init(decode_tokens, 0, opts.batch_size);
        batch_inited = true;

        LOG("running untimed warmup prefill ...\n");
        {
            const auto tokens = make_dummy_tokens_for_step(n_vocab, opts.batch_size, opts.seq_len, 0);
            (void) base_lora_run_prefill_decode(
                    ctx, batch, mem, params.n_batch, tokens,
                    opts.batch_size, opts.seq_len, padded_seq_len, pad_token);
        }

        int64_t total_us = 0;
        int64_t total_processed_tokens = 0;
        for (int iter = 0; iter < opts.iters; ++iter) {
            const auto tokens = make_dummy_tokens_for_step(n_vocab, opts.batch_size, opts.seq_len, iter + 1);
            const int64_t t_us = base_lora_run_prefill_decode(
                    ctx, batch, mem, params.n_batch, tokens,
                    opts.batch_size, opts.seq_len, padded_seq_len, pad_token);
            total_us += t_us;
            total_processed_tokens += total_tokens;

            const double speed = (double) total_tokens / ((double) t_us / 1e6);
            LOG("iter %d/%d: prefill=%.3f ms speed=%.2f tokens/s\n",
                    iter + 1, opts.iters, t_us / 1000.0, speed);
        }

        const double avg_ms = (double) total_us / 1000.0 / (double) opts.iters;
        const double avg_speed = (double) total_processed_tokens / ((double) total_us / 1e6);
        LOG("\n");
        LOG("base-lora summary:\n");
        LOG("  mode                         = %s\n", has_lora ? "runtime-lora" : "base-only");
        LOG("  iterations                   = %d\n", opts.iters);
        LOG("  tokens / iteration           = %d\n", total_tokens);
        LOG("  total processed tokens       = %lld\n", (long long) total_processed_tokens);
        LOG("  avg runtime latency / prefill = %.3f ms\n", avg_ms);
        LOG("  avg prefill speed            = %.2f tokens/s\n", avg_speed);
        LOG("\n");

        llama_batch_free(batch);
        batch_inited = false;
        init.reset();
        if (backend_inited) {
            llama_backend_free();
            backend_inited = false;
        }
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("error: %s\n", e.what());
        if (batch_inited) {
            llama_batch_free(batch);
        }
        if (backend_inited) {
            llama_backend_free();
        }
        return 1;
    }
}
