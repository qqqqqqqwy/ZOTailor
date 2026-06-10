#include "common.h"
#include "ggml-backend.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
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

static const uint64_t DEFAULT_SEED = 1337;

struct mezo_options {
    std::string model_path;
    int         seq_len    = 32;
    int         batch_size = 4;
    int         steps      = 5;
    float       epsilon    = 1e-3f;
    float       lr         = 1e-7f;
    uint64_t    seed       = DEFAULT_SEED;
    int         threads    = 0;
};

struct trainable_tensor {
    std::string  name;
    ggml_tensor * tensor = nullptr;
    ggml_type    type   = GGML_TYPE_COUNT;
    int64_t      n_elms = 0;
    size_t       nbytes = 0;
    size_t       index  = 0;
};

struct forward_result {
    float   loss = 0.0f;
    int64_t time_us = 0;
};

struct timing_totals {
    int64_t perturb_us = 0;
    int64_t infer_us   = 0;
    int64_t step_us    = 0;
};

static void print_usage(char ** argv) {
    std::fprintf(stderr,
        "\nusage:\n"
        "  %s -m <model-f16.gguf> [options]\n\n"
        "options:\n"
        "  -m, --model <path>      path to FP16 GGUF model (required)\n"
        "  --seq-len <N>          dummy tokens per sequence (default: 32)\n"
        "  --batch-size <N>       number of dummy sequences (default: 4)\n"
        "  --steps <N>            MeZO steps (default: 5)\n"
        "  --epsilon <F>          perturbation scale (default: 1e-3)\n"
        "  --lr <F>               learning rate (default: 1e-7)\n"
        "  --seed <N>             random seed (default: 1337)\n"
        "  --threads <N>          CPU threads, 0 uses llama.cpp default\n"
        "  -h, --help             show this help\n\n"
        "example:\n"
        "  %s -m /path/to/model-f16.gguf --seq-len 32 --batch-size 4 --steps 5\n\n",
        argv[0], argv[0]);
}

static bool parse_int_arg(const char * text, int & dst, const char * name) {
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
            value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "invalid integer for %s: %s\n", name, text);
        return false;
    }
    dst = (int) value;
    return true;
}

static bool parse_u64_arg(const char * text, uint64_t & dst, const char * name) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        std::fprintf(stderr, "invalid integer for %s: %s\n", name, text);
        return false;
    }
    dst = (uint64_t) value;
    return true;
}

static bool parse_float_arg(const char * text, float & dst, const char * name) {
    char * end = nullptr;
    errno = 0;
    const float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value)) {
        std::fprintf(stderr, "invalid float for %s: %s\n", name, text);
        return false;
    }
    dst = value;
    return true;
}

static bool parse_args(int argc, char ** argv, mezo_options & opts) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            print_usage(argv);
            std::exit(0);
        } else if (std::strcmp(arg, "-m") == 0 || std::strcmp(arg, "--model") == 0) {
            const char * value = need_value(arg);
            if (value == nullptr) {
                return false;
            }
            opts.model_path = value;
        } else if (std::strcmp(arg, "--seq-len") == 0) {
            const char * value = need_value(arg);
            if (value == nullptr || !parse_int_arg(value, opts.seq_len, arg)) {
                return false;
            }
        } else if (std::strcmp(arg, "--batch-size") == 0) {
            const char * value = need_value(arg);
            if (value == nullptr || !parse_int_arg(value, opts.batch_size, arg)) {
                return false;
            }
        } else if (std::strcmp(arg, "--steps") == 0) {
            const char * value = need_value(arg);
            if (value == nullptr || !parse_int_arg(value, opts.steps, arg)) {
                return false;
            }
        } else if (std::strcmp(arg, "--epsilon") == 0) {
            const char * value = need_value(arg);
            if (value == nullptr || !parse_float_arg(value, opts.epsilon, arg)) {
                return false;
            }
        } else if (std::strcmp(arg, "--lr") == 0) {
            const char * value = need_value(arg);
            if (value == nullptr || !parse_float_arg(value, opts.lr, arg)) {
                return false;
            }
        } else if (std::strcmp(arg, "--seed") == 0) {
            const char * value = need_value(arg);
            if (value == nullptr || !parse_u64_arg(value, opts.seed, arg)) {
                return false;
            }
        } else if (std::strcmp(arg, "--threads") == 0) {
            const char * value = need_value(arg);
            if (value == nullptr || !parse_int_arg(value, opts.threads, arg)) {
                return false;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            return false;
        }
    }

    if (opts.model_path.empty()) {
        std::fprintf(stderr, "missing required -m/--model\n");
        return false;
    }
    if (opts.seq_len < 2) {
        std::fprintf(stderr, "--seq-len must be >= 2\n");
        return false;
    }
    if (opts.batch_size < 1) {
        std::fprintf(stderr, "--batch-size must be >= 1\n");
        return false;
    }
    if (opts.steps < 1) {
        std::fprintf(stderr, "--steps must be >= 1\n");
        return false;
    }
    if (!(opts.epsilon > 0.0f)) {
        std::fprintf(stderr, "--epsilon must be > 0\n");
        return false;
    }
    if (!std::isfinite(opts.lr)) {
        std::fprintf(stderr, "--lr must be finite\n");
        return false;
    }
    if (opts.threads < 0) {
        std::fprintf(stderr, "--threads must be >= 0\n");
        return false;
    }
    return true;
}

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static uint32_t derive_seed(uint64_t seed, int step, size_t tensor_index) {
    uint64_t x = seed;
    x ^= 0xd1b54a32d192ed03ULL * (uint64_t) (step + 1);
    x ^= 0xabc98388fb8fac03ULL * (uint64_t) (tensor_index + 1);
    return (uint32_t) splitmix64(x);
}

static std::vector<llama_token> make_dummy_tokens(int32_t n_vocab, int batch_size, int seq_len, uint64_t seed, int step) {
    std::mt19937 rng((uint32_t) splitmix64(seed ^ (0x544f4b53ULL + (uint64_t) step)));
    std::uniform_int_distribution<int32_t> dist(0, n_vocab - 1);

    std::vector<llama_token> tokens((size_t) batch_size * (size_t) seq_len);
    for (llama_token & token : tokens) {
        token = (llama_token) dist(rng);
    }
    return tokens;
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
    return (float) (std::log(sum) + (double) max_logit - (double) logits[label]);
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
            std::fprintf(stderr, "llama_decode failed, ret = %d\n", ret);
            return false;
        }
    }
    return true;
}

static forward_result run_forward(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len) {
    common_batch_clear(batch);

    for (int s = 0; s < batch_size; ++s) {
        for (int t = 0; t < seq_len; ++t) {
            common_batch_add(batch, tokens[(size_t) s * seq_len + t], t, { (llama_seq_id) s }, true);
        }
    }

    llama_memory_clear(mem, false);

    forward_result res;
    const int64_t t_start = ggml_time_us();
    if (!decode_helper(ctx, batch, n_batch)) {
        throw std::runtime_error("llama_decode failed");
    }
    llama_synchronize(ctx);
    res.time_us = ggml_time_us() - t_start;

    double loss_sum = 0.0;
    int64_t loss_count = 0;
    for (int s = 0; s < batch_size; ++s) {
        for (int t = 0; t < seq_len - 1; ++t) {
            const int row = s * seq_len + t;
            const llama_token label = tokens[(size_t) s * seq_len + (t + 1)];
            const float * logits = llama_get_logits_ith(ctx, row);
            if (logits == nullptr) {
                throw std::runtime_error("failed to fetch logits row");
            }
            loss_sum += logits_cross_entropy(logits, n_vocab, label);
            ++loss_count;
        }
    }

    res.loss = (float) (loss_sum / (double) loss_count);
    return res;
}

static std::vector<trainable_tensor> collect_trainable_tensors(const llama_model * model) {
    std::vector<trainable_tensor> result;
    const auto & tensors = llama_internal_get_tensor_map(model);

    size_t skipped = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_tensor * tensor = tensors[i].second;
        if (tensor == nullptr || tensor->buffer == nullptr) {
            ++skipped;
            continue;
        }

        if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) {
            ++skipped;
            continue;
        }

        trainable_tensor cur;
        cur.name   = tensors[i].first;
        cur.tensor = tensor;
        cur.type   = tensor->type;
        cur.n_elms = ggml_nelements(tensor);
        cur.nbytes = ggml_nbytes(tensor);
        cur.index  = result.size();
        result.push_back(std::move(cur));
    }

    std::fprintf(stderr, "trainable tensors: %zu, skipped tensors: %zu\n", result.size(), skipped);
    return result;
}

static int64_t apply_noise_to_tensor(const trainable_tensor & tw, int step, float coeff, uint64_t seed) {
    std::mt19937 rng(derive_seed(seed, step, tw.index));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int64_t t_start = ggml_time_us();

    if (tw.type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data((size_t) tw.n_elms);
        ggml_backend_tensor_get(tw.tensor, data.data(), 0, tw.nbytes);
        for (ggml_fp16_t & v_h : data) {
            const float v = ggml_fp16_to_fp32(v_h) + coeff * dist(rng);
            v_h = ggml_fp32_to_fp16(v);
        }
        ggml_backend_tensor_set(tw.tensor, data.data(), 0, tw.nbytes);
    } else if (tw.type == GGML_TYPE_F32) {
        std::vector<float> data((size_t) tw.n_elms);
        ggml_backend_tensor_get(tw.tensor, data.data(), 0, tw.nbytes);
        for (float & v : data) {
            v += coeff * dist(rng);
        }
        ggml_backend_tensor_set(tw.tensor, data.data(), 0, tw.nbytes);
    }

    return ggml_time_us() - t_start;
}

static int64_t apply_noise_to_tensors(const std::vector<trainable_tensor> & tensors, int step, float coeff, uint64_t seed) {
    int64_t total_us = 0;
    for (const auto & tw : tensors) {
        total_us += apply_noise_to_tensor(tw, step, coeff, seed);
    }
    return total_us;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    mezo_options opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch batch = {};
    bool batch_inited = false;

    try {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        // MeZO updates model tensors in place, so mmap-backed weights must be disabled.
        model_params.use_mmap = false;

        model = llama_model_load_from_file(opts.model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load model");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (n_vocab <= 0) {
            throw std::runtime_error("invalid vocabulary size");
        }

        const int total_tokens = opts.batch_size * opts.seq_len;

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx       = (uint32_t) total_tokens;
        ctx_params.n_batch     = (uint32_t) total_tokens;
        ctx_params.n_ubatch    = (uint32_t) total_tokens;
        ctx_params.n_seq_max   = (uint32_t) opts.batch_size;
        ctx_params.offload_kqv = false;
        ctx_params.op_offload  = false;
        ctx_params.no_perf     = false;
        if (opts.threads > 0) {
            ctx_params.n_threads = opts.threads;
            ctx_params.n_threads_batch = opts.threads;
        }

        ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create llama_context");
        }

        llama_memory_t mem = llama_get_memory(ctx);
        batch = llama_batch_init(total_tokens, 0, opts.batch_size);
        batch_inited = true;

        const auto trainable = collect_trainable_tensors(model);
        if (trainable.empty()) {
            throw std::runtime_error("no F16/F32 trainable tensors found; use an FP16 GGUF model");
        }

        uint64_t n_trainable_elms = 0;
        size_t trainable_bytes = 0;
        for (const auto & tw : trainable) {
            n_trainable_elms += (uint64_t) tw.n_elms;
            trainable_bytes  += tw.nbytes;
        }

        std::fprintf(stderr, "\nllama-zoo-mezo configuration:\n");
        std::fprintf(stderr, "  model       = %s\n", opts.model_path.c_str());
        std::fprintf(stderr, "  batch_size  = %d\n", opts.batch_size);
        std::fprintf(stderr, "  seq_len     = %d\n", opts.seq_len);
        std::fprintf(stderr, "  steps       = %d\n", opts.steps);
        std::fprintf(stderr, "  epsilon     = %.6g\n", (double) opts.epsilon);
        std::fprintf(stderr, "  lr          = %.6g\n", (double) opts.lr);
        std::fprintf(stderr, "  seed        = %llu\n", (unsigned long long) opts.seed);
        std::fprintf(stderr, "  threads     = %d\n", opts.threads);
        std::fprintf(stderr, "  use_mmap    = false\n");
        std::fprintf(stderr, "  trainable   = %zu tensors, %.3f M elems, %.3f MiB\n",
                trainable.size(),
                (double) n_trainable_elms / 1e6,
                (double) trainable_bytes / 1024.0 / 1024.0);
        std::fprintf(stderr, "\n");

        timing_totals totals;
        int64_t total_forward_tokens = 0;

        for (int step = 0; step < opts.steps; ++step) {
            const auto tokens = make_dummy_tokens(n_vocab, opts.batch_size, opts.seq_len, opts.seed, step);
            const int64_t t_step_start = ggml_time_us();

            const int64_t t_apply_plus_us = apply_noise_to_tensors(trainable, step, +opts.epsilon, opts.seed);
            const forward_result plus = run_forward(ctx, batch, mem, total_tokens, n_vocab, tokens, opts.batch_size, opts.seq_len);

            const int64_t t_apply_minus_us = apply_noise_to_tensors(trainable, step, -2.0f * opts.epsilon, opts.seed);
            const forward_result minus = run_forward(ctx, batch, mem, total_tokens, n_vocab, tokens, opts.batch_size, opts.seq_len);

            const float g = (plus.loss - minus.loss) / (2.0f * opts.epsilon);
            const int64_t t_apply_update_us = apply_noise_to_tensors(trainable, step, opts.epsilon - opts.lr * g, opts.seed);

            const int64_t t_step_us = ggml_time_us() - t_step_start;
            const int64_t t_perturb_us = t_apply_plus_us + t_apply_minus_us + t_apply_update_us;
            const int64_t t_infer_us = plus.time_us + minus.time_us;

            totals.perturb_us += t_perturb_us;
            totals.infer_us   += t_infer_us;
            totals.step_us    += t_step_us;
            total_forward_tokens += 2LL * total_tokens;

            const double infer_speed = (double) (2 * total_tokens) / ((double) t_infer_us / 1e6);
            std::fprintf(stderr,
                    "step %d/%d: loss_plus=%.6f loss_minus=%.6f g=%.8e | "
                    "perturb=%.3f ms infer=%.3f ms total=%.3f ms speed=%.2f t/s "
                    "(plus=%.3f ms minus=%.3f ms update=%.3f ms)\n",
                    step + 1, opts.steps,
                    (double) plus.loss, (double) minus.loss, (double) g,
                    t_perturb_us / 1000.0,
                    t_infer_us / 1000.0,
                    t_step_us / 1000.0,
                    infer_speed,
                    t_apply_plus_us / 1000.0,
                    t_apply_minus_us / 1000.0,
                    t_apply_update_us / 1000.0);
        }

        const double avg_perturb_ms = (double) totals.perturb_us / 1000.0 / (double) opts.steps;
        const double avg_infer_ms   = (double) totals.infer_us   / 1000.0 / (double) opts.steps;
        const double avg_step_ms    = (double) totals.step_us    / 1000.0 / (double) opts.steps;
        const double avg_speed      = (double) total_forward_tokens / ((double) totals.infer_us / 1e6);

        std::fprintf(stderr, "\nMeZO summary:\n");
        std::fprintf(stderr, "  avg perturb time / step = %.3f ms\n", avg_perturb_ms);
        std::fprintf(stderr, "  avg inference time / step = %.3f ms\n", avg_infer_ms);
        std::fprintf(stderr, "  avg total time / step = %.3f ms\n", avg_step_ms);
        std::fprintf(stderr, "  avg prefill speed = %.2f tokens/s\n", avg_speed);
        std::fprintf(stderr, "\n");

        llama_batch_free(batch);
        batch_inited = false;
        llama_free(ctx);
        ctx = nullptr;
        llama_model_free(model);
        model = nullptr;
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        if (batch_inited) {
            llama_batch_free(batch);
        }
        if (ctx != nullptr) {
            llama_free(ctx);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
        llama_backend_free();
        return 1;
    }
}
