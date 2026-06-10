#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-adapter.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

static const char * DEFAULT_MODEL_PATH = "/data/local/tmp/gguf/TinyLlama-1.1B-Chat-v1.0-F16.gguf";
static const char * DEFAULT_LORA_PATH  = "/data/local/tmp/gguf/lora.gguf";
static const uint32_t DEFAULT_SEED     = 1337;

enum class zo_mode {
    CPU,
    COOP,
};

enum class lora_exec_mode {
    RUNTIME,
    FUSED_HTP,
};

enum class lora_noise_slot {
    A,
    B,
};

enum class lora_work_slot {
    PLUS,
    MINUS,
};

enum class lora_push_source {
    WORK,
    MASTER,
    PLUS,
    MINUS,
};

struct zo_options {
    zo_mode        mode          = zo_mode::CPU;
    lora_exec_mode lora_exec     = lora_exec_mode::RUNTIME;
    bool           lora_exec_set = false;
    int            steps         = 5;
    int            batch_size    = 4;
    int            seq_len       = 32;
    float          epsilon       = 1e-2f;
    float          lr            = 5e-5f;
    bool           pipeline      = false;
    bool           antithetic    = false;
};

struct forward_result {
    float   loss    = 0.0f;
    int64_t t_pp_us = 0;
};

struct paired_forward_result {
    float   loss_plus  = 0.0f;
    float   loss_minus = 0.0f;
    int64_t t_pp_us    = 0;
};

struct logits_snapshot {
    int32_t n_vocab = 0;
    int32_t n_rows  = 0;
    std::vector<float> logits;
    std::vector<llama_token> labels;
};

struct lora_b_tensor {
    std::string       name;
    ggml_tensor *     tensor = nullptr;
    ggml_tensor *     b_plus_tensor = nullptr;
    ggml_tensor *     b_minus_tensor = nullptr;
    ggml_tensor *     b_plus_rm_tensor = nullptr;
    ggml_tensor *     b_minus_rm_tensor = nullptr;
    ggml_type         type   = GGML_TYPE_COUNT;
    int64_t           n_elms = 0;
    size_t            nbytes = 0;
    size_t            rm_nbytes = 0;
    std::vector<float> master;
    std::vector<float> work;
    std::vector<float> noise_a;
    std::vector<float> noise_b;
    std::vector<float> plus;
    std::vector<float> minus;
    std::vector<float> plus_rm;
    std::vector<float> minus_rm;
    std::vector<uint8_t> io_buf;
};

struct perf_stats {
    int64_t total_pp_us                    = 0;
    int64_t total_step_us                  = 0;
    int64_t total_push_plus_us             = 0;
    int64_t total_push_minus_us            = 0;
    int64_t total_push_antithetic_b_us     = 0;
    int64_t total_prepare_noise_us         = 0;
    int64_t total_prepare_plus_us          = 0;
    int64_t total_prepare_minus_us         = 0;
    int64_t total_prepare_update_us        = 0;
    int64_t total_prepare_next_noise_us    = 0;
    int64_t total_wait_minus_ready_us      = 0;
    int64_t total_wait_next_noise_ready_us = 0;
    int64_t total_wait_plus_loss_ready_us  = 0;
    int64_t total_copy_plus_logits_us      = 0;
    int64_t total_forwards                 = 0;
};

static bool zo_hexagon_compiled() {
#ifdef GGML_USE_HEXAGON
    return true;
#else
    return false;
#endif
}

static const char * zo_mode_name(zo_mode mode) {
    switch (mode) {
        case zo_mode::CPU:  return "cpu";
        case zo_mode::COOP: return "coop";
    }
    return "unknown";
}

static const char * lora_exec_mode_name(lora_exec_mode mode) {
    switch (mode) {
        case lora_exec_mode::RUNTIME:   return "runtime";
        case lora_exec_mode::FUSED_HTP: return "fused-htp";
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

static bool lora_backend_name_contains(const char * name, const char * needle) {
    return name != nullptr && std::strstr(name, needle) != nullptr;
}

static bool lora_tensor_on_hexagon(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
    const char * buffer_name = ggml_backend_buffer_name(tensor->buffer);
    const char * buft_name = buft != nullptr ? ggml_backend_buft_name(buft) : nullptr;

    return lora_backend_name_contains(buffer_name, "HTP") ||
           lora_backend_name_contains(buffer_name, "Hexagon") ||
           lora_backend_name_contains(buft_name, "HTP") ||
           lora_backend_name_contains(buft_name, "Hexagon");
}

static bool lora_antithetic_uses_rm(const lora_b_tensor & tensor) {
    return tensor.b_plus_rm_tensor != nullptr &&
           tensor.b_minus_rm_tensor != nullptr &&
           lora_tensor_on_hexagon(tensor.b_plus_rm_tensor) &&
           lora_tensor_on_hexagon(tensor.b_minus_rm_tensor);
}

static bool lora_antithetic_uses_plain(const lora_b_tensor & tensor) {
    return tensor.b_plus_tensor != nullptr &&
           tensor.b_minus_tensor != nullptr &&
           !lora_tensor_on_hexagon(tensor.b_plus_tensor) &&
           !lora_tensor_on_hexagon(tensor.b_minus_tensor);
}

static const char * lora_tensor_buffer_name(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return "none";
    }
    const char * name = ggml_backend_buffer_name(tensor->buffer);
    return name != nullptr ? name : "unknown";
}

static void print_usage(int /*argc*/, char ** argv) {
    LOG("\n");
    LOG("usage:\n");
    LOG("  %s [common llama.cpp args] [zoo args]\n", argv[0]);
    LOG("\n");
    LOG("zoo args:\n");
    LOG("  --mode <cpu|coop>         execution mode (default: cpu)\n");
    LOG("  --lora-exec <runtime|fused-htp>\n");
    LOG("                            LoRA graph path (default: runtime in cpu, fused-htp in coop)\n");
    LOG("  --pipeline <true|false>   overlap CPU LoRA-B prep with NPU forward in coop mode (default: false)\n");
    LOG("  --antithetic <true|false> pair +/- LoRA perturbations in one fused HTP forward (default: false)\n");
    LOG("  --steps <N>               number of zero-order steps (default: 5)\n");
    LOG("  --batch-size <N>          number of sequences per step (default: 4)\n");
    LOG("  --seq-len <N>             tokens per sequence (default: 32)\n");
    LOG("  --epsilon <F>             perturbation scale (default: 1e-2)\n");
    LOG("  --lr <F>                  learning rate (default: 5e-5)\n");
    LOG("\n");
    LOG("defaults:\n");
    LOG("  -m      %s\n", DEFAULT_MODEL_PATH);
    LOG("  --lora  %s\n", DEFAULT_LORA_PATH);
    LOG("\n");
    LOG("examples:\n");
    LOG("  %s --mode cpu --device none --lora %s -m %s\n", argv[0], DEFAULT_LORA_PATH, DEFAULT_MODEL_PATH);
    LOG("  %s --mode coop --lora-exec fused-htp --device HTP0 --lora %s -m %s -ngl 99\n",
            argv[0], DEFAULT_LORA_PATH, DEFAULT_MODEL_PATH);
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

static bool parse_float_arg(const std::string & value, float & dst, const char * name) {
    try {
        dst = std::stof(value);
        return true;
    } catch (const std::exception &) {
        LOG_ERR("%s: invalid float for %s: %s\n", __func__, name, value.c_str());
        return false;
    }
}

static bool parse_bool_arg(const std::string & value, bool & dst, const char * name) {
    if (value == "true" || value == "1" || value == "on" || value == "yes") {
        dst = true;
        return true;
    }
    if (value == "false" || value == "0" || value == "off" || value == "no") {
        dst = false;
        return true;
    }

    LOG_ERR("%s: invalid boolean for %s: %s\n", __func__, name, value.c_str());
    return false;
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

static bool prescan_mode(int argc, char ** argv, zo_options & opts) {
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
            opts.mode = zo_mode::CPU;
        } else if (value == "coop") {
            opts.mode = zo_mode::COOP;
        } else {
            LOG_ERR("%s: invalid mode: %s\n", __func__, value.c_str());
            return false;
        }
    }

    return true;
}

static bool parse_custom_args(int argc, char ** argv, zo_options & opts, std::vector<char *> & filtered_argv) {
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

            if (split_arg(arg, "--lora-exec", value)) {
                if (value.empty()) {
                    value = take_next("--lora-exec");
                }

                if (value == "runtime") {
                    opts.lora_exec = lora_exec_mode::RUNTIME;
                } else if (value == "fused-htp") {
                    opts.lora_exec = lora_exec_mode::FUSED_HTP;
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

            if (split_arg(arg, "--seq-len", value)) {
                if (value.empty()) {
                    value = take_next("--seq-len");
                }
                if (!parse_int_arg(value, opts.seq_len, "--seq-len")) {
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

static void validate_options(const zo_options & opts) {
    if (opts.steps <= 0) {
        throw std::runtime_error("steps must be > 0");
    }
    if (opts.batch_size <= 0) {
        throw std::runtime_error("batch-size must be > 0");
    }
    if (opts.seq_len < 2) {
        throw std::runtime_error("seq-len must be >= 2 for causal loss");
    }
    if (!(opts.epsilon > 0.0f)) {
        throw std::runtime_error("epsilon must be > 0");
    }
    if (!(opts.lr > 0.0f)) {
        throw std::runtime_error("lr must be > 0");
    }
    if (opts.pipeline && opts.mode != zo_mode::COOP) {
        throw std::runtime_error("--pipeline true requires --mode coop");
    }
}

static std::string tensor_shape_str(const ggml_tensor * tensor) {
    return std::to_string((int64_t) tensor->ne[0]) + "x" +
           std::to_string((int64_t) tensor->ne[1]) + "x" +
           std::to_string((int64_t) tensor->ne[2]) + "x" +
           std::to_string((int64_t) tensor->ne[3]);
}

static void deserialize_tensor_to_float(const std::vector<uint8_t> & src, ggml_type type, std::vector<float> & dst) {
    if (type == GGML_TYPE_F32) {
        const float * src_f32 = reinterpret_cast<const float *>(src.data());
        std::copy(src_f32, src_f32 + dst.size(), dst.begin());
        return;
    }

    if (type == GGML_TYPE_F16) {
        const ggml_fp16_t * src_f16 = reinterpret_cast<const ggml_fp16_t *>(src.data());
        for (size_t i = 0; i < dst.size(); ++i) {
            dst[i] = ggml_fp16_to_fp32(src_f16[i]);
        }
        return;
    }

    throw std::runtime_error("unsupported LoRA B tensor type for CPU arithmetic");
}

static void serialize_float_to_tensor(const std::vector<float> & src, ggml_type type, std::vector<uint8_t> & dst) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst.data(), src.data(), src.size() * sizeof(float));
        return;
    }

    if (type == GGML_TYPE_F16) {
        ggml_fp16_t * dst_f16 = reinterpret_cast<ggml_fp16_t *>(dst.data());
        for (size_t i = 0; i < src.size(); ++i) {
            dst_f16[i] = ggml_fp32_to_fp16(src[i]);
        }
        return;
    }

    throw std::runtime_error("unsupported LoRA B tensor type for backend upload");
}

static std::vector<lora_b_tensor> collect_lora_b_tensors(llama_adapter_lora * adapter) {
    std::vector<lora_b_tensor> tensors;
    tensors.reserve(adapter->ab_map.size());

    for (auto & it : adapter->ab_map) {
        ggml_tensor * b = it.second.b;
        if (b == nullptr) {
            continue;
        }

        lora_b_tensor entry;
        entry.name           = b->name;
        entry.tensor         = b;
        entry.b_plus_tensor  = it.second.b_plus;
        entry.b_minus_tensor = it.second.b_minus;
        entry.b_plus_rm_tensor  = it.second.b_plus_rm;
        entry.b_minus_rm_tensor = it.second.b_minus_rm;
        entry.type           = b->type;
        entry.n_elms         = ggml_nelements(b);
        entry.nbytes         = ggml_nbytes(b);
        entry.rm_nbytes      = (size_t) entry.n_elms * sizeof(float);

        if (entry.type != GGML_TYPE_F16 && entry.type != GGML_TYPE_F32) {
            throw std::runtime_error("unsupported LoRA B tensor type for " + entry.name);
        }

        entry.master.resize((size_t) entry.n_elms);
        entry.work.resize((size_t) entry.n_elms);
        entry.noise_a.resize((size_t) entry.n_elms);
        entry.noise_b.resize((size_t) entry.n_elms);
        entry.plus.resize((size_t) entry.n_elms);
        entry.minus.resize((size_t) entry.n_elms);
        entry.plus_rm.resize((size_t) entry.n_elms);
        entry.minus_rm.resize((size_t) entry.n_elms);
        entry.io_buf.resize(entry.nbytes);

        ggml_backend_tensor_get(b, entry.io_buf.data(), 0, entry.nbytes);
        deserialize_tensor_to_float(entry.io_buf, entry.type, entry.master);
        entry.work = entry.master;
        entry.plus = entry.master;
        entry.minus = entry.master;
        if (entry.b_plus_tensor != nullptr || entry.b_minus_tensor != nullptr) {
            if (entry.b_plus_tensor == nullptr || entry.b_minus_tensor == nullptr ||
                entry.b_plus_tensor->type != entry.type ||
                entry.b_minus_tensor->type != entry.type ||
                ggml_nelements(entry.b_plus_tensor) != entry.n_elms ||
                ggml_nelements(entry.b_minus_tensor) != entry.n_elms ||
                ggml_nbytes(entry.b_plus_tensor) != entry.nbytes ||
                ggml_nbytes(entry.b_minus_tensor) != entry.nbytes) {
                throw std::runtime_error("LoRA antithetic B tensor shape/type mismatch for " + entry.name);
            }
            std::fill(entry.io_buf.begin(), entry.io_buf.end(), 0);
            ggml_backend_tensor_set(entry.b_plus_tensor, entry.io_buf.data(), 0, entry.nbytes);
            ggml_backend_tensor_set(entry.b_minus_tensor, entry.io_buf.data(), 0, entry.nbytes);
        }
        if (entry.b_plus_rm_tensor != nullptr || entry.b_minus_rm_tensor != nullptr) {
            if (entry.b_plus_rm_tensor == nullptr || entry.b_minus_rm_tensor == nullptr ||
                entry.b_plus_rm_tensor->type != GGML_TYPE_F32 ||
                entry.b_minus_rm_tensor->type != GGML_TYPE_F32 ||
                entry.b_plus_rm_tensor->ne[0] != b->ne[1] ||
                entry.b_plus_rm_tensor->ne[1] != b->ne[0] ||
                entry.b_minus_rm_tensor->ne[0] != entry.b_plus_rm_tensor->ne[0] ||
                entry.b_minus_rm_tensor->ne[1] != entry.b_plus_rm_tensor->ne[1] ||
                ggml_nbytes(entry.b_plus_rm_tensor) != entry.rm_nbytes ||
                ggml_nbytes(entry.b_minus_rm_tensor) != entry.rm_nbytes) {
                throw std::runtime_error("LoRA antithetic B-RM tensor shape/type mismatch for " + entry.name);
            }
            std::fill(entry.plus_rm.begin(), entry.plus_rm.end(), 0.0f);
            ggml_backend_tensor_set(entry.b_plus_rm_tensor, entry.plus_rm.data(), 0, entry.rm_nbytes);
            ggml_backend_tensor_set(entry.b_minus_rm_tensor, entry.plus_rm.data(), 0, entry.rm_nbytes);
        }

        tensors.emplace_back(std::move(entry));
    }

    if (tensors.empty()) {
        throw std::runtime_error("no LoRA B tensors found in adapter");
    }

    return tensors;
}

static uint32_t derive_step_seed(uint32_t base, int step, uint32_t stream) {
    uint64_t x = (uint64_t) base;
    x ^= (uint64_t) stream + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    x ^= (uint64_t) (uint32_t) step * 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (uint32_t) x;
}

static std::vector<float> & noise_buffer(lora_b_tensor & tensor, lora_noise_slot slot) {
    return slot == lora_noise_slot::A ? tensor.noise_a : tensor.noise_b;
}

static const std::vector<float> & noise_buffer(const lora_b_tensor & tensor, lora_noise_slot slot) {
    return slot == lora_noise_slot::A ? tensor.noise_a : tensor.noise_b;
}

static std::vector<float> & work_buffer(lora_b_tensor & tensor, lora_work_slot slot) {
    return slot == lora_work_slot::PLUS ? tensor.plus : tensor.minus;
}

static const std::vector<float> & push_source_buffer(const lora_b_tensor & tensor, lora_push_source source) {
    switch (source) {
        case lora_push_source::WORK:   return tensor.work;
        case lora_push_source::MASTER: return tensor.master;
        case lora_push_source::PLUS:   return tensor.plus;
        case lora_push_source::MINUS:  return tensor.minus;
    }
    return tensor.work;
}

static void set_antithetic_rm_from_master(lora_b_tensor & tensor) {
    const bool fill_rm = lora_antithetic_uses_rm(tensor);
    const bool fill_plain = lora_antithetic_uses_plain(tensor);
    if (!fill_rm && !fill_plain) {
        throw std::runtime_error("antithetic LoRA has no usable B+/B- tensors for " + tensor.name);
    }

    const int64_t rank = tensor.tensor->ne[0];
    const int64_t n    = tensor.tensor->ne[1];
    for (int64_t in = 0; in < n; ++in) {
        for (int64_t ir = 0; ir < rank; ++ir) {
            const size_t plain_idx = (size_t) in * (size_t) rank + (size_t) ir;
            const size_t rm_idx    = (size_t) ir * (size_t) n + (size_t) in;
            if (fill_plain) {
                tensor.plus[plain_idx]  = tensor.master[plain_idx];
                tensor.minus[plain_idx] = tensor.master[plain_idx];
            }
            if (fill_rm) {
                tensor.plus_rm[rm_idx]  = tensor.master[plain_idx];
                tensor.minus_rm[rm_idx] = tensor.master[plain_idx];
            }
        }
    }
}

static void sample_noise_to_slot(std::vector<lora_b_tensor> & tensors, lora_noise_slot slot, std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto & tensor : tensors) {
        auto & noise = noise_buffer(tensor, slot);
        for (float & x : noise) {
            x = dist(rng);
        }
    }
}

static int64_t sample_noise_for_step(std::vector<lora_b_tensor> & tensors, lora_noise_slot slot, int step) {
    const int64_t t_start = ggml_time_us();
    std::mt19937 rng(derive_step_seed(DEFAULT_SEED, step, 0x4e4f4953U));
    sample_noise_to_slot(tensors, slot, rng);
    return ggml_time_us() - t_start;
}

static uint32_t derive_tensor_noise_seed(int step, size_t tensor_index) {
    return derive_step_seed(DEFAULT_SEED, step, 0x5a4f544eU ^ (uint32_t) tensor_index);
}

static int64_t sample_noise_for_step_tensorwise(std::vector<lora_b_tensor> & tensors, lora_noise_slot slot, int step) {
    const int64_t t_start = ggml_time_us();
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t it = 0; it < tensors.size(); ++it) {
        std::mt19937 rng(derive_tensor_noise_seed(step, it));
        auto & noise = noise_buffer(tensors[it], slot);
        for (float & x : noise) {
            x = dist(rng);
        }
    }

    return ggml_time_us() - t_start;
}

static int64_t make_work_from_master_regen_noise(std::vector<lora_b_tensor> & tensors, int step, float epsilon_scale) {
    const int64_t t_start = ggml_time_us();
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t it = 0; it < tensors.size(); ++it) {
        auto & tensor = tensors[it];
        std::mt19937 rng(derive_tensor_noise_seed(step, it));
        for (size_t i = 0; i < tensor.master.size(); ++i) {
            const float noise = dist(rng);
            tensor.work[i] = tensor.master[i] + epsilon_scale * noise;
        }
    }

    return ggml_time_us() - t_start;
}

static int64_t make_antithetic_slots_from_master_regen_noise(std::vector<lora_b_tensor> & tensors, int step, float epsilon) {
    const int64_t t_start = ggml_time_us();
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t it = 0; it < tensors.size(); ++it) {
        auto & tensor = tensors[it];
        const bool fill_rm = lora_antithetic_uses_rm(tensor);
        const bool fill_plain = lora_antithetic_uses_plain(tensor);
        if (!fill_rm && !fill_plain) {
            throw std::runtime_error("antithetic LoRA has no usable B+/B- tensors for " + tensor.name);
        }

        const int64_t rank = tensor.tensor->ne[0];
        const int64_t n    = tensor.tensor->ne[1];

        std::mt19937 rng_plus(derive_tensor_noise_seed(step, it));
        for (size_t i = 0; i < tensor.master.size(); ++i) {
            const float noise = dist(rng_plus);
            const int64_t in = (int64_t) i / rank;
            const int64_t ir = (int64_t) i - in * rank;
            const size_t rm_idx = (size_t) ir * (size_t) n + (size_t) in;
            const float value = tensor.master[i] + epsilon * noise;
            if (fill_plain) {
                tensor.plus[i] = value;
            }
            if (fill_rm) {
                tensor.plus_rm[rm_idx] = value;
            }
        }

        // Regenerate the same noise stream instead of caching it. This keeps
        // the non-pipeline antithetic path as a strict no-noise-cache baseline.
        std::mt19937 rng_minus(derive_tensor_noise_seed(step, it));
        for (size_t i = 0; i < tensor.master.size(); ++i) {
            const float noise = dist(rng_minus);
            const int64_t in = (int64_t) i / rank;
            const int64_t ir = (int64_t) i - in * rank;
            const size_t rm_idx = (size_t) ir * (size_t) n + (size_t) in;
            const float value = tensor.master[i] - epsilon * noise;
            if (fill_plain) {
                tensor.minus[i] = value;
            }
            if (fill_rm) {
                tensor.minus_rm[rm_idx] = value;
            }
        }
    }

    return ggml_time_us() - t_start;
}

static int64_t make_antithetic_slots_from_master_slot(
        std::vector<lora_b_tensor> & tensors,
        lora_noise_slot              noise_slot_id,
        float                        epsilon) {
    const int64_t t_start = ggml_time_us();

    for (auto & tensor : tensors) {
        const bool fill_rm = lora_antithetic_uses_rm(tensor);
        const bool fill_plain = lora_antithetic_uses_plain(tensor);
        if (!fill_rm && !fill_plain) {
            throw std::runtime_error("antithetic LoRA has no usable B+/B- tensors for " + tensor.name);
        }

        const int64_t rank = tensor.tensor->ne[0];
        const int64_t n    = tensor.tensor->ne[1];
        const auto & noise = noise_buffer(tensor, noise_slot_id);
        for (size_t i = 0; i < tensor.master.size(); ++i) {
            const int64_t in = (int64_t) i / rank;
            const int64_t ir = (int64_t) i - in * rank;
            const size_t rm_idx = (size_t) ir * (size_t) n + (size_t) in;
            const float plus_value = tensor.master[i] + epsilon * noise[i];
            const float minus_value = tensor.master[i] - epsilon * noise[i];
            if (fill_plain) {
                tensor.plus[i] = plus_value;
                tensor.minus[i] = minus_value;
            }
            if (fill_rm) {
                tensor.plus_rm[rm_idx]  = plus_value;
                tensor.minus_rm[rm_idx] = minus_value;
            }
        }
    }

    return ggml_time_us() - t_start;
}

static int64_t make_slot_from_master(
        std::vector<lora_b_tensor> & tensors,
        lora_work_slot               dst_slot,
        lora_noise_slot              noise_slot_id,
        float                        epsilon_scale) {
    const int64_t t_start = ggml_time_us();
    for (auto & tensor : tensors) {
        auto & dst = work_buffer(tensor, dst_slot);
        const auto & noise = noise_buffer(tensor, noise_slot_id);
        for (size_t i = 0; i < tensor.master.size(); ++i) {
            dst[i] = tensor.master[i] + epsilon_scale * noise[i];
        }
    }
    return ggml_time_us() - t_start;
}

static void apply_update_to_master_from_slot(std::vector<lora_b_tensor> & tensors, lora_noise_slot slot, float scale) {
    for (auto & tensor : tensors) {
        const auto & noise = noise_buffer(tensor, slot);
        for (size_t i = 0; i < tensor.master.size(); ++i) {
            tensor.master[i] += scale * noise[i];
        }
    }
}

static int64_t apply_update_to_master_regen_noise(std::vector<lora_b_tensor> & tensors, int step, float scale) {
    const int64_t t_start = ggml_time_us();
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t it = 0; it < tensors.size(); ++it) {
        auto & tensor = tensors[it];
        std::mt19937 rng(derive_tensor_noise_seed(step, it));
        for (size_t i = 0; i < tensor.master.size(); ++i) {
            const float noise = dist(rng);
            tensor.master[i] += scale * noise;
        }
        tensor.work = tensor.master;
    }

    return ggml_time_us() - t_start;
}

static bool lora_debug_enabled() {
    const char * env = std::getenv("GGML_HEXAGON_LORA_DEBUG");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static void write_floats_to_tensor(
        ggml_tensor * tensor,
        const std::vector<float> & src,
        std::vector<uint8_t> & io_buf) {
    if (tensor == nullptr) {
        throw std::runtime_error("attempted to write LoRA data to a null tensor");
    }

    const size_t nbytes = ggml_nbytes(tensor);
    if (io_buf.size() < nbytes) {
        io_buf.resize(nbytes);
    }
    serialize_float_to_tensor(src, tensor->type, io_buf);
    ggml_backend_tensor_set(tensor, io_buf.data(), 0, nbytes);
}

static int64_t push_b_tensors_from(
        std::vector<lora_b_tensor> & tensors,
        lora_push_source             source,
        const char *                 label) {
    const int64_t t_start = ggml_time_us();
    const bool debug = lora_debug_enabled();

    double sum_abs_src = 0.0;
    double sum_abs_readback = 0.0;
    double max_abs_readback_diff = 0.0;
    size_t total_bytes = 0;
    size_t readback_tensors = 0;
    for (auto & tensor : tensors) {
        const auto & src = push_source_buffer(tensor, source);
        if (debug) {
            for (const float v : src) {
                sum_abs_src += std::fabs((double) v);
            }
            total_bytes += tensor.nbytes;
        }
        serialize_float_to_tensor(src, tensor.type, tensor.io_buf);
        ggml_backend_tensor_set(tensor.tensor, tensor.io_buf.data(), 0, tensor.nbytes);

        if (debug) {
            std::vector<uint8_t> readback_buf(tensor.nbytes);
            std::vector<float> readback(src.size());
            ggml_backend_tensor_get(tensor.tensor, readback_buf.data(), 0, tensor.nbytes);
            deserialize_tensor_to_float(readback_buf, tensor.type, readback);
            for (size_t i = 0; i < readback.size(); ++i) {
                const double rv = readback[i];
                sum_abs_readback += std::fabs(rv);
                max_abs_readback_diff = std::max(max_abs_readback_diff, std::fabs(rv - (double) src[i]));
            }
            readback_tensors++;
        }
    }

    const int64_t t_us = ggml_time_us() - t_start;
    if (debug) {
        LOG("lora_b push debug: source=%s tensors=%zu bytes=%.3f MiB sum_abs_src=%.8e readback_tensors=%zu sum_abs_readback=%.8e max_abs_readback_diff=%.8e time=%.3f ms\n",
                label, tensors.size(), (double) total_bytes / 1024.0 / 1024.0, sum_abs_src,
                readback_tensors, sum_abs_readback, max_abs_readback_diff, t_us / 1000.0);
    }

    return t_us;
}

static int64_t push_b_tensors(std::vector<lora_b_tensor> & tensors) {
    return push_b_tensors_from(tensors, lora_push_source::WORK, "work");
}

static int64_t push_antithetic_b_tensors(std::vector<lora_b_tensor> & tensors, const char * label) {
    const int64_t t_start = ggml_time_us();
    const bool debug = lora_debug_enabled();

    double sum_abs_plus = 0.0;
    double sum_abs_minus = 0.0;
    size_t total_bytes = 0;
    for (auto & tensor : tensors) {
        const bool push_rm = lora_antithetic_uses_rm(tensor);
        const bool push_plain = lora_antithetic_uses_plain(tensor);
        if (!push_rm && !push_plain) {
            throw std::runtime_error("antithetic LoRA has no usable B+/B- tensors for " + tensor.name);
        }

        if (push_rm) {
            ggml_backend_tensor_set(tensor.b_plus_rm_tensor, tensor.plus_rm.data(), 0, tensor.rm_nbytes);
            ggml_backend_tensor_set(tensor.b_minus_rm_tensor, tensor.minus_rm.data(), 0, tensor.rm_nbytes);
            if (debug) {
                for (const float v : tensor.plus_rm) {
                    sum_abs_plus += std::fabs((double) v);
                }
                for (const float v : tensor.minus_rm) {
                    sum_abs_minus += std::fabs((double) v);
                }
                total_bytes += 2 * tensor.rm_nbytes;
            }
        }

        if (push_plain) {
            write_floats_to_tensor(tensor.b_plus_tensor, tensor.plus, tensor.io_buf);
            write_floats_to_tensor(tensor.b_minus_tensor, tensor.minus, tensor.io_buf);
            if (debug) {
                for (const float v : tensor.plus) {
                    sum_abs_plus += std::fabs((double) v);
                }
                for (const float v : tensor.minus) {
                    sum_abs_minus += std::fabs((double) v);
                }
                total_bytes += 2 * tensor.nbytes;
            }
        }
    }

    const int64_t t_us = ggml_time_us() - t_start;
    if (debug) {
        LOG("lora_antithetic_b_rm push debug: source=%s tensors=%zu bytes=%.3f MiB sum_abs_plus=%.8e sum_abs_minus=%.8e time=%.3f ms\n",
                label, tensors.size(), (double) total_bytes / 1024.0 / 1024.0,
                sum_abs_plus, sum_abs_minus, t_us / 1000.0);
    }

    return t_us;
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
    const int64_t t_end = ggml_time_us();

    return t_end - t_start;
}

static float compute_loss_from_current_logits(
        llama_context * ctx,
        int32_t n_vocab,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len) {
    double loss_sum = 0.0;
    int64_t loss_count = 0;

    for (int s = 0; s < batch_size; ++s) {
        const int t = seq_len - 2;
        const int batch_token_index = s * seq_len + t;
        const llama_token label = tokens[(size_t) s * seq_len + (t + 1)];
        const float * logits = llama_get_logits_ith(ctx, batch_token_index);
        if (logits == nullptr) {
            throw std::runtime_error("failed to fetch logits row");
        }

        loss_sum += logits_cross_entropy(logits, n_vocab, label);
        ++loss_count;
    }

    return (float) (loss_sum / (double) loss_count);
}

static std::vector<llama_token> make_paired_tokens(const std::vector<llama_token> & tokens, int batch_size, int seq_len) {
    std::vector<llama_token> paired((size_t) 2 * (size_t) batch_size * (size_t) seq_len);
    const size_t n = (size_t) batch_size * (size_t) seq_len;
    std::copy(tokens.begin(), tokens.end(), paired.begin());
    std::copy(tokens.begin(), tokens.end(), paired.begin() + n);
    return paired;
}

static paired_forward_result compute_paired_loss_from_current_logits(
        llama_context * ctx,
        int32_t n_vocab,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len) {
    paired_forward_result res;
    double loss_plus_sum  = 0.0;
    double loss_minus_sum = 0.0;
    int64_t loss_count = 0;

    for (int s = 0; s < batch_size; ++s) {
        const int t = seq_len - 2;
        const int plus_batch_token_index  = s * seq_len + t;
        const int minus_batch_token_index = (batch_size + s) * seq_len + t;
        const llama_token label = tokens[(size_t) s * seq_len + (t + 1)];

        const float * logits_plus = llama_get_logits_ith(ctx, plus_batch_token_index);
        if (logits_plus == nullptr) {
            throw std::runtime_error("failed to fetch paired plus logits row");
        }
        const float * logits_minus = llama_get_logits_ith(ctx, minus_batch_token_index);
        if (logits_minus == nullptr) {
            throw std::runtime_error("failed to fetch paired minus logits row");
        }

        loss_plus_sum  += logits_cross_entropy(logits_plus,  n_vocab, label);
        loss_minus_sum += logits_cross_entropy(logits_minus, n_vocab, label);
        ++loss_count;
    }

    res.loss_plus  = (float) (loss_plus_sum  / (double) loss_count);
    res.loss_minus = (float) (loss_minus_sum / (double) loss_count);
    return res;
}

static logits_snapshot copy_loss_logits_from_current(
        llama_context * ctx,
        int32_t n_vocab,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len) {
    logits_snapshot snap;
    snap.n_vocab = n_vocab;
    snap.n_rows  = batch_size;
    snap.logits.resize((size_t) snap.n_rows * (size_t) n_vocab);
    snap.labels.resize((size_t) snap.n_rows);

    int row = 0;
    for (int s = 0; s < batch_size; ++s) {
        const int t = seq_len - 2;
        const int batch_token_index = s * seq_len + t;
        const llama_token label = tokens[(size_t) s * seq_len + (t + 1)];
        const float * logits = llama_get_logits_ith(ctx, batch_token_index);
        if (logits == nullptr) {
            throw std::runtime_error("failed to fetch logits row for snapshot");
        }

        std::memcpy(snap.logits.data() + (size_t) row * (size_t) n_vocab, logits, (size_t) n_vocab * sizeof(float));
        snap.labels[(size_t) row] = label;
        ++row;
    }

    return snap;
}

static float compute_loss_from_snapshot(const logits_snapshot & snap) {
    double loss_sum = 0.0;
    for (int32_t row = 0; row < snap.n_rows; ++row) {
        const float * logits = snap.logits.data() + (size_t) row * (size_t) snap.n_vocab;
        loss_sum += logits_cross_entropy(logits, snap.n_vocab, snap.labels[(size_t) row]);
    }

    return (float) (loss_sum / (double) snap.n_rows);
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

static paired_forward_result run_paired_prefill_forward(
        llama_context * ctx,
        llama_batch & batch,
        llama_memory_t mem,
        int32_t n_batch,
        int32_t n_vocab,
        const std::vector<llama_token> & tokens,
        int batch_size,
        int seq_len) {
    const auto paired_tokens = make_paired_tokens(tokens, batch_size, seq_len);

    paired_forward_result res;
    res.t_pp_us = run_prefill_decode(ctx, batch, mem, n_batch, paired_tokens, 2 * batch_size, seq_len);

    paired_forward_result losses = compute_paired_loss_from_current_logits(ctx, n_vocab, tokens, batch_size, seq_len);
    res.loss_plus  = losses.loss_plus;
    res.loss_minus = losses.loss_minus;
    return res;
}

static void log_lora_summary(const std::vector<lora_b_tensor> & tensors) {
    int64_t total_elms = 0;
    size_t  total_bytes = 0;

    LOG("LoRA B tensors:\n");
    for (const auto & tensor : tensors) {
        total_elms += tensor.n_elms;
        total_bytes += tensor.nbytes;
        LOG("  %s : type=%s shape=%s n_elms=%lld size=%.3f MiB\n",
                tensor.name.c_str(),
                ggml_type_name(tensor.type),
                tensor_shape_str(tensor.tensor).c_str(),
                (long long) tensor.n_elms,
                tensor.nbytes / 1024.0 / 1024.0);
    }

    LOG("total LoRA B parameters: %lld elements, %.3f MiB\n",
            (long long) total_elms,
            total_bytes / 1024.0 / 1024.0);
}

static void log_lora_placement_summary(
        const std::vector<lora_b_tensor> & tensors,
        bool fused_requested,
        bool antithetic) {
    size_t htp_targets = 0;
    size_t cpu_targets = 0;
    size_t htp_rm_targets = 0;
    size_t cpu_plain_targets = 0;

    for (const auto & tensor : tensors) {
        const bool on_htp = lora_tensor_on_hexagon(tensor.tensor);
        htp_targets += on_htp ? 1 : 0;
        cpu_targets += on_htp ? 0 : 1;
        if (antithetic) {
            htp_rm_targets += lora_antithetic_uses_rm(tensor) ? 1 : 0;
            cpu_plain_targets += lora_antithetic_uses_plain(tensor) ? 1 : 0;
        }
    }

    LOG("LoRA backend placement: total=%zu htp=%zu cpu=%zu",
            tensors.size(), htp_targets, cpu_targets);
    if (antithetic) {
        LOG(" htp_antithetic_rm=%zu cpu_antithetic_plain=%zu",
                htp_rm_targets, cpu_plain_targets);
    }
    LOG("\n");

    if (fused_requested && cpu_targets > 0) {
        LOG("LoRA placement note: %zu CPU-resident target(s) will use runtime LoRA fallback; only HTP-resident targets use fused LoRA. Partial offload is expected to be slower than -ngl all/99.\n",
                cpu_targets);
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx      = 128;
    params.n_batch    = 128;
    params.n_ubatch   = 128;
    params.n_parallel = 4;
    params.n_sequences = 4;

    zo_options opts;

    common_init();

    if (!prescan_mode(argc, argv, opts)) {
        return 1;
    }

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

        if (!opts.lora_exec_set) {
            opts.lora_exec = opts.mode == zo_mode::COOP ? lora_exec_mode::FUSED_HTP : lora_exec_mode::RUNTIME;
        }

        if (opts.mode == zo_mode::CPU && opts.lora_exec == lora_exec_mode::FUSED_HTP) {
            throw std::runtime_error("--lora-exec fused-htp requires --mode coop");
        }
        if (opts.antithetic && (opts.mode != zo_mode::COOP || opts.lora_exec != lora_exec_mode::FUSED_HTP)) {
            throw std::runtime_error("--antithetic true requires --mode coop --lora-exec fused-htp");
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
            set_env_var("GGML_HEXAGON_FUSED_LORA", opts.lora_exec == lora_exec_mode::FUSED_HTP ? "auto" : "0");
            set_env_var("GGML_HEXAGON_ANTITHETIC_LORA", opts.antithetic ? "1" : "0");
        }

        LOG("\n");
        LOG("zoo-zo-lora-fa configuration:\n");
        LOG("  mode       = %s\n", zo_mode_name(opts.mode));
        LOG("  lora_exec  = %s\n", lora_exec_mode_name(opts.lora_exec));
        LOG("  pipeline   = %s\n", opts.pipeline ? "true" : "false");
        LOG("  antithetic = %s\n", opts.antithetic ? "true" : "false");
        LOG("  model      = %s\n", params.model.path.c_str());
        LOG("  lora       = %s\n", params.lora_adapters[0].path.c_str());
        LOG("  batch_size = %d\n", opts.batch_size);
        LOG("  seq_len    = %d\n", opts.seq_len);
        LOG("  steps      = %d\n", opts.steps);
        LOG("  epsilon    = %.6g\n", opts.epsilon);
        LOG("  lr         = %.6g\n", opts.lr);
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

        auto & loras = init->lora();
        if (loras.empty() || loras.front() == nullptr) {
            throw std::runtime_error("no LoRA adapter was loaded");
        }

        llama_adapter_lora * adapter = loras.front().get();
        auto lora_b_tensors = collect_lora_b_tensors(adapter);
        log_lora_summary(lora_b_tensors);
        log_lora_placement_summary(
                lora_b_tensors,
                opts.mode == zo_mode::COOP && opts.lora_exec == lora_exec_mode::FUSED_HTP,
                opts.antithetic);
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
            if (opts.antithetic) {
                for (auto & tensor : lora_b_tensors) {
                    set_antithetic_rm_from_master(tensor);
                }
                (void) push_antithetic_b_tensors(lora_b_tensors, "warmup-master");
                (void) run_paired_prefill_forward(ctx, batch, mem, params.n_batch, n_vocab, warmup_tokens, opts.batch_size, opts.seq_len);
            } else {
                (void) run_prefill_forward(ctx, batch, mem, params.n_batch, n_vocab, warmup_tokens, opts.batch_size, opts.seq_len);
            }
        }

        perf_stats stats;

        if (!opts.pipeline) {
            if (!opts.antithetic) {
            for (int step = 0; step < opts.steps; ++step) {
                const auto tokens = make_dummy_tokens_for_step(n_vocab, opts.batch_size, opts.seq_len, step);

                const int64_t t_step_start = ggml_time_us();

                const int64_t t_prepare_plus_us = make_work_from_master_regen_noise(lora_b_tensors, step, +opts.epsilon);
                const int64_t t_push_plus_us = push_b_tensors(lora_b_tensors);
                const forward_result plus = run_prefill_forward(ctx, batch, mem, params.n_batch, n_vocab, tokens, opts.batch_size, opts.seq_len);

                const int64_t t_prepare_minus_us = make_work_from_master_regen_noise(lora_b_tensors, step, -opts.epsilon);
                const int64_t t_push_minus_us = push_b_tensors(lora_b_tensors);
                const forward_result minus = run_prefill_forward(ctx, batch, mem, params.n_batch, n_vocab, tokens, opts.batch_size, opts.seq_len);

                const float g = (plus.loss - minus.loss) / (2.0f * opts.epsilon);

                const int64_t t_prepare_update_us = apply_update_to_master_regen_noise(lora_b_tensors, step, -opts.lr * g);

                const int64_t t_step_us = ggml_time_us() - t_step_start;

                stats.total_pp_us          += plus.t_pp_us + minus.t_pp_us;
                stats.total_step_us        += t_step_us;
                stats.total_push_plus_us   += t_push_plus_us;
                stats.total_push_minus_us  += t_push_minus_us;
                stats.total_prepare_plus_us  += t_prepare_plus_us;
                stats.total_prepare_minus_us += t_prepare_minus_us;
                stats.total_prepare_update_us += t_prepare_update_us;
                stats.total_forwards       += 2;

                const double speed_pp_plus  = (double) total_tokens / ((double) plus.t_pp_us  / 1e6);
                const double speed_pp_minus = (double) total_tokens / ((double) minus.t_pp_us / 1e6);
                const float loss_delta = plus.loss - minus.loss;

                LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                    "t_pp_plus=%.3f ms speed_pp_plus=%.2f t/s | "
                    "t_pp_minus=%.3f ms speed_pp_minus=%.2f t/s | "
                    "step=%.3f ms\n",
                        step + 1, opts.steps,
                        plus.loss, minus.loss, (double) loss_delta, (double) g,
                        plus.t_pp_us / 1000.0, speed_pp_plus,
                        minus.t_pp_us / 1000.0, speed_pp_minus,
                        t_step_us / 1000.0);

                if (opts.mode == zo_mode::COOP) {
                    LOG("  serial prep: plus=%.3f ms minus=%.3f ms update=%.3f ms (regen-noise, no noise cache)\n",
                            t_prepare_plus_us / 1000.0,
                            t_prepare_minus_us / 1000.0,
                            t_prepare_update_us / 1000.0);
                    LOG("  approx push/reload before forward: plus=%.3f ms minus=%.3f ms\n",
                            t_push_plus_us / 1000.0,
                            t_push_minus_us / 1000.0);
                }
            }
            } else {
                for (int step = 0; step < opts.steps; ++step) {
                    const auto tokens = make_dummy_tokens_for_step(n_vocab, opts.batch_size, opts.seq_len, step);

                    const int64_t t_step_start = ggml_time_us();

                    const int64_t t_prepare_b_pair_us = make_antithetic_slots_from_master_regen_noise(lora_b_tensors, step, opts.epsilon);
                    const int64_t t_push_b_pair_us = push_antithetic_b_tensors(lora_b_tensors, "regen-b-pair");
                    const paired_forward_result paired = run_paired_prefill_forward(
                            ctx, batch, mem, params.n_batch, n_vocab, tokens, opts.batch_size, opts.seq_len);

                    const float g = (paired.loss_plus - paired.loss_minus) / (2.0f * opts.epsilon);

                    const int64_t t_prepare_update_us = apply_update_to_master_regen_noise(lora_b_tensors, step, -opts.lr * g);

                    const int64_t t_step_us = ggml_time_us() - t_step_start;

                    stats.total_pp_us              += paired.t_pp_us;
                    stats.total_step_us            += t_step_us;
                    stats.total_push_antithetic_b_us += t_push_b_pair_us;
                    stats.total_prepare_plus_us    += t_prepare_b_pair_us;
                    stats.total_prepare_update_us  += t_prepare_update_us;
                    stats.total_forwards           += 1;

                    const double speed_pp_paired = (double) (2 * total_tokens) / ((double) paired.t_pp_us / 1e6);
                    const float loss_delta = paired.loss_plus - paired.loss_minus;

                    LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                        "t_pp_paired=%.3f ms speed_pp_paired=%.2f t/s | step=%.3f ms\n",
                            step + 1, opts.steps,
                            paired.loss_plus, paired.loss_minus, (double) loss_delta, (double) g,
                            paired.t_pp_us / 1000.0, speed_pp_paired,
                            t_step_us / 1000.0);

                    LOG("  antithetic serial prep: b_pair=%.3f ms update=%.3f ms (regen-noise, no noise cache)\n",
                            t_prepare_b_pair_us / 1000.0,
                            t_prepare_update_us / 1000.0);
                    LOG("  approx push/reload before paired forward: b_pair=%.3f ms\n",
                            t_push_b_pair_us / 1000.0);
                }
            }
        } else {
            if (!opts.antithetic) {
            const int64_t t_initial_noise_us = sample_noise_for_step(lora_b_tensors, lora_noise_slot::A, 0);
            stats.total_prepare_noise_us += t_initial_noise_us;
            stats.total_step_us += t_initial_noise_us;
            LOG("pipeline initial noise prepare: %.3f ms\n", t_initial_noise_us / 1000.0);

            for (int step = 0; step < opts.steps; ++step) {
                const auto tokens = make_dummy_tokens_for_step(n_vocab, opts.batch_size, opts.seq_len, step);
                const lora_noise_slot cur_noise  = (step % 2 == 0) ? lora_noise_slot::A : lora_noise_slot::B;
                const lora_noise_slot next_noise = (step % 2 == 0) ? lora_noise_slot::B : lora_noise_slot::A;

                const int64_t t_step_start = ggml_time_us();

                const int64_t t_prepare_plus_us = make_slot_from_master(lora_b_tensors, lora_work_slot::PLUS, cur_noise, +opts.epsilon);
                const int64_t t_push_plus_us = push_b_tensors_from(lora_b_tensors, lora_push_source::PLUS, "plus");

                auto minus_future = std::async(std::launch::async, [&lora_b_tensors, cur_noise, epsilon = opts.epsilon]() {
                    return make_slot_from_master(lora_b_tensors, lora_work_slot::MINUS, cur_noise, -epsilon);
                });

                forward_result plus;
                plus.t_pp_us = run_prefill_decode(ctx, batch, mem, params.n_batch, tokens, opts.batch_size, opts.seq_len);

                // The next decode reuses llama.cpp's internal logits buffer.
                // Snapshot the supervised rows before launching the async CE.
                const int64_t t_copy_plus_logits_start = ggml_time_us();
                auto plus_logits = copy_loss_logits_from_current(ctx, n_vocab, tokens, opts.batch_size, opts.seq_len);
                const int64_t t_copy_plus_logits_us = ggml_time_us() - t_copy_plus_logits_start;
                auto plus_loss_future = std::async(std::launch::async, [snap = std::move(plus_logits)]() {
                    return compute_loss_from_snapshot(snap);
                });

                const int64_t t_wait_minus_start = ggml_time_us();
                const int64_t t_prepare_minus_us = minus_future.get();
                const int64_t t_wait_minus_ready_us = ggml_time_us() - t_wait_minus_start;

                const int64_t t_push_minus_us = push_b_tensors_from(lora_b_tensors, lora_push_source::MINUS, "minus");

                const bool has_next_step = step + 1 < opts.steps;
                std::future<int64_t> next_noise_future;
                if (has_next_step) {
                    next_noise_future = std::async(std::launch::async, [&lora_b_tensors, next_noise, next_step = step + 1]() {
                        return sample_noise_for_step(lora_b_tensors, next_noise, next_step);
                    });
                }

                const forward_result minus = run_prefill_forward(ctx, batch, mem, params.n_batch, n_vocab, tokens, opts.batch_size, opts.seq_len);

                const int64_t t_wait_plus_loss_start = ggml_time_us();
                plus.loss = plus_loss_future.get();
                const int64_t t_wait_plus_loss_ready_us = ggml_time_us() - t_wait_plus_loss_start;

                const float g = (plus.loss - minus.loss) / (2.0f * opts.epsilon);

                apply_update_to_master_from_slot(lora_b_tensors, cur_noise, -opts.lr * g);

                int64_t t_prepare_next_noise_us = 0;
                int64_t t_wait_next_noise_ready_us = 0;
                if (has_next_step) {
                    const int64_t t_wait_next_start = ggml_time_us();
                    t_prepare_next_noise_us = next_noise_future.get();
                    t_wait_next_noise_ready_us = ggml_time_us() - t_wait_next_start;
                    stats.total_prepare_noise_us += t_prepare_next_noise_us;
                }

                const int64_t t_step_us = ggml_time_us() - t_step_start;

                stats.total_pp_us                    += plus.t_pp_us + minus.t_pp_us;
                stats.total_step_us                  += t_step_us;
                stats.total_push_plus_us             += t_push_plus_us;
                stats.total_push_minus_us            += t_push_minus_us;
                stats.total_prepare_plus_us          += t_prepare_plus_us;
                stats.total_prepare_minus_us         += t_prepare_minus_us;
                stats.total_prepare_next_noise_us    += t_prepare_next_noise_us;
                stats.total_wait_minus_ready_us      += t_wait_minus_ready_us;
                stats.total_wait_next_noise_ready_us += t_wait_next_noise_ready_us;
                stats.total_wait_plus_loss_ready_us  += t_wait_plus_loss_ready_us;
                stats.total_copy_plus_logits_us      += t_copy_plus_logits_us;
                stats.total_forwards                 += 2;

                const double speed_pp_plus  = (double) total_tokens / ((double) plus.t_pp_us  / 1e6);
                const double speed_pp_minus = (double) total_tokens / ((double) minus.t_pp_us / 1e6);
                const float loss_delta = plus.loss - minus.loss;

                LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                    "t_pp_plus=%.3f ms speed_pp_plus=%.2f t/s | "
                    "t_pp_minus=%.3f ms speed_pp_minus=%.2f t/s | "
                    "step=%.3f ms\n",
                        step + 1, opts.steps,
                        plus.loss, minus.loss, (double) loss_delta, (double) g,
                        plus.t_pp_us / 1000.0, speed_pp_plus,
                        minus.t_pp_us / 1000.0, speed_pp_minus,
                        t_step_us / 1000.0);

                LOG("  pipeline prep: plus=%.3f ms minus=%.3f ms next_noise=%.3f ms | copy_plus_logits=%.3f ms wait_minus=%.3f ms wait_next_noise=%.3f ms wait_plus_loss=%.3f ms\n",
                        t_prepare_plus_us / 1000.0,
                        t_prepare_minus_us / 1000.0,
                        t_prepare_next_noise_us / 1000.0,
                        t_copy_plus_logits_us / 1000.0,
                        t_wait_minus_ready_us / 1000.0,
                        t_wait_next_noise_ready_us / 1000.0,
                        t_wait_plus_loss_ready_us / 1000.0);

                LOG("  approx push/reload before forward: plus=%.3f ms minus=%.3f ms\n",
                        t_push_plus_us / 1000.0,
                        t_push_minus_us / 1000.0);
            }
            } else {
                const int64_t t_initial_noise_us = sample_noise_for_step_tensorwise(lora_b_tensors, lora_noise_slot::A, 0);
                stats.total_prepare_noise_us += t_initial_noise_us;
                stats.total_step_us += t_initial_noise_us;
                LOG("antithetic pipeline initial noise prepare: %.3f ms\n", t_initial_noise_us / 1000.0);

                for (int step = 0; step < opts.steps; ++step) {
                    const auto tokens = make_dummy_tokens_for_step(n_vocab, opts.batch_size, opts.seq_len, step);
                    const lora_noise_slot cur_noise  = (step % 2 == 0) ? lora_noise_slot::A : lora_noise_slot::B;
                    const lora_noise_slot next_noise = (step % 2 == 0) ? lora_noise_slot::B : lora_noise_slot::A;

                    const int64_t t_step_start = ggml_time_us();

                    const int64_t t_prepare_b_pair_us =
                        make_antithetic_slots_from_master_slot(lora_b_tensors, cur_noise, opts.epsilon);
                    const int64_t t_push_b_pair_us = push_antithetic_b_tensors(lora_b_tensors, "slot-b-pair");

                    const bool has_next_step = step + 1 < opts.steps;
                    std::future<int64_t> next_noise_future;
                    if (has_next_step) {
                        next_noise_future = std::async(std::launch::async, [&lora_b_tensors, next_noise, next_step = step + 1]() {
                            return sample_noise_for_step_tensorwise(lora_b_tensors, next_noise, next_step);
                        });
                    }

                    const paired_forward_result paired = run_paired_prefill_forward(
                            ctx, batch, mem, params.n_batch, n_vocab, tokens, opts.batch_size, opts.seq_len);

                    const float g = (paired.loss_plus - paired.loss_minus) / (2.0f * opts.epsilon);

                    apply_update_to_master_from_slot(lora_b_tensors, cur_noise, -opts.lr * g);

                    int64_t t_prepare_next_noise_us = 0;
                    int64_t t_wait_next_noise_ready_us = 0;
                    if (has_next_step) {
                        const int64_t t_wait_next_start = ggml_time_us();
                        t_prepare_next_noise_us = next_noise_future.get();
                        t_wait_next_noise_ready_us = ggml_time_us() - t_wait_next_start;
                        stats.total_prepare_noise_us += t_prepare_next_noise_us;
                    }

                    const int64_t t_step_us = ggml_time_us() - t_step_start;

                    stats.total_pp_us                    += paired.t_pp_us;
                    stats.total_step_us                  += t_step_us;
                    stats.total_push_antithetic_b_us     += t_push_b_pair_us;
                    stats.total_prepare_plus_us          += t_prepare_b_pair_us;
                    stats.total_prepare_next_noise_us    += t_prepare_next_noise_us;
                    stats.total_wait_next_noise_ready_us += t_wait_next_noise_ready_us;
                    stats.total_forwards                 += 1;

                    const double speed_pp_paired = (double) (2 * total_tokens) / ((double) paired.t_pp_us / 1e6);
                    const float loss_delta = paired.loss_plus - paired.loss_minus;

                    LOG("step %d/%d: loss_plus=%.6f loss_minus=%.6f loss_delta=%.8e g=%.8e | "
                        "t_pp_paired=%.3f ms speed_pp_paired=%.2f t/s | step=%.3f ms\n",
                            step + 1, opts.steps,
                            paired.loss_plus, paired.loss_minus, (double) loss_delta, (double) g,
                            paired.t_pp_us / 1000.0, speed_pp_paired,
                            t_step_us / 1000.0);

                    LOG("  antithetic pipeline prep: b_pair=%.3f ms next_noise=%.3f ms | wait_next_noise=%.3f ms\n",
                            t_prepare_b_pair_us / 1000.0,
                            t_prepare_next_noise_us / 1000.0,
                            t_wait_next_noise_ready_us / 1000.0);
                    LOG("  approx push/reload before paired forward: b_pair=%.3f ms\n",
                            t_push_b_pair_us / 1000.0);
                }
            }
        }

        const int tokens_per_forward = opts.antithetic ? 2 * total_tokens : total_tokens;
        const double avg_speed_pp = (double) (stats.total_forwards * tokens_per_forward) / ((double) stats.total_pp_us / 1e6);
        const double avg_step_ms  = (double) stats.total_step_us / (double) opts.steps / 1000.0;

        LOG("\n");
        LOG("summary:\n");
        LOG("  pipeline            = %s\n", opts.pipeline ? "true" : "false");
        LOG("  antithetic          = %s\n", opts.antithetic ? "true" : "false");
        LOG("  speed_pp            = %.2f t/s\n", avg_speed_pp);
        LOG("  avg_step_time       = %.3f ms\n", avg_step_ms);
        if (opts.antithetic) {
            LOG("  avg_t_pp_per_paired_fwd = %.3f ms\n", (double) stats.total_pp_us / (double) stats.total_forwards / 1000.0);
        } else {
            LOG("  avg_t_pp_per_fwd    = %.3f ms\n", (double) stats.total_pp_us / (double) stats.total_forwards / 1000.0);
        }
        LOG("  avg_prepare_noise   = %.3f ms\n", (double) stats.total_prepare_noise_us / (double) opts.steps / 1000.0);
        if (opts.antithetic) {
            LOG("  avg_prepare_b_pair  = %.3f ms\n", (double) stats.total_prepare_plus_us / (double) opts.steps / 1000.0);
        } else {
            LOG("  avg_prepare_plus    = %.3f ms\n", (double) stats.total_prepare_plus_us / (double) opts.steps / 1000.0);
            LOG("  avg_prepare_minus   = %.3f ms\n", (double) stats.total_prepare_minus_us / (double) opts.steps / 1000.0);
        }
        LOG("  avg_prepare_update  = %.3f ms\n", (double) stats.total_prepare_update_us / (double) opts.steps / 1000.0);

        if (opts.pipeline) {
            const double next_noise_denom = (double) std::max(1, opts.steps - 1);
            LOG("  avg_prepare_next_noise = %.3f ms\n", (double) stats.total_prepare_next_noise_us / next_noise_denom / 1000.0);
            LOG("  avg_copy_plus_logits = %.3f ms\n", (double) stats.total_copy_plus_logits_us / (double) opts.steps / 1000.0);
            LOG("  avg_wait_minus_ready = %.3f ms\n", (double) stats.total_wait_minus_ready_us / (double) opts.steps / 1000.0);
            LOG("  avg_wait_next_noise_ready = %.3f ms\n", (double) stats.total_wait_next_noise_ready_us / next_noise_denom / 1000.0);
            LOG("  avg_wait_plus_loss_ready = %.3f ms\n", (double) stats.total_wait_plus_loss_ready_us / (double) opts.steps / 1000.0);
        }

        if (opts.mode == zo_mode::COOP) {
            if (opts.antithetic) {
                const double avg_push_before_forward_ms =
                    (double) stats.total_push_antithetic_b_us / (double) stats.total_forwards / 1000.0;
                LOG("  avg_push_before_fwd = %.3f ms (approx LoRA B+/B- RM push time)\n", avg_push_before_forward_ms);
                LOG("  avg_push_b_pair     = %.3f ms\n", (double) stats.total_push_antithetic_b_us / (double) opts.steps / 1000.0);
            } else {
                const double avg_push_before_forward_ms =
                    (double) (stats.total_push_plus_us + stats.total_push_minus_us) /
                    (double) (stats.total_forwards) / 1000.0;
                LOG("  avg_push_before_fwd = %.3f ms (approx repack/reload push time)\n", avg_push_before_forward_ms);
                LOG("  avg_push_plus       = %.3f ms\n", (double) stats.total_push_plus_us / (double) opts.steps / 1000.0);
                LOG("  avg_push_minus      = %.3f ms\n", (double) stats.total_push_minus_us / (double) opts.steps / 1000.0);
            }
        }
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
