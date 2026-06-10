#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

int zoo_lora_fa_benchmark_main(int argc, char ** argv);

#define main zoo_lora_fa_benchmark_main
#include "zo-lora-fa.cpp"
#undef main

namespace {

struct repack_options {
    int iters = 10;
};

struct repack_timing {
    int64_t dequant_us = 0;
    int64_t merge_us   = 0;
    int64_t quant_us   = 0;
    int64_t repack_us  = 0;

    int64_t total_us() const {
        return dequant_us + merge_us + quant_us + repack_us;
    }
};

struct repack_weight {
    std::string        name;
    ggml_tensor *      base = nullptr;
    ggml_type          type = GGML_TYPE_COUNT;
    int64_t            k = 0;
    int64_t            n = 0;
    int64_t            r = 0;
    size_t             b_index = 0;
    float              scale = 1.0f;
    std::vector<uint8_t> base_quant_original;
    std::vector<float>   lora_a;
};

struct repack_scratch {
    std::vector<float>   merged_f32;
    std::vector<uint8_t> merged_quant;
};

static void repack_print_usage(int /*argc*/, char ** argv) {
    LOG("\n");
    LOG("usage:\n");
    LOG("  %s -m <int4.gguf> --lora <adapter.gguf> --device HTP0 -ngl 99 [options]\n", argv[0]);
    LOG("\n");
    LOG("repack benchmark options:\n");
    LOG("  --iters <N>        merge+NPU-repack iterations to time (default: 10)\n");
    LOG("\n");
    LOG("example:\n");
    LOG("  %s -m ./model-int4.gguf --lora ./lora.gguf --device HTP0 -ngl 99 --iters 10\n", argv[0]);
    LOG("\n");
}

static bool repack_parse_args(int argc, char ** argv, repack_options & opts, std::vector<char *> & filtered_argv) {
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

    if (opts.iters <= 0) {
        LOG_ERR("invalid --iters: must be > 0\n");
        return false;
    }

    return true;
}

static bool repack_is_int4_type(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ4_NL;
}

static const char * repack_tensor_type_name(ggml_type type) {
    const char * name = ggml_type_name(type);
    return name != nullptr ? name : "unknown";
}

static void repack_dequantize_to_float(const repack_weight & w, repack_scratch & scratch) {
    const ggml_type_traits * traits = ggml_get_type_traits(w.type);
    if (traits == nullptr || traits->to_float == nullptr) {
        throw std::runtime_error("missing dequantization trait for " + w.name);
    }
    scratch.merged_f32.resize((size_t) ggml_nelements(w.base));
    traits->to_float(w.base_quant_original.data(), scratch.merged_f32.data(), ggml_nelements(w.base));
}

static void repack_quantize_from_float(const repack_weight & w, repack_scratch & scratch) {
    if (ggml_quantize_requires_imatrix(w.type)) {
        throw std::runtime_error("quantization type requires imatrix and is unsupported: " + w.name);
    }

    scratch.merged_quant.resize(w.base_quant_original.size());
    const size_t written = ggml_quantize_chunk(
            w.type,
            scratch.merged_f32.data(),
            scratch.merged_quant.data(),
            0,
            w.n,
            w.k,
            nullptr);

    if (written != scratch.merged_quant.size()) {
        throw std::runtime_error("quantized byte count mismatch for " + w.name);
    }
}

static void repack_apply_lora_delta(
        const repack_weight & w,
        const std::vector<lora_b_tensor> & lora_b_tensors,
        repack_scratch & scratch) {
    const auto & b = lora_b_tensors[w.b_index].master;
    for (int64_t out = 0; out < w.n; ++out) {
        for (int64_t ir = 0; ir < w.r; ++ir) {
            const float bv = w.scale * b[(size_t) out * (size_t) w.r + (size_t) ir];
            if (bv == 0.0f) {
                continue;
            }

            const size_t a_base = (size_t) ir  * (size_t) w.k;
            const size_t w_base = (size_t) out * (size_t) w.k;
            for (int64_t in = 0; in < w.k; ++in) {
                scratch.merged_f32[w_base + (size_t) in] += w.lora_a[a_base + (size_t) in] * bv;
            }
        }
    }
}

static repack_timing repack_once(
        std::vector<repack_weight> & weights,
        const std::vector<lora_b_tensor> & lora_b_tensors,
        repack_scratch & scratch) {
    repack_timing timing;

    for (auto & w : weights) {
        const int64_t t_dequant_start = ggml_time_us();
        repack_dequantize_to_float(w, scratch);
        timing.dequant_us += ggml_time_us() - t_dequant_start;

        const int64_t t_merge_start = ggml_time_us();
        repack_apply_lora_delta(w, lora_b_tensors, scratch);
        timing.merge_us += ggml_time_us() - t_merge_start;

        const int64_t t_quant_start = ggml_time_us();
        repack_quantize_from_float(w, scratch);
        timing.quant_us += ggml_time_us() - t_quant_start;

        const int64_t t_repack_start = ggml_time_us();
        ggml_backend_tensor_set(w.base, scratch.merged_quant.data(), 0, scratch.merged_quant.size());
        timing.repack_us += ggml_time_us() - t_repack_start;
    }

    return timing;
}

static std::vector<repack_weight> collect_repack_weights(
        llama_adapter_lora * adapter,
        const std::vector<lora_b_tensor> & lora_b_tensors) {
    std::unordered_map<ggml_tensor *, size_t> b_to_index;
    for (size_t i = 0; i < lora_b_tensors.size(); ++i) {
        b_to_index[lora_b_tensors[i].tensor] = i;
    }

    std::vector<repack_weight> weights;
    size_t skipped_not_htp = 0;
    size_t skipped_missing = 0;

    for (auto & it : adapter->ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & lw = it.second;
        if (lw.a == nullptr || lw.b == nullptr) {
            ++skipped_missing;
            continue;
        }
        if (name.size() >= std::strlen("token_embd.weight") &&
                name.compare(name.size() - std::strlen("token_embd.weight"),
                    std::strlen("token_embd.weight"), "token_embd.weight") == 0) {
            throw std::runtime_error("repack benchmark does not support token embedding LoRA: " + name);
        }

        const auto b_pos = b_to_index.find(lw.b);
        if (b_pos == b_to_index.end()) {
            throw std::runtime_error("internal error: missing LoRA B tensor for " + name);
        }

        ggml_tensor * base = const_cast<ggml_tensor *>(adapter->model->get_tensor(name.c_str()));
        if (base == nullptr) {
            throw std::runtime_error("base tensor not found for LoRA target: " + name);
        }

        if (!lora_tensor_on_hexagon(base)) {
            ++skipped_not_htp;
            continue;
        }

        if (!repack_is_int4_type(base->type)) {
            throw std::runtime_error(
                    "HTP LoRA target is not supported INT4 type: " + name +
                    " type=" + repack_tensor_type_name(base->type));
        }
        if (lw.a->type != GGML_TYPE_F16 && lw.a->type != GGML_TYPE_F32) {
            throw std::runtime_error("LoRA-A must be F16/F32 for " + name);
        }
        if (base->ne[0] != lw.a->ne[0] || base->ne[1] != lw.b->ne[1] || lw.a->ne[1] != lw.b->ne[0]) {
            throw std::runtime_error("LoRA shape mismatch for " + name);
        }

        const ggml_type_traits * traits = ggml_get_type_traits(base->type);
        if (traits == nullptr || traits->to_float == nullptr) {
            throw std::runtime_error("missing INT4 dequantization trait for " + name);
        }
        if (base->ne[0] % traits->blck_size != 0) {
            throw std::runtime_error("INT4 base K dimension is not aligned to quant block size for " + name);
        }

        repack_weight entry;
        entry.name = name;
        entry.base = base;
        entry.type = base->type;
        entry.k = base->ne[0];
        entry.n = base->ne[1];
        entry.r = lw.b->ne[0];
        entry.b_index = b_pos->second;
        entry.scale = lw.get_scale(adapter->alpha, 1.0f);

        const size_t base_nbytes = ggml_nbytes(base);
        entry.base_quant_original.resize(base_nbytes);

        ggml_backend_tensor_get(base, entry.base_quant_original.data(), 0, base_nbytes);

        std::vector<uint8_t> a_buf(ggml_nbytes(lw.a));
        entry.lora_a.resize((size_t) ggml_nelements(lw.a));
        ggml_backend_tensor_get(lw.a, a_buf.data(), 0, a_buf.size());
        deserialize_tensor_to_float(a_buf, lw.a->type, entry.lora_a);

        weights.emplace_back(std::move(entry));
    }

    LOG("repack target collection: selected=%zu skipped_not_htp=%zu skipped_missing=%zu\n",
            weights.size(), skipped_not_htp, skipped_missing);

    if (weights.empty()) {
        throw std::runtime_error("no LoRA-adapted INT4 HTP base tensors found; use --device HTP0 -ngl 99 with an INT4 model");
    }

    return weights;
}

static std::string repack_first_lora_path(const common_params & params) {
    if (params.lora_adapters.empty()) {
        return "<none>";
    }
    return params.lora_adapters.front().path;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx       = 32;
    params.n_batch     = 32;
    params.n_ubatch    = 32;
    params.n_parallel  = 1;
    params.n_sequences = 1;
    params.warmup      = false;
    params.lora_init_without_apply = true;

    repack_options opts;
    std::vector<char *> filtered_argv;

    common_init();

    if (!repack_parse_args(argc, argv, opts, filtered_argv)) {
        repack_print_usage(argc, argv);
        return 1;
    }

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_BENCH, repack_print_usage)) {
        return 1;
    }
    params.lora_init_without_apply = true;

    bool backend_inited = false;

    try {
        if (!zo_hexagon_compiled()) {
            throw std::runtime_error("llama-zoo-repack requires a GGML_USE_HEXAGON build");
        }
        if (params.model.path.empty()) {
            throw std::runtime_error("missing model path: use -m <int4.gguf>");
        }
        if (params.lora_adapters.empty()) {
            throw std::runtime_error("missing LoRA adapter: use --lora <adapter.gguf>");
        }
        if (params.lora_adapters.size() != 1) {
            throw std::runtime_error("llama-zoo-repack supports exactly one LoRA adapter");
        }

        LOG("\n");
        LOG("llama-zoo-repack configuration:\n");
        LOG("  model      = %s\n", params.model.path.c_str());
        LOG("  lora       = %s\n", repack_first_lora_path(params).c_str());
        LOG("  iters      = %d\n", opts.iters);
        LOG("  n_gpu_layers = %d\n", params.n_gpu_layers);
        LOG("  devices    = %zu\n", params.devices.size());
        LOG("  lora_init_without_apply = true\n");
        LOG("\n");

        llama_backend_init();
        backend_inited = true;
        llama_numa_init(params.numa);

        auto init = common_init_from_params(params);
        if (!init) {
            throw std::runtime_error("common_init_from_params failed");
        }

        llama_model * model = init->model();
        if (model == nullptr) {
            throw std::runtime_error("failed to initialize model");
        }

        auto & loras = init->lora();
        if (loras.empty() || loras.front() == nullptr) {
            throw std::runtime_error("no LoRA adapter was loaded");
        }

        llama_adapter_lora * adapter = loras.front().get();
        auto lora_b_tensors = collect_lora_b_tensors(adapter);
        log_lora_summary(lora_b_tensors);
        log_lora_placement_summary(lora_b_tensors, false, false);

        auto weights = collect_repack_weights(adapter, lora_b_tensors);

        size_t total_quant_bytes = 0;
        size_t total_lora_a_bytes = 0;
        size_t max_scratch_f32_bytes = 0;
        size_t max_scratch_quant_bytes = 0;
        uint64_t total_elements = 0;
        for (const auto & w : weights) {
            total_quant_bytes += w.base_quant_original.size();
            total_lora_a_bytes += w.lora_a.size() * sizeof(float);
            max_scratch_f32_bytes = std::max(max_scratch_f32_bytes, (size_t) ggml_nelements(w.base) * sizeof(float));
            max_scratch_quant_bytes = std::max(max_scratch_quant_bytes, w.base_quant_original.size());
            total_elements += (uint64_t) ggml_nelements(w.base);
        }

        LOG("repack benchmark targets:\n");
        LOG("  adapted HTP INT4 tensors = %zu\n", weights.size());
        LOG("  quantized bytes          = %.3f MiB\n", (double) total_quant_bytes / 1024.0 / 1024.0);
        LOG("  cached LoRA-A bytes      = %.3f MiB\n", (double) total_lora_a_bytes / 1024.0 / 1024.0);
        LOG("  max scratch FP32 bytes   = %.3f MiB\n", (double) max_scratch_f32_bytes / 1024.0 / 1024.0);
        LOG("  max scratch INT4 bytes   = %.3f MiB\n", (double) max_scratch_quant_bytes / 1024.0 / 1024.0);
        LOG("  logical elements         = %.3f M\n", (double) total_elements / 1e6);
        LOG("\n");

        repack_scratch scratch;
        repack_timing totals;
        for (int iter = 0; iter < opts.iters; ++iter) {
            const repack_timing timing = repack_once(weights, lora_b_tensors, scratch);
            totals.dequant_us += timing.dequant_us;
            totals.merge_us   += timing.merge_us;
            totals.quant_us   += timing.quant_us;
            totals.repack_us  += timing.repack_us;

            LOG("iter %d/%d: merge+repack=%.3f ms | dequant=%.3f ms merge=%.3f ms quant=%.3f ms tensor_set_repack=%.3f ms\n",
                    iter + 1, opts.iters,
                    timing.total_us() / 1000.0,
                    timing.dequant_us / 1000.0,
                    timing.merge_us / 1000.0,
                    timing.quant_us / 1000.0,
                    timing.repack_us / 1000.0);
        }

        const double denom = (double) opts.iters * 1000.0;
        LOG("\n");
        LOG("repack summary:\n");
        LOG("  avg merge+repack time / iter = %.3f ms\n", totals.total_us() / denom);
        LOG("  avg dequant time / iter      = %.3f ms\n", totals.dequant_us / denom);
        LOG("  avg LoRA merge time / iter   = %.3f ms\n", totals.merge_us / denom);
        LOG("  avg quantize time / iter     = %.3f ms\n", totals.quant_us / denom);
        LOG("  avg NPU repack time / iter   = %.3f ms\n", totals.repack_us / denom);
        LOG("\n");

        init.reset();
        if (backend_inited) {
            llama_backend_free();
            backend_inited = false;
        }
        ggml_quantize_free();
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("error: %s\n", e.what());
        if (backend_inited) {
            llama_backend_free();
        }
        ggml_quantize_free();
        return 1;
    }
}
