#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-base.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#if __HVX_ARCH__ < 79
#define HTP_LORA_HVX_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HTP_LORA_HVX_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HTP_LORA_HVX_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HTP_LORA_HVX_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

struct htp_lora_context {
    struct htp_ops_context * octx;
    float *                  tmp;
    float                    scale;
    uint32_t                 positive_rows;
    uint32_t                 k;
    uint32_t                 r;
    uint32_t                 m;
    uint32_t                 n;
    int                      b_rm_f32;
};

struct htp_lora_direct_acc_context {
    const struct htp_tensor * dst;
    const float *             tmp;
    const float *             b_rm_plus;
    const float *             b_rm_minus;
    float                     b_scale;
    uint32_t                  positive_rows;
    uint32_t                  r;
    uint32_t                  m;
    uint32_t                  n;
};

static inline float htp_lora_get_f32_2d(const struct htp_tensor * t, uint32_t i0, uint32_t i1) {
    const uint8_t * p = (const uint8_t *)(uintptr_t) t->data + i1*t->nb[1] + i0*t->nb[0];

    switch (t->type) {
        case HTP_TYPE_F32:
            return *(const float *) p;
        case HTP_TYPE_F16:
            return (float) *(const _Float16 *) p;
        default:
            return 0.0f;
    }
}

static inline float * htp_lora_dst_f32_2d(const struct htp_tensor * t, uint32_t i0, uint32_t i1) {
    return (float *) ((uint8_t *)(uintptr_t) t->data + i1*t->nb[1] + i0*t->nb[0]);
}

static inline float htp_lora_absf(float v) {
    return v < 0.0f ? -v : v;
}

static float htp_lora_get_scale(struct htp_ops_context * octx) {
    if (octx->op_desc && octx->op_desc->lora_scale != 0.0f) {
        return octx->op_desc->lora_scale;
    }

    float scale = 0.0f;
    memcpy(&scale, octx->op_params, sizeof(scale));
    return scale;
}

static uint32_t htp_lora_get_positive_rows(struct htp_ops_context * octx) {
    return octx->op_desc ? octx->op_desc->lora_positive_rows : 0;
}

static inline int htp_lora_profile_enabled(struct htp_ops_context * octx) {
    return (octx->flags & HTP_OPFLAGS_LORA_PROFILE) && octx->op_desc;
}

static inline uint64_t htp_lora_profile_start(struct htp_ops_context * octx) {
    return htp_lora_profile_enabled(octx) ? HAP_perf_get_qtimer_count() : 0;
}

static inline uint32_t htp_lora_profile_stop_us(uint64_t start) {
    return start ? (uint32_t) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - start) : 0;
}

static void htp_lora_clear_spad_cache(struct htp_ops_context * octx) {
    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->src2_spad.src = NULL;
    octx->src3_spad.src = NULL;
    octx->dst_spad.src  = NULL;
}

static struct htp_tensor htp_lora_make_f32_tensor(float * data, uint32_t ne0, uint32_t ne1) {
    struct htp_tensor t;
    memset(&t, 0, sizeof(t));

    t.data  = (uint32_t) (uintptr_t) data;
    t.size  = ne0 * ne1 * sizeof(float);
    t.flags = HTP_TENSOR_COMPUTE | HTP_TENSOR_FLUSHED;
    t.type  = HTP_TYPE_F32;
    t.bi    = 0;

    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = ne0 * sizeof(float);
    t.nb[2] = ne1 * t.nb[1];
    t.nb[3] = t.nb[2];

    return t;
}

static int htp_lora_call_matmul(
        struct htp_ops_context * octx,
        const struct htp_tensor * src0,
        const struct htp_tensor * src1,
        const struct htp_tensor * dst,
        uint32_t flags) {
    struct htp_tensor dummy_src2;
    memset(&dummy_src2, 0, sizeof(dummy_src2));
    dummy_src2.type  = HTP_TYPE_F32;
    dummy_src2.ne[0] = 1;
    dummy_src2.ne[1] = 1;
    dummy_src2.ne[2] = 1;
    dummy_src2.ne[3] = 1;
    dummy_src2.nb[0] = sizeof(float);
    dummy_src2.nb[1] = sizeof(float);
    dummy_src2.nb[2] = sizeof(float);
    dummy_src2.nb[3] = sizeof(float);

    const enum htp_op_code    saved_op      = octx->op;
    const uint32_t            saved_flags   = octx->flags;
    struct htp_op_desc *      saved_op_desc = octx->op_desc;
    const struct htp_tensor * saved_src[HTP_OP_MAX_INPUTS];
    const struct htp_tensor * saved_dst     = octx->dst;

    memcpy(saved_src, octx->src, sizeof(saved_src));

    memset(octx->src, 0, sizeof(octx->src));
    octx->src[0]  = src0;
    octx->src[1]  = src1;
    octx->src[2]  = &dummy_src2;
    octx->dst     = dst;
    octx->op      = HTP_OP_MUL_MAT;
    octx->flags   = flags;
    octx->op_desc = NULL;

    // Internal matmuls use the same VTCM scratch.  Clear cached src markers so
    // stale quantized activation data from the outer matmul is never reused.
    htp_lora_clear_spad_cache(octx);

    const int status = op_matmul(octx);

    memcpy(octx->src, saved_src, sizeof(saved_src));
    octx->dst     = saved_dst;
    octx->op      = saved_op;
    octx->flags   = saved_flags;
    octx->op_desc = saved_op_desc;

    // The last internal matmul may leave VTCM cache metadata pointing at
    // temporary heap tensors.  The data will be freed, so invalidate metadata.
    htp_lora_clear_spad_cache(octx);

    return status;
}

static void htp_lora_debug_write(
        struct htp_ops_context * octx,
        float                    scale,
        uint32_t                 k,
        uint32_t                 r,
        uint32_t                 m,
        uint32_t                 n,
        const float *            tmp,
        uint32_t                 status,
        uint32_t                 stage) {
    if (!(octx->flags & HTP_OPFLAGS_LORA_DEBUG) || !octx->op_desc) {
        return;
    }

    static unsigned int n_printed = 0;
    const unsigned int debug_idx = ++n_printed;

    const struct htp_tensor * lora_b       = octx->src[3];
    const struct htp_tensor * lora_b_minus = octx->src[4];
    struct htp_op_desc *      op     = octx->op_desc;
    const int deep_debug = (octx->flags & HTP_OPFLAGS_LORA_DEBUG_DEEP) != 0;
    const int b_rm_f32 = (octx->flags & HTP_OPFLAGS_LORA_B_RM_F32) != 0;

    float sum_abs_tmp     = 0.0f;
    float sum_abs_b       = 0.0f;
    float sum_abs_b_minus = 0.0f;
    float sum_abs_delta   = 0.0f;

    if (deep_debug && tmp) {
        for (uint32_t im = 0; im < m; ++im) {
            for (uint32_t ir = 0; ir < r; ++ir) {
                sum_abs_tmp += htp_lora_absf(tmp[(size_t) im * r + ir]);
            }
        }
    }

    if (deep_debug && lora_b && (lora_b->type == HTP_TYPE_F16 || lora_b->type == HTP_TYPE_F32)) {
        for (uint32_t in = 0; in < n; ++in) {
            for (uint32_t ir = 0; ir < r; ++ir) {
                sum_abs_b += htp_lora_absf(b_rm_f32 ? htp_lora_get_f32_2d(lora_b, in, ir) : htp_lora_get_f32_2d(lora_b, ir, in));
            }
        }
    }

    if (deep_debug && lora_b_minus && (lora_b_minus->type == HTP_TYPE_F16 || lora_b_minus->type == HTP_TYPE_F32)) {
        for (uint32_t in = 0; in < n; ++in) {
            for (uint32_t ir = 0; ir < r; ++ir) {
                sum_abs_b_minus += htp_lora_absf(b_rm_f32 ? htp_lora_get_f32_2d(lora_b_minus, in, ir) : htp_lora_get_f32_2d(lora_b_minus, ir, in));
            }
        }
    }

    if (deep_debug && tmp && lora_b && (lora_b->type == HTP_TYPE_F16 || lora_b->type == HTP_TYPE_F32)) {
        for (uint32_t in = 0; in < n; ++in) {
            for (uint32_t im = 0; im < m; ++im) {
                float delta = 0.0f;
                for (uint32_t ir = 0; ir < r; ++ir) {
                    delta += scale * (b_rm_f32 ? htp_lora_get_f32_2d(lora_b, in, ir) : htp_lora_get_f32_2d(lora_b, ir, in)) *
                             tmp[(size_t) im * r + ir];
                }
                sum_abs_delta += htp_lora_absf(delta);
            }
        }
    }

    op->debug_k               = k;
    op->debug_r               = r;
    op->debug_m               = m;
    op->debug_n               = n;
    op->debug_scale           = scale;
    op->debug_sum_abs_tmp     = sum_abs_tmp;
    op->debug_sum_abs_b       = sum_abs_b;
    op->debug_sum_abs_b_minus = sum_abs_b_minus;
    op->debug_sum_abs_delta   = sum_abs_delta;
    op->debug_status          = status;
    op->debug_stage           = stage;
    op->debug_flags          |= HTP_OP_DEBUG_LORA_STATS;

    if (debug_idx <= 32) {
        FARF(ALWAYS, "lora_mul_mat debug #%u: status=%u stage=%u deep=%d scale=%f k=%u r=%u m=%u n=%u sum_abs_tmp=%f sum_abs_b=%f sum_abs_b_minus=%f sum_abs_delta=%f",
                debug_idx, status, stage, deep_debug, scale, k, r, m, n, sum_abs_tmp, sum_abs_b, sum_abs_b_minus, sum_abs_delta);
    }
}

static void lora_compute_tmp_job(unsigned int nth, unsigned int ith, void * data) {
    struct htp_lora_context * lctx = (struct htp_lora_context *) data;
    struct htp_ops_context *  octx = lctx->octx;

    const struct htp_tensor * x      = octx->src[1];
    const struct htp_tensor * lora_a = octx->src[2];

    const uint32_t k = lctx->k;
    const uint32_t r = lctx->r;
    const uint32_t m = lctx->m;

    const uint32_t chunk = (m + nth - 1) / nth;
    const uint32_t start = chunk * ith;
    const uint32_t end   = MIN(start + chunk, m);

    if ((octx->flags & HTP_OPFLAGS_LORA_FAST) && r == 8) {
        for (uint32_t im = start; im < end; ++im) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;

            for (uint32_t ik = 0; ik < k; ++ik) {
                const float xv = htp_lora_get_f32_2d(x, ik, im);
                s0 += htp_lora_get_f32_2d(lora_a, ik, 0) * xv;
                s1 += htp_lora_get_f32_2d(lora_a, ik, 1) * xv;
                s2 += htp_lora_get_f32_2d(lora_a, ik, 2) * xv;
                s3 += htp_lora_get_f32_2d(lora_a, ik, 3) * xv;
                s4 += htp_lora_get_f32_2d(lora_a, ik, 4) * xv;
                s5 += htp_lora_get_f32_2d(lora_a, ik, 5) * xv;
                s6 += htp_lora_get_f32_2d(lora_a, ik, 6) * xv;
                s7 += htp_lora_get_f32_2d(lora_a, ik, 7) * xv;
            }

            float * tmp = lctx->tmp + (size_t) im * 8;
            tmp[0] = s0; tmp[1] = s1; tmp[2] = s2; tmp[3] = s3;
            tmp[4] = s4; tmp[5] = s5; tmp[6] = s6; tmp[7] = s7;
        }
        return;
    }

    for (uint32_t im = start; im < end; ++im) {
        float acc[32];
        for (uint32_t ir = 0; ir < r; ++ir) {
            acc[ir] = 0.0f;
        }

        for (uint32_t ik = 0; ik < k; ++ik) {
            const float xv = htp_lora_get_f32_2d(x, ik, im);
            for (uint32_t ir = 0; ir < r; ++ir) {
                acc[ir] += htp_lora_get_f32_2d(lora_a, ik, ir) * xv;
            }
        }

        memcpy(lctx->tmp + (size_t) im * r, acc, (size_t) r * sizeof(float));
    }
}

static void lora_accumulate_job(unsigned int nth, unsigned int ith, void * data) {
    struct htp_lora_context * lctx = (struct htp_lora_context *) data;
    struct htp_ops_context *  octx = lctx->octx;

    const struct htp_tensor * lora_b = octx->src[3];
    const struct htp_tensor * lora_b_minus = octx->src[4];
    const struct htp_tensor * dst    = octx->dst;
    const int antithetic = (octx->flags & HTP_OPFLAGS_LORA_ANTITHETIC) && lora_b_minus;

    const uint32_t r = lctx->r;
    const uint32_t n = lctx->n;
    const uint32_t m = lctx->m;

    const uint32_t chunk = (n + nth - 1) / nth;
    const uint32_t start = chunk * ith;
    const uint32_t end   = MIN(start + chunk, n);

    if ((octx->flags & HTP_OPFLAGS_LORA_FAST) && r == 8) {
        for (uint32_t in = start; in < end; ++in) {
            for (uint32_t im = 0; im < m; ++im) {
                const struct htp_tensor * b_src = (antithetic && im >= lctx->positive_rows) ? lora_b_minus : lora_b;
                const float b0 = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, 0) : htp_lora_get_f32_2d(b_src, 0, in));
                const float b1 = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, 1) : htp_lora_get_f32_2d(b_src, 1, in));
                const float b2 = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, 2) : htp_lora_get_f32_2d(b_src, 2, in));
                const float b3 = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, 3) : htp_lora_get_f32_2d(b_src, 3, in));
                const float b4 = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, 4) : htp_lora_get_f32_2d(b_src, 4, in));
                const float b5 = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, 5) : htp_lora_get_f32_2d(b_src, 5, in));
                const float b6 = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, 6) : htp_lora_get_f32_2d(b_src, 6, in));
                const float b7 = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, 7) : htp_lora_get_f32_2d(b_src, 7, in));
                const float * tmp = lctx->tmp + (size_t) im * 8;
                float delta = b0 * tmp[0] + b1 * tmp[1] + b2 * tmp[2] + b3 * tmp[3] +
                              b4 * tmp[4] + b5 * tmp[5] + b6 * tmp[6] + b7 * tmp[7];

                float * out = htp_lora_dst_f32_2d(dst, in, im);
                *out += delta;
            }
        }
        return;
    }

    for (uint32_t in = start; in < end; ++in) {
        float b[32];

        for (uint32_t im = 0; im < m; ++im) {
            const struct htp_tensor * b_src = (antithetic && im >= lctx->positive_rows) ? lora_b_minus : lora_b;
            for (uint32_t ir = 0; ir < r; ++ir) {
                b[ir] = lctx->scale * (lctx->b_rm_f32 ? htp_lora_get_f32_2d(b_src, in, ir) : htp_lora_get_f32_2d(b_src, ir, in));
            }
            float delta = 0.0f;
            for (uint32_t ir = 0; ir < r; ++ir) {
                delta += b[ir] * lctx->tmp[(size_t) im * r + ir];
            }

            float * out = htp_lora_dst_f32_2d(dst, in, im);
            *out += delta;
        }
    }
}

static int htp_lora_is_contiguous_f32_2d(const struct htp_tensor * t, uint32_t ne0, uint32_t ne1) {
    return t &&
           t->type == HTP_TYPE_F32 &&
           t->ne[0] == ne0 &&
           t->ne[1] == ne1 &&
           t->ne[2] == 1 &&
           t->ne[3] == 1 &&
           t->nb[0] == sizeof(float) &&
           t->nb[1] == ne0 * sizeof(float);
}

static void htp_lora_make_b_rank_major_f32(
        const struct htp_tensor * lora_b,
        float *                   b_rm,
        float                     scale,
        uint32_t                  r,
        uint32_t                  n) {
    // HTP tensors are contiguous over output channels (N) for each token.
    // Keep B rank-major so the direct accumulate kernel can vectorize across N.
    for (uint32_t in = 0; in < n; ++in) {
        for (uint32_t ir = 0; ir < r; ++ir) {
            b_rm[ir * (size_t) n + in] = scale * htp_lora_get_f32_2d(lora_b, ir, in);
        }
    }
}

// r must be a multiple of 8.  Each 8-rank segment keeps the original rank-8
// register footprint (8 splats + 8 B rows + acc); higher ranks do r/8 passes
// over the output row instead of holding all splats live at once.
static void lora_rankx8_direct_acc_hvx_job(unsigned int nth, unsigned int ith, void * data) {
    struct htp_lora_direct_acc_context * actx = (struct htp_lora_direct_acc_context *) data;

    float * restrict dst = (float *) (uintptr_t) actx->dst->data;

    const uint32_t r = actx->r;
    const uint32_t m = actx->m;
    const uint32_t n = actx->n;

    const uint32_t chunk = (m + nth - 1) / nth;
    const uint32_t start = chunk * ith;
    const uint32_t end   = MIN(start + chunk, m);

    for (uint32_t im = start; im < end; ++im) {
        const float * restrict b_rm_base = (actx->b_rm_minus && im >= actx->positive_rows) ? actx->b_rm_minus : actx->b_rm_plus;
        float * restrict out = dst + (size_t) im * n;
        const float b_scale = actx->b_scale;

        for (uint32_t ir0 = 0; ir0 < r; ir0 += 8) {
            const float * restrict tmp  = actx->tmp + (size_t) im * r + ir0;
            const float * restrict b_rm = b_rm_base + (size_t) ir0 * n;

            const HVX_Vector vt0 = hvx_vec_splat_f32(tmp[0] * b_scale);
            const HVX_Vector vt1 = hvx_vec_splat_f32(tmp[1] * b_scale);
            const HVX_Vector vt2 = hvx_vec_splat_f32(tmp[2] * b_scale);
            const HVX_Vector vt3 = hvx_vec_splat_f32(tmp[3] * b_scale);
            const HVX_Vector vt4 = hvx_vec_splat_f32(tmp[4] * b_scale);
            const HVX_Vector vt5 = hvx_vec_splat_f32(tmp[5] * b_scale);
            const HVX_Vector vt6 = hvx_vec_splat_f32(tmp[6] * b_scale);
            const HVX_Vector vt7 = hvx_vec_splat_f32(tmp[7] * b_scale);

            uint32_t in = 0;
            for (; in + VLEN_FP32 <= n; in += VLEN_FP32) {
                HVX_Vector acc = ((const HVX_UVector *) (out + in))[0];

                const HVX_Vector b0 = ((const HVX_UVector *) (b_rm + 0 * (size_t) n + in))[0];
                const HVX_Vector b1 = ((const HVX_UVector *) (b_rm + 1 * (size_t) n + in))[0];
                const HVX_Vector b2 = ((const HVX_UVector *) (b_rm + 2 * (size_t) n + in))[0];
                const HVX_Vector b3 = ((const HVX_UVector *) (b_rm + 3 * (size_t) n + in))[0];
                const HVX_Vector b4 = ((const HVX_UVector *) (b_rm + 4 * (size_t) n + in))[0];
                const HVX_Vector b5 = ((const HVX_UVector *) (b_rm + 5 * (size_t) n + in))[0];
                const HVX_Vector b6 = ((const HVX_UVector *) (b_rm + 6 * (size_t) n + in))[0];
                const HVX_Vector b7 = ((const HVX_UVector *) (b_rm + 7 * (size_t) n + in))[0];

                acc = HTP_LORA_HVX_ADD_F32(acc, HTP_LORA_HVX_MUL_F32(b0, vt0));
                acc = HTP_LORA_HVX_ADD_F32(acc, HTP_LORA_HVX_MUL_F32(b1, vt1));
                acc = HTP_LORA_HVX_ADD_F32(acc, HTP_LORA_HVX_MUL_F32(b2, vt2));
                acc = HTP_LORA_HVX_ADD_F32(acc, HTP_LORA_HVX_MUL_F32(b3, vt3));
                acc = HTP_LORA_HVX_ADD_F32(acc, HTP_LORA_HVX_MUL_F32(b4, vt4));
                acc = HTP_LORA_HVX_ADD_F32(acc, HTP_LORA_HVX_MUL_F32(b5, vt5));
                acc = HTP_LORA_HVX_ADD_F32(acc, HTP_LORA_HVX_MUL_F32(b6, vt6));
                acc = HTP_LORA_HVX_ADD_F32(acc, HTP_LORA_HVX_MUL_F32(b7, vt7));

                hvx_vec_store_u(out + in, VLEN, acc);
            }

            for (; in < n; ++in) {
                float delta = 0.0f;
                delta += b_rm[0 * (size_t) n + in] * tmp[0] * b_scale;
                delta += b_rm[1 * (size_t) n + in] * tmp[1] * b_scale;
                delta += b_rm[2 * (size_t) n + in] * tmp[2] * b_scale;
                delta += b_rm[3 * (size_t) n + in] * tmp[3] * b_scale;
                delta += b_rm[4 * (size_t) n + in] * tmp[4] * b_scale;
                delta += b_rm[5 * (size_t) n + in] * tmp[5] * b_scale;
                delta += b_rm[6 * (size_t) n + in] * tmp[6] * b_scale;
                delta += b_rm[7 * (size_t) n + in] * tmp[7] * b_scale;
                out[in] += delta;
            }
        }
    }
}

static int htp_lora_accumulate_rankx8_hvx(
        struct htp_ops_context * octx,
        const struct htp_tensor * dst,
        const float *             tmp,
        const float *             b_rm_plus,
        const float *             b_rm_minus,
        float                     b_scale,
        uint32_t                  r,
        uint32_t                  n,
        uint32_t                  m,
        uint32_t                  positive_rows) {
    if (!htp_lora_is_contiguous_f32_2d(dst, n, m)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_lora_direct_acc_context actx = {
        .dst  = dst,
        .tmp  = tmp,
        .b_rm_plus  = b_rm_plus,
        .b_rm_minus = b_rm_minus,
        .b_scale = b_scale,
        .positive_rows = positive_rows,
        .r    = r,
        .m    = m,
        .n    = n,
    };

    const uint32_t n_threads = octx->n_threads ? MIN(octx->n_threads, (uint32_t) MAX_NUM_WORKERS) : 1;
    AEEResult ret = worker_pool_run_func(octx->ctx->worker_pool, lora_rankx8_direct_acc_hvx_job, &actx, n_threads);
    if (ret != AEE_SUCCESS) {
        FARF(ERROR, "lora_mul_mat: worker_pool_run_func(rankx8-direct-acc) failed: 0x%x", ret);
        return HTP_STATUS_INTERNAL_ERR;
    }

    return HTP_STATUS_OK;
}

static int htp_lora_compute_fast(
        struct htp_ops_context * octx,
        const struct htp_tensor * x,
        const struct htp_tensor * lora_a,
        const struct htp_tensor * lora_b,
        const struct htp_tensor * lora_b_minus,
        const struct htp_tensor * dst,
        float scale,
        uint32_t positive_rows,
        uint32_t k,
        uint32_t r,
        uint32_t m,
        uint32_t n,
        int      b_rm_f32) {
    if (r == 0 || r > 32 || (r % 8) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (!htp_lora_is_contiguous_f32_2d(dst, n, m)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t tmp_size  = (size_t) r * (size_t) m * sizeof(float);
    const size_t b_rm_size = (size_t) r * (size_t) n * sizeof(float);
    const int antithetic = lora_b_minus != NULL;

    float * tmp = (float *) memalign(VLEN, hex_align_up(tmp_size, VLEN));
    float * b_rm_plus_alloc  = b_rm_f32 ? NULL : (float *) memalign(VLEN, hex_align_up(b_rm_size, VLEN));
    float * b_rm_minus_alloc = (!b_rm_f32 && antithetic) ? (float *) memalign(VLEN, hex_align_up(b_rm_size, VLEN)) : NULL;
    const float * b_rm_plus  = b_rm_f32 ? (const float *) (uintptr_t) lora_b->data : b_rm_plus_alloc;
    const float * b_rm_minus = b_rm_f32 && antithetic ? (const float *) (uintptr_t) lora_b_minus->data : b_rm_minus_alloc;

    if (!tmp || !b_rm_plus || (antithetic && !b_rm_minus)) {
        FARF(ERROR, "lora_mul_mat: failed to allocate fast tmp buffers tmp=%p b_plus=%p b_minus=%p sizes=%zu/%zu",
                tmp, (const void *) b_rm_plus, (const void *) b_rm_minus, tmp_size, b_rm_size);
        htp_lora_debug_write(octx, scale, k, r, m, n, tmp, HTP_STATUS_INTERNAL_ERR, HTP_LORA_STAGE_OOM);
        free(tmp);
        free(b_rm_plus_alloc);
        free(b_rm_minus_alloc);
        return HTP_STATUS_INTERNAL_ERR;
    }

    struct htp_tensor tmp_tensor = htp_lora_make_f32_tensor(tmp, r, m);

    uint64_t t0 = htp_lora_profile_start(octx);
    int err = htp_lora_call_matmul(octx, lora_a, x, &tmp_tensor, 0);
    if (htp_lora_profile_enabled(octx)) {
        octx->op_desc->lora_prof_tmp_us = htp_lora_profile_stop_us(t0);
    }
    if (!htp_status_is_ok(err)) {
        FARF(ERROR, "lora_mul_mat: fast tmp=A*x matmul failed: %d", err);
        htp_lora_debug_write(octx, scale, k, r, m, n, tmp, (uint32_t) err, HTP_LORA_STAGE_TMP_MATMUL);
        free(tmp);
        free(b_rm_plus_alloc);
        free(b_rm_minus_alloc);
        return err;
    }

    t0 = htp_lora_profile_start(octx);
    if (!b_rm_f32) {
        htp_lora_make_b_rank_major_f32(lora_b, b_rm_plus_alloc, scale, r, n);
        if (antithetic) {
            htp_lora_make_b_rank_major_f32(lora_b_minus, b_rm_minus_alloc, scale, r, n);
        }
    }
    if (htp_lora_profile_enabled(octx)) {
        octx->op_desc->lora_prof_bprep_us = htp_lora_profile_stop_us(t0);
    }

    // Directly accumulate B * (A * x) into dst.  This avoids materializing the
    // full F32 delta[N, M] tensor and the second generic matmul.
    t0 = htp_lora_profile_start(octx);
    err = htp_lora_accumulate_rankx8_hvx(octx, dst, tmp, b_rm_plus, b_rm_minus, b_rm_f32 ? scale : 1.0f, r, n, m, positive_rows);
    if (htp_lora_profile_enabled(octx)) {
        octx->op_desc->lora_prof_acc_us = htp_lora_profile_stop_us(t0);
    }
    if (!htp_status_is_ok(err)) {
        htp_lora_debug_write(octx, scale, k, r, m, n, tmp, (uint32_t) err, HTP_LORA_STAGE_SCALE_ADD);
        free(tmp);
        free(b_rm_plus_alloc);
        free(b_rm_minus_alloc);
        return err;
    }

    htp_lora_debug_write(octx, scale, k, r, m, n, tmp, HTP_STATUS_OK, HTP_LORA_STAGE_DONE);

    free(tmp);
    free(b_rm_plus_alloc);
    free(b_rm_minus_alloc);
    return HTP_STATUS_OK;
}

int op_lora_mul_mat(struct htp_ops_context * octx) {
    const float scale = htp_lora_get_scale(octx);

    const struct htp_tensor * w      = octx->src[0];
    const struct htp_tensor * x      = octx->src[1];
    const struct htp_tensor * lora_a = octx->src[2];
    const struct htp_tensor * lora_b = octx->src[3];
    const struct htp_tensor * lora_b_minus = octx->src[4];
    const struct htp_tensor * dst    = octx->dst;
    const int antithetic = (octx->flags & HTP_OPFLAGS_LORA_ANTITHETIC) != 0;
    const int b_rm_f32 = (octx->flags & HTP_OPFLAGS_LORA_B_RM_F32) != 0;

    htp_lora_debug_write(octx, scale, 0, 0, 0, 0, NULL, HTP_STATUS_OK, HTP_LORA_STAGE_ENTER);

    if (!w || !x || !lora_a || !lora_b || !dst) {
        htp_lora_debug_write(octx, scale, 0, 0, 0, 0, NULL, HTP_STATUS_INVAL_PARAMS, HTP_LORA_STAGE_BAD_PARAMS);
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t k = w->ne[0];
    const uint32_t n = w->ne[1];
    const uint32_t m = x->ne[1];
    const uint32_t r = lora_a->ne[1];
    const uint32_t positive_rows = htp_lora_get_positive_rows(octx);

    if (htp_lora_profile_enabled(octx)) {
        octx->op_desc->lora_prof_main_us  = 0;
        octx->op_desc->lora_prof_tmp_us   = 0;
        octx->op_desc->lora_prof_bprep_us = 0;
        octx->op_desc->lora_prof_acc_us   = 0;
        octx->op_desc->debug_k = k;
        octx->op_desc->debug_r = r;
        octx->op_desc->debug_m = m;
        octx->op_desc->debug_n = n;
    }

    if (dst->type != HTP_TYPE_F32 ||
        (w->type != HTP_TYPE_F16 && w->type != HTP_TYPE_Q4_0) ||
        lora_a->type != HTP_TYPE_F16 ||
        (!b_rm_f32 && lora_b->type != HTP_TYPE_F16) ||
        (b_rm_f32 && lora_b->type != HTP_TYPE_F32) ||
        (antithetic && (!lora_b_minus ||
            (!b_rm_f32 && lora_b_minus->type != HTP_TYPE_F16) ||
            (b_rm_f32 && lora_b_minus->type != HTP_TYPE_F32))) ||
        (x->type != HTP_TYPE_F32 && x->type != HTP_TYPE_F16)) {
        FARF(ERROR, "lora_mul_mat: only F16/Q4_0 W, F16 A/B(+/-), RM-F32 B(+/-), and F16/F32 activation are supported");
        htp_lora_debug_write(octx, scale, k, r, m, n, NULL, HTP_STATUS_NO_SUPPORT, HTP_LORA_STAGE_UNSUPPORTED);
        return HTP_STATUS_NO_SUPPORT;
    }

    if (w->type == HTP_TYPE_Q4_0 && (k % QK_Q4_0x4x2) != 0) {
        FARF(ERROR, "lora_mul_mat: Q4_0 W requires K aligned to %u", (unsigned) QK_Q4_0x4x2);
        htp_lora_debug_write(octx, scale, k, r, m, n, NULL, HTP_STATUS_NO_SUPPORT, HTP_LORA_STAGE_UNSUPPORTED);
        return HTP_STATUS_NO_SUPPORT;
    }

    if (r == 0 || r > 32 ||
        (antithetic && (r % 8) != 0) ||
        x->ne[0] != k ||
        lora_a->ne[0] != k ||
        (!b_rm_f32 && (lora_b->ne[0] != r || lora_b->ne[1] != n)) ||
        (b_rm_f32 && (lora_b->ne[0] != n || lora_b->ne[1] != r)) ||
        (antithetic && (lora_b_minus->ne[0] != lora_b->ne[0] || lora_b_minus->ne[1] != lora_b->ne[1])) ||
        (antithetic && (positive_rows == 0 || positive_rows * 2 != m)) ||
        dst->ne[0] != n ||
        dst->ne[1] != m) {
        FARF(ERROR, "lora_mul_mat: unsupported dimensions");
        htp_lora_debug_write(octx, scale, k, r, m, n, NULL, HTP_STATUS_NO_SUPPORT, HTP_LORA_STAGE_UNSUPPORTED);
        return HTP_STATUS_NO_SUPPORT;
    }

    uint64_t t0 = htp_lora_profile_start(octx);
    int err = htp_lora_call_matmul(octx, w, x, dst, octx->flags);
    if (htp_lora_profile_enabled(octx)) {
        octx->op_desc->lora_prof_main_us = htp_lora_profile_stop_us(t0);
    }

    if (!htp_status_is_ok(err)) {
        htp_lora_debug_write(octx, scale, k, r, m, n, NULL, (uint32_t) err, HTP_LORA_STAGE_MATMUL);
        return err;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        htp_lora_debug_write(octx, scale, k, r, m, n, NULL, HTP_STATUS_OK, HTP_LORA_STAGE_SKIP);
        return HTP_STATUS_OK;
    }

    if (scale == 0.0f) {
        htp_lora_debug_write(octx, scale, k, r, m, n, NULL, HTP_STATUS_OK, HTP_LORA_STAGE_SCALE_ZERO);
        return HTP_STATUS_OK;
    }

    if (octx->flags & HTP_OPFLAGS_LORA_FAST) {
        err = htp_lora_compute_fast(
                octx, x, lora_a, lora_b, antithetic ? lora_b_minus : NULL, dst,
                scale, antithetic ? positive_rows : m, k, r, m, n, b_rm_f32);
        if (htp_status_is_ok(err)) {
            return HTP_STATUS_OK;
        }

        if (err != HTP_STATUS_NO_SUPPORT) {
            return err;
        }

        FARF(HIGH, "lora_mul_mat: fast path unsupported, falling back to scalar reference");
    }

    float * tmp = (float *) malloc((size_t) r * (size_t) m * sizeof(float));
    if (!tmp) {
        FARF(ERROR, "lora_mul_mat: failed to allocate tmp buffer");
        htp_lora_debug_write(octx, scale, k, r, m, n, NULL, HTP_STATUS_INTERNAL_ERR, HTP_LORA_STAGE_OOM);
        return HTP_STATUS_INTERNAL_ERR;
    }

    struct htp_lora_context lctx = {
        .octx  = octx,
        .tmp   = tmp,
        .scale = scale,
        .positive_rows = antithetic ? positive_rows : m,
        .k     = k,
        .r     = r,
        .m     = m,
        .n     = n,
        .b_rm_f32 = b_rm_f32,
    };

    const uint32_t n_threads = octx->n_threads ? MIN(octx->n_threads, (uint32_t) MAX_NUM_WORKERS) : 1;

    AEEResult ret = worker_pool_run_func(octx->ctx->worker_pool, lora_compute_tmp_job, &lctx, n_threads);
    if (ret != AEE_SUCCESS) {
        htp_lora_debug_write(octx, scale, k, r, m, n, tmp, HTP_STATUS_INTERNAL_ERR, HTP_LORA_STAGE_TMP_FAIL);
        free(tmp);
        FARF(ERROR, "lora_mul_mat: worker_pool_run_func(tmp) failed: 0x%x", ret);
        return HTP_STATUS_INTERNAL_ERR;
    }

    ret = worker_pool_run_func(octx->ctx->worker_pool, lora_accumulate_job, &lctx, n_threads);

    if (ret != AEE_SUCCESS) {
        htp_lora_debug_write(octx, scale, k, r, m, n, tmp, HTP_STATUS_INTERNAL_ERR, HTP_LORA_STAGE_ACC_FAIL);
        free(tmp);
        FARF(ERROR, "lora_mul_mat: worker_pool_run_func(acc) failed: 0x%x", ret);
        return HTP_STATUS_INTERNAL_ERR;
    }

    htp_lora_debug_write(octx, scale, k, r, m, n, tmp, HTP_STATUS_OK, HTP_LORA_STAGE_DONE);
    free(tmp);

    return HTP_STATUS_OK;
}
