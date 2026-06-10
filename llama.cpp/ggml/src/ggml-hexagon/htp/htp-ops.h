#ifndef HTP_OPS_H
#define HTP_OPS_H

#include <assert.h>

// ggml-common.h must be included prio to this header

enum htp_status {
    HTP_STATUS_OK             = 1,
    HTP_STATUS_INTERNAL_ERR   = 2,
    HTP_STATUS_NO_SUPPORT     = 3,
    HTP_STATUS_INVAL_PARAMS   = 4,
    HTP_STATUS_VTCM_TOO_SMALL = 5,
    HTP_STATUS_ABI_MISMATCH   = 6,
};

static inline int htp_status_is_ok(int status) {
    return status == 0 || status == HTP_STATUS_OK;
}

#define HTP_OPS_ABI_MAGIC   0x48545032u /* "HTP2" */
#define HTP_OPS_ABI_VERSION 7u

// First set of values must match the ggml_type.
// Duplicated here because we can't include full ggml.h in the htp build.
// We have some static_asserts in the cpp code to ensure things are in sync.
enum htp_data_type {
    HTP_TYPE_F32    = 0,
    HTP_TYPE_F16    = 1,
    HTP_TYPE_Q4_0   = 2,
    HTP_TYPE_Q8_0   = 8,
    HTP_TYPE_IQ4_NL = 20,
    HTP_TYPE_I32    = 26,
    HTP_TYPE_I64    = 27,
    HTP_TYPE_MXFP4  = 39,

    // types used internally for repack, dyn.quant, etc
    HTP_TYPE_Q4_0x4x2 = 200,
    HTP_TYPE_Q8_0x4x2,
    HTP_TYPE_MXFP4x4x2,

    HTP_TYPE_INVALID
};

// Constats for internal types
#define QK_Q4_0x4x2  256  // 4x Q4_0  blocks packed with next 4x Q4_0 blocks (size in bytes 128)
#define QK_Q8_0x4x2  256  // 4x Q8_0  blocks concat with next 4x Q8_0 blocks
#define QK_MXFP4x4x2 256  // 4x MXFP4 blocks concat with next 4x MXFP4 blocks


// Mask to enable various stages of the Ops.
// Used for debugging and profiling.
enum htp_op_mask {
    HTP_OPMASK_QUEUE    = (1 << 0),  // Enable Queueing (ie calls into the DSP)
    HTP_OPMASK_COMPUTE  = (1 << 1),  // Enable Compute
};

// Do not reorder first 4 (used as an index)
enum htp_op_code {
    HTP_OP_MUL = 0,
    HTP_OP_ADD = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_LORA_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_RMS_NORM,
    HTP_OP_UNARY_SILU,
    HTP_OP_UNARY_GELU,
    HTP_OP_UNARY_SIGMOID,
    HTP_OP_UNARY_EXP,
    HTP_OP_UNARY_NEG,
    HTP_OP_UNARY_SOFTPLUS,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_SOFTMAX,
    HTP_OP_ADD_ID,
    HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_SET_ROWS,
    HTP_OP_GET_ROWS,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_ARGSORT,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_SUM_ROWS,
    HTP_OP_SSM_CONV,
    HTP_OP_REPEAT,
    HTP_OP_CUMSUM,

    HTP_OP_INVALID
};

#define HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#define HTP_OP_MAX_INPUTS  6    // aka GGML_MAX_SRCS
#define HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS

#define HTP_OP_MAX_BUFS    8
#define HTP_OP_MAX_REQS    256
#define HTP_OP_MAX_TENSORS (HTP_OP_MAX_REQS * HTP_OP_MAX_INPUTS + HTP_OP_MAX_REQS)

#if __HVX_ARCH__ < 75
#define HTP_OP_MAX_VMEM    (3167538380u)
#else
#define HTP_OP_MAX_VMEM    (3221225472u)
#endif

#define HTP_MMAP_MAX_VMEM  (2147483648u)

enum htp_tensor_flags {
    HTP_TENSOR_COMPUTE = (1U << 0), // Tensor buffer temporal compute data (not weights)
    HTP_TENSOR_FLUSHED = (1U << 1), // Tensor buffer has been flushed (set by the NPU)
    HTP_TENSOR_WEIGHT_PRETILED = (1U << 2), // F16 weight is already in HMX tile-major layout
    HTP_TENSOR_HOST_DIRTY = (1U << 3) // Host rewrote persistent tensor data after DSP may have cached it
};

// Tensor descriptor
struct htp_tensor {
    uint32_t data;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t size;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint16_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
struct htp_buf_desc {
    uint64_t base;     // base address
    uint64_t size;     // total size
    uint32_t flags;    // buffer flags (unused)
    uint32_t fd;       // file descriptor
};

enum htp_op_flags {
    HTP_OPFLAGS_SKIP_COMPUTE  = (1U << 0), // Skip actual computation (used for profiling)
    HTP_OPFLAGS_LORA_DEBUG    = (1U << 1), // Print lightweight fused LoRA stats
    HTP_OPFLAGS_LORA_FAST     = (1U << 2), // Enable experimental hand-optimized LoRA low-rank kernels
    HTP_OPFLAGS_LORA_DEBUG_DEEP = (1U << 3), // Enable expensive full LoRA debug reductions
    HTP_OPFLAGS_LORA_ANTITHETIC = (1U << 4), // Select B_plus/B_minus per token row in fused LoRA
    HTP_OPFLAGS_LORA_PROFILE  = (1U << 5), // Return lightweight fused LoRA internal timing
    HTP_OPFLAGS_LORA_B_RM_F32 = (1U << 6), // LoRA B inputs are rank-major F32 [N, rank]
};

enum htp_op_debug_flags {
    HTP_OP_DEBUG_LORA_STATS = (1U << 0),
};

enum htp_lora_debug_stage {
    HTP_LORA_STAGE_NONE        = 0,
    HTP_LORA_STAGE_ENTER       = 1,
    HTP_LORA_STAGE_BAD_PARAMS  = 2,
    HTP_LORA_STAGE_UNSUPPORTED = 3,
    HTP_LORA_STAGE_MATMUL      = 4,
    HTP_LORA_STAGE_SKIP        = 5,
    HTP_LORA_STAGE_SCALE_ZERO  = 6,
    HTP_LORA_STAGE_OOM         = 7,
    HTP_LORA_STAGE_TMP_FAIL    = 8,
    HTP_LORA_STAGE_ACC_FAIL    = 9,
    HTP_LORA_STAGE_DONE        = 10,
    HTP_LORA_STAGE_TMP_MATMUL  = 11,
    HTP_LORA_STAGE_DELTA_MATMUL = 12,
    HTP_LORA_STAGE_SCALE_ADD   = 13,
};

// Op descriptor
struct htp_op_desc {
    uint32_t opcode;                    // GGML/HTP Op
    uint32_t flags;                     // Op flags
    int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
    uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
    uint16_t dst;                       // Output tensor index

    // the rest is filled in-place by the NPU
    uint32_t prof_usecs;                // Number of usec per request
    uint32_t prof_cycles;               // Number of cycles per request
    uint32_t prof_pkts;                 // Number of instruction packets per request
    uint32_t unused;
    float    lora_scale;                // Explicit GGML_OP_LORA_MUL_MAT scale; avoids implicit op_params ABI assumptions
    uint32_t lora_positive_rows;        // Token rows using B_plus; remaining rows use B_minus
    uint32_t lora_unused0;
    uint32_t debug_flags;
    uint32_t debug_k;
    uint32_t debug_r;
    uint32_t debug_m;
    uint32_t debug_n;
    float    debug_scale;
    float    debug_sum_abs_tmp;
    float    debug_sum_abs_b;
    float    debug_sum_abs_delta;
    uint32_t debug_status;
    uint32_t debug_stage;
    float    debug_sum_abs_b_minus;   // DSP-side sum_abs of src[4]=lora_b_minus_rm (deep mode only)
    uint32_t debug_unused2;
    uint32_t lora_prof_main_us;
    uint32_t lora_prof_tmp_us;
    uint32_t lora_prof_bprep_us;
    uint32_t lora_prof_acc_us;
};

struct htp_opbatch_req {
    uint32_t n_bufs;      // Number of buffers
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
    uint32_t flags;       // unused
    uint32_t abi_magic;
    uint32_t abi_version;
    uint32_t op_desc_size;
    uint32_t tensor_size;
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};

struct htp_opbatch_rsp {
    uint32_t status;     // HTP_STATUS_...
    uint32_t n_bufs;
    uint32_t n_tensors;
    uint32_t n_ops;
    uint32_t abi_magic;
    uint32_t abi_version;
    uint32_t op_desc_size;
    uint32_t tensor_size;
    // struct htp_op_req ops[];     -- dspqueue buf 0
};

#endif /* HTP_OPS_H */
