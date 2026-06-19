#include "llm/llm.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <half.hpp>
#include <rapidjson/document.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using MNN::Express::NCHW;
using MNN::Express::VARP;
using MNN::Express::VARPS;
using MNN::Express::_Input;
using MNN::Transformer::Llm;

struct Args {
    std::string config;
    std::string loraMeta;
    std::string trainData;
    std::string validData;
    int steps = 2000;
    int batch = 4;
    int seqLen = 64;
    int evalStep = 50;
    float epsilon = 1.0e-2f;
    float lr = 5.0e-5f;
    int threads = 4;
    int seed = 1337;
    std::string precision = "low";
    std::string runtimeMemory = "high";
    std::string bParamPrecision = "fp16";
    std::string bStorageLayout = "mnn";
    int attentionMode = 8;
    bool forceFloatWeight = false;
    bool compactNormalWeight = true;
    bool verbose = false;
    bool debugMNN = false;
    bool diagZoCheck = false;
    int diagRepeat = 0;  // if > 0, repeat identical forwards and exit
    bool diagBReadback = false;
    int baselineStep = 0;  // if > 0, evaluate unperturbed B every N steps
    int diagIsolateB = 0;  // if > 0, run the per-target B isolation diagnostic and exit
    float diagIsolateValue = 0.1f;  // value to set the isolated target's B elements to
    int diagPrefillDecode = 0;  // if > 0, compare full prefill with token-by-token decode and exit
    bool forceFullCausalMask = false;
    int diagLayerwise = 0;  // if > 0, dump layer-wise activations for N warmup samples and exit
    std::vector<int> diagLayerwiseLayers{0, 1, 21};
    int diagLayerwiseValues = 8;
    int diagLoraPerturb = 0;  // if > 0, run single-target nonzero LoRA-B perturb diagnostics and exit
    int diagLoraA = 0;  // if > 0, dump selected LoRA-A constants and x@A for N warmup samples and exit
    int diagLoraBranch = 0;  // if > 0, dump B=0/+eps/-eps LoRA branch internals and exit
    std::vector<std::string> diagLoraTargets{
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.ffn_gate.weight",
        "blk.21.ffn_up.weight",
    };
    int diagLoraValues = 8;
    std::string diagLoraPattern = "noise";
    bool diagLoraPropagation = false;
    bool diagAttnInternals = false;
    std::vector<int> diagAttnHeads{0};
    int diagAttnValues = 16;
};

struct Target {
    std::string inputName;
    std::string canonicalKey;
    std::string targetOp;
    std::string injectionPoint;
    std::string hiddenInputSource;
    std::string group;
    std::string module;
    std::vector<int> shape;
    int layer = -1;
    int hiddenInputTensor = -1;
    int inputSize = 0;
    int outputSize = 0;
    float scaleFoldedIntoA = 1.0f;
    size_t offset = 0;
    size_t size = 0;
};

struct Metadata {
    int rank = 8;
    int alpha = 8;
    int vocabSize = 32000;
    size_t totalBSize = 0;
    std::vector<Target> targets;
};

struct Sample {
    std::vector<int> tokens;
    int label = 0;
    int targetToken = 0;
};

struct MnnActivationDebugState {
    bool enabled = false;
    bool active = false;
    bool layerwise = false;
    bool loraBranch = false;
    int forwardId = 0;
    int dataIndex = -1;
    int sampleOrdinal = -1;
    int printedOps = 0;
    int maxOpsPerForward = 96;
    int maxTensorSlots = 3;
    int maxTensorValues = 6;
    int seqLen = -1;
    int finalPos = -1;
    int label = -1;
    int targetToken = -1;
    std::string tag;
    std::string activeCanonicalKey;
    std::string activeInjectionPoint;
    std::string activeTargetOp;
    std::string activeModule;
    std::string activeGroup;
    bool loraPropagation = false;
    int activeLayer = -1;
    std::vector<std::string> targetPrefixes;
    std::vector<int> layerwiseLayers;
    std::mutex mutex;
};

struct LossDebugState {
    bool enabled = false;
    bool printed = false;
    int sampleLimit = 1;
    int printedSamples = 0;
    bool printEmbedding = false;
    std::string tag;
    const std::vector<int>* classTokens = nullptr;
    Llm* llm = nullptr;
    MnnActivationDebugState* activationDebug = nullptr;
};

static bool fail(const std::string& message) {
    std::cerr << "zo_lora_fa_sst2 error: " << message << "\n";
    return false;
}

static bool readFile(const std::string& path, std::string* content) {
    std::ifstream is(path.c_str(), std::ios::binary);
    if (!is.is_open()) {
        return fail("failed to open " + path);
    }
    std::ostringstream os;
    os << is.rdbuf();
    *content = os.str();
    return true;
}

static std::string dirname(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

static std::string joinPath(const std::string& dir, const std::string& name) {
    if (name.empty() || (!name.empty() && (name[0] == '/' || name[0] == '\\'))) {
        return name;
    }
    if (dir.empty() || dir == ".") {
        return name;
    }
    char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') {
        return dir + name;
    }
    return dir + "/" + name;
}

static bool fileExists(const std::string& path) {
    std::ifstream is(path.c_str(), std::ios::binary);
    return is.good();
}

static void setProcessEnv(const std::string& name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

static void unsetProcessEnv(const std::string& name) {
#ifdef _WIN32
    _putenv_s(name.c_str(), "");
#else
    unsetenv(name.c_str());
#endif
}

static std::string joinIntList(const std::vector<int>& values) {
    std::ostringstream os;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << values[i];
    }
    return os.str();
}

static bool endsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool containsString(const std::string& value, const std::string& needle) {
    return value.find(needle) != std::string::npos;
}

static std::string trim(const std::string& value);

static std::string tensorShapeString(const MNN::Tensor* tensor) {
    if (tensor == nullptr) {
        return "[]";
    }
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < tensor->dimensions(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << tensor->length(i);
    }
    os << "]";
    return os.str();
}

static std::string tensorTypeString(const MNN::Tensor* tensor) {
    if (tensor == nullptr) {
        return "null";
    }
    const auto type = tensor->getType();
    std::ostringstream os;
    os << "code" << type.code << "_bits" << type.bits;
    return os.str();
}

static bool parseIntList(const std::string& text, std::vector<int>* values) {
    if (values == nullptr) {
        return false;
    }
    values->clear();
    std::string stripped = trim(text);
    if (stripped.empty() || stripped == "all" || stripped == "*") {
        return true;
    }
    std::stringstream ss(stripped);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            return false;
        }
        char* end = nullptr;
        long parsed = std::strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
            return false;
        }
        values->push_back(static_cast<int>(parsed));
    }
    return true;
}

static bool parseStringList(const std::string& text, std::vector<std::string>* values) {
    if (values == nullptr) {
        return false;
    }
    values->clear();
    std::string stripped = trim(text);
    if (stripped.empty() || stripped == "all" || stripped == "*") {
        return true;
    }
    std::stringstream ss(stripped);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            return false;
        }
        values->push_back(item);
    }
    return true;
}

static bool isLayerSelected(const MnnActivationDebugState& state, int layer) {
    return state.layerwiseLayers.empty() ||
           std::find(state.layerwiseLayers.begin(), state.layerwiseLayers.end(), layer) !=
               state.layerwiseLayers.end();
}

static bool parseLayerIndexWithMarker(const std::string& name, const std::string& marker, int* layer) {
    size_t pos = name.find(marker);
    if (pos == std::string::npos) {
        return false;
    }
    pos += marker.size();
    if (pos >= name.size() || !std::isdigit(static_cast<unsigned char>(name[pos]))) {
        return false;
    }
    int value = 0;
    while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
        value = value * 10 + (name[pos] - '0');
        ++pos;
    }
    *layer = value;
    return true;
}

static bool parseMnnLayerIndex(const std::string& name, int* layer) {
    return parseLayerIndexWithMarker(name, "/blocks.", layer) ||
           parseLayerIndexWithMarker(name, "/layers.", layer);
}

static bool matchMnnLayerwisePoint(const MnnActivationDebugState& state, const std::string& name,
                                   int* layer, std::string* point) {
    if (containsString(name, "lm_head") || containsString(name, "/lm/Linear")) {
        *layer = -1;
        *point = "lm_head";
        return true;
    }
    if (name == "hidden_states" || containsString(name, "/final_layernorm/") ||
        containsString(name, "/norm/Mul_1")) {
        *layer = -1;
        *point = "final_norm";
        return true;
    }

    int localLayer = -1;
    if (!parseMnnLayerIndex(name, &localLayer) || !isLayerSelected(state, localLayer)) {
        return false;
    }

    std::string localPoint;
    if (containsString(name, "/input_layernorm/") && containsString(name, "Mul_1")) {
        localPoint = "attn_norm";
    } else if (containsString(name, "/self_attn/q_proj/") &&
               (containsString(name, "FakeLinear") || containsString(name, "Linear/post_reshape") ||
                containsString(name, "Linear/post_convert"))) {
        localPoint = "q_raw";
    } else if (containsString(name, "/self_attn/k_proj/") &&
               (containsString(name, "FakeLinear") || containsString(name, "Linear/post_reshape") ||
                containsString(name, "Linear/post_convert"))) {
        localPoint = "k_raw";
    } else if (containsString(name, "/self_attn/v_proj/") &&
               (containsString(name, "FakeLinear") || containsString(name, "Linear/post_reshape") ||
                containsString(name, "Linear/post_convert"))) {
        localPoint = "v_raw";
    } else if (containsString(name, "/self_attn/Add_1")) {
        localPoint = "k_rope";
    } else if (containsString(name, "/self_attn/Add")) {
        localPoint = "q_rope";
    } else if (containsString(name, "FusedAttention") || containsString(name, "/self_attn/attn_context")) {
        localPoint = "attn_context";
    } else if (containsString(name, "/self_attn/o_proj/") &&
               (containsString(name, "FakeLinear") || containsString(name, "Linear/post_reshape") ||
                containsString(name, "Linear/post_convert"))) {
        localPoint = "o_proj";
    } else if ((containsString(name, "/blocks.") || containsString(name, "/layers.")) &&
               containsString(name, "/Add") && !containsString(name, "/self_attn/") &&
               !containsString(name, "/mlp/") && !containsString(name, "Add_1")) {
        localPoint = "attn_resid";
    } else if (containsString(name, "/post_attention_layernorm/") && containsString(name, "Mul_1")) {
        localPoint = "ffn_norm";
    } else if (containsString(name, "/mlp/gate_proj/") &&
               (containsString(name, "FakeLinear") || containsString(name, "Linear/post_reshape") ||
                containsString(name, "Linear/post_convert"))) {
        localPoint = "gate";
    } else if (containsString(name, "/mlp/up_proj/") &&
               (containsString(name, "FakeLinear") || containsString(name, "Linear/post_reshape") ||
                containsString(name, "Linear/post_convert"))) {
        localPoint = "up";
    } else if (containsString(name, "/mlp/act_fn/") && containsString(name, "Mul")) {
        localPoint = "act";
    } else if (containsString(name, "/mlp/Mul")) {
        localPoint = "mlp_mul";
    } else if (containsString(name, "/mlp/down_proj/") &&
               (containsString(name, "FakeLinear") || containsString(name, "Linear/post_reshape") ||
                containsString(name, "Linear/post_convert"))) {
        localPoint = "down";
    } else if ((containsString(name, "/blocks.") || containsString(name, "/layers.")) &&
               containsString(name, "/Add_1") && !containsString(name, "/self_attn/") &&
               !containsString(name, "/mlp/")) {
        localPoint = "layer_out";
    }

    if (localPoint.empty()) {
        return false;
    }
    *layer = localLayer;
    *point = localPoint;
    return true;
}

static int chooseMnnTokenDim(const std::vector<int>& dims, int seqLen) {
    if (seqLen <= 0) {
        return -1;
    }
    if (dims.size() == 4 && dims[0] == seqLen && dims[2] == 1 && dims[3] == 1) {
        return 0;
    }
    const int preferred[] = {1, 2, 0, 3};
    for (int dim : preferred) {
        if (dim >= 0 && dim < static_cast<int>(dims.size()) && dims[dim] == seqLen) {
            return dim;
        }
    }
    for (int i = 0; i < static_cast<int>(dims.size()); ++i) {
        if (dims[i] == seqLen) {
            return i;
        }
    }
    return -1;
}

static std::vector<int64_t> rowMajorStrides(const std::vector<int>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    int64_t stride = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= std::max(dims[i], 1);
    }
    return strides;
}

struct TensorValueStats {
    int64_t count = 0;
    double sumAbs = 0.0;
    double sumSq = 0.0;
    float maxAbs = 0.0f;
    float minValue = 0.0f;
    float maxValue = 0.0f;
    std::vector<float> firstValues;
};

static void updateTensorValueStats(TensorValueStats* stats, float value, int maxValues) {
    if (stats->count == 0) {
        stats->minValue = value;
        stats->maxValue = value;
    } else {
        stats->minValue = std::min(stats->minValue, value);
        stats->maxValue = std::max(stats->maxValue, value);
    }
    const float absValue = std::fabs(value);
    stats->sumAbs += static_cast<double>(absValue);
    stats->sumSq += static_cast<double>(value) * static_cast<double>(value);
    stats->maxAbs = std::max(stats->maxAbs, absValue);
    if (static_cast<int>(stats->firstValues.size()) < maxValues) {
        stats->firstValues.push_back(value);
    }
    stats->count++;
}

static void printStatsFields(const char* prefix, const TensorValueStats& stats) {
    if (stats.count <= 0) {
        std::cerr << " " << prefix << "_count=0"
                  << " " << prefix << "_mean_abs=nan"
                  << " " << prefix << "_rms=nan"
                  << " " << prefix << "_max_abs=nan"
                  << " " << prefix << "_min=nan"
                  << " " << prefix << "_max=nan";
        return;
    }
    const double denom = static_cast<double>(stats.count);
    std::cerr << " " << prefix << "_count=" << stats.count
              << " " << prefix << "_mean_abs=" << (stats.sumAbs / denom)
              << " " << prefix << "_rms=" << std::sqrt(stats.sumSq / denom)
              << " " << prefix << "_max_abs=" << stats.maxAbs
              << " " << prefix << "_min=" << stats.minValue
              << " " << prefix << "_max=" << stats.maxValue;
}

static void printFirstValuesField(const char* name, const std::vector<float>& values) {
    std::cerr << " " << name << "=[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            std::cerr << ",";
        }
        std::cerr << values[i];
    }
    std::cerr << "]";
}

static bool isAttentionPropagationTarget(const MnnActivationDebugState& state) {
    return state.activeLayer >= 0 && state.activeGroup == "self_attn" &&
           (state.activeModule == "q_proj" || state.activeModule == "k_proj" ||
            state.activeModule == "v_proj");
}

static std::string activeModulePointPrefix(const MnnActivationDebugState& state) {
    if (state.activeModule == "q_proj") {
        return "q";
    }
    if (state.activeModule == "k_proj") {
        return "k";
    }
    if (state.activeModule == "v_proj") {
        return "v";
    }
    if (state.activeModule == "o_proj") {
        return "o";
    }
    if (state.activeModule == "gate_proj") {
        return "gate";
    }
    if (state.activeModule == "up_proj") {
        return "up";
    }
    if (state.activeModule == "down_proj") {
        return "down";
    }
    return "";
}

static bool isMnnLoraPropagationOp(const MnnActivationDebugState& state, const std::string& name) {
    if (!state.loraPropagation || !isAttentionPropagationTarget(state)) {
        return false;
    }
    const std::string layer = std::to_string(state.activeLayer);
    const std::string blockPrefix = "/blocks." + layer + "/self_attn/";
    const std::string layerPrefix = "/layers." + layer + "/self_attn/";
    return name == blockPrefix + "Reshape_output_0" ||
           name == blockPrefix + "Add_output_0" ||
           name == blockPrefix + "Reshape_1_output_0" ||
           name == blockPrefix + "Add_1_output_0" ||
           name == blockPrefix + "Reshape_2_output_0" ||
           name == layerPrefix + "FusedAttention";
}

static bool shouldDumpMnnActivationOp(const MnnActivationDebugState& state, const std::string& name) {
    if (state.loraBranch && !state.activeTargetOp.empty()) {
        if (name == state.activeTargetOp + "/post_convert" ||
            name == state.activeTargetOp + "/post_reshape") {
            return true;
        }
    }
    if (state.loraBranch && isMnnLoraPropagationOp(state, name)) {
        return true;
    }
    if (name.find("/lora_fa/") == std::string::npos) {
        return false;
    }
    if (!endsWith(name, "/reshape_in") &&
        !endsWith(name, "/matmul_A") &&
        !endsWith(name, "/matmul_B") &&
        !endsWith(name, "/reshape_out") &&
        !endsWith(name, "/pack_nc4hw4") &&
        !endsWith(name, "/add")) {
        return false;
    }
    if (state.targetPrefixes.empty()) {
        return true;
    }
    for (const auto& prefix : state.targetPrefixes) {
        if (name.find(prefix) == 0) {
            return true;
        }
    }
    return false;
}

static void printMnnTensorActivationStats(const MnnActivationDebugState& state,
                                          const std::string& phase,
                                          const MNN::OperatorInfo* info,
                                          const MNN::Tensor* tensor,
                                          int tensorIndex) {
    std::cerr << "mnn_activation_debug"
              << " tag=" << state.tag
              << " forward_id=" << state.forwardId
              << " data_index=" << state.dataIndex
              << " sample_ordinal=" << state.sampleOrdinal
              << " phase=" << phase
              << " op_name=\"" << (info != nullptr ? info->name() : std::string("<null>")) << "\""
              << " op_type=\"" << (info != nullptr ? info->type() : std::string("<null>")) << "\""
              << " tensor_index=" << tensorIndex;
    if (tensor == nullptr) {
        std::cerr << " tensor=null\n";
        return;
    }

    std::unique_ptr<MNN::Tensor> hostTensor(MNN::Tensor::createHostTensorFromDevice(tensor, true));
    const MNN::Tensor* readable = hostTensor ? hostTensor.get() : tensor;
    const auto type = readable->getType();
    const int elementCount = readable->elementSize();
    std::vector<int> dims;
    dims.reserve(readable->dimensions());
    for (int i = 0; i < readable->dimensions(); ++i) {
        dims.push_back(readable->length(i));
    }
    std::cerr << " shape=" << tensorShapeString(readable)
              << " dtype=" << tensorTypeString(readable)
              << " elements=" << elementCount;

    if (type.code != halide_type_float || (type.bits != 32 && type.bits != 16)) {
        std::cerr << " stats=unsupported_dtype\n";
        return;
    }
    if (elementCount <= 0) {
        std::cerr << " stats=empty\n";
        return;
    }

    const float* ptr32 = nullptr;
    const half_float::half* ptr16 = nullptr;
    if (type.bits == 32) {
        ptr32 = readable->host<float>();
        if (ptr32 == nullptr) {
            std::cerr << " stats=null_host\n";
            return;
        }
    } else {
        ptr16 = readable->host<half_float::half>();
        if (ptr16 == nullptr) {
            std::cerr << " stats=null_host\n";
            return;
        }
    }

    auto readValue = [&](int index) -> float {
        return type.bits == 32 ? ptr32[index] : static_cast<float>(ptr16[index]);
    };

    TensorValueStats fullStats;
    TensorValueStats vecStats;
    const int selectedDim = chooseMnnTokenDim(dims, state.seqLen);
    int selectedToken = selectedDim >= 0 ? state.finalPos : -1;
    if (selectedDim >= 0 && (selectedToken < 0 || selectedToken >= dims[selectedDim])) {
        selectedToken = 0;
    }
    const std::vector<int64_t> strides = rowMajorStrides(dims);
    for (int i = 0; i < elementCount; ++i) {
        const float value = readValue(i);
        updateTensorValueStats(&fullStats, value, state.maxTensorValues);
        bool inVector = selectedDim < 0;
        if (selectedDim >= 0 && selectedDim < static_cast<int>(strides.size()) && strides[selectedDim] > 0) {
            const int coord = static_cast<int>((static_cast<int64_t>(i) / strides[selectedDim]) %
                                               std::max(dims[selectedDim], 1));
            inVector = coord == selectedToken;
        }
        if (inVector) {
            updateTensorValueStats(&vecStats, value, state.maxTensorValues);
        }
    }
    std::cerr << " selected_dim=" << selectedDim
              << " selected_token=" << selectedToken;
    printStatsFields("full", fullStats);
    printStatsFields("vec", vecStats);
    printFirstValuesField("vec_first", vecStats.firstValues);
    std::cerr << "\n";
}

static void printMnnLayerwiseActivationStats(const MnnActivationDebugState& state,
                                             const MNN::OperatorInfo* info,
                                             const MNN::Tensor* tensor,
                                             int tensorIndex,
                                             int layer,
                                             const std::string& point) {
    std::cerr << "activation_debug"
              << " framework=mnn"
              << " tag=" << state.tag
              << " forward_id=" << state.forwardId
              << " sample_ordinal=" << state.sampleOrdinal
              << " data_index=" << state.dataIndex
              << " label=" << state.label
              << " target=" << state.targetToken
              << " final_pos=" << state.finalPos
              << " layer=" << layer
              << " point=" << point
              << " name=\"" << (info != nullptr ? info->name() : std::string("<null>")) << "\""
              << " tensor_index=" << tensorIndex;
    if (tensor == nullptr) {
        std::cerr << " tensor=null\n";
        return;
    }

    std::unique_ptr<MNN::Tensor> hostTensor(MNN::Tensor::createHostTensorFromDevice(tensor, true));
    const MNN::Tensor* readable = hostTensor ? hostTensor.get() : tensor;
    const auto type = readable->getType();
    const int elementCount = readable->elementSize();
    std::vector<int> dims;
    dims.reserve(readable->dimensions());
    for (int i = 0; i < readable->dimensions(); ++i) {
        dims.push_back(readable->length(i));
    }
    std::cerr << " shape=" << tensorShapeString(readable)
              << " dtype=" << tensorTypeString(readable)
              << " elements=" << elementCount
              << " seq_len=" << state.seqLen;

    if (type.code != halide_type_float || (type.bits != 32 && type.bits != 16)) {
        std::cerr << " stats=unsupported_dtype\n";
        return;
    }
    if (elementCount <= 0) {
        std::cerr << " stats=empty\n";
        return;
    }

    const float* ptr32 = nullptr;
    const half_float::half* ptr16 = nullptr;
    if (type.bits == 32) {
        ptr32 = readable->host<float>();
        if (ptr32 == nullptr) {
            std::cerr << " stats=null_host\n";
            return;
        }
    } else {
        ptr16 = readable->host<half_float::half>();
        if (ptr16 == nullptr) {
            std::cerr << " stats=null_host\n";
            return;
        }
    }

    auto readValue = [&](int index) -> float {
        return type.bits == 32 ? ptr32[index] : static_cast<float>(ptr16[index]);
    };

    TensorValueStats fullStats;
    TensorValueStats vecStats;
    const int selectedDim = chooseMnnTokenDim(dims, state.seqLen);
    int selectedToken = selectedDim >= 0 ? state.finalPos : -1;
    if (selectedDim >= 0 && (selectedToken < 0 || selectedToken >= dims[selectedDim])) {
        selectedToken = 0;
    }
    const std::vector<int64_t> strides = rowMajorStrides(dims);
    for (int i = 0; i < elementCount; ++i) {
        const float value = readValue(i);
        updateTensorValueStats(&fullStats, value, state.maxTensorValues);
        bool inVector = selectedDim < 0;
        if (selectedDim >= 0 && selectedDim < static_cast<int>(strides.size()) && strides[selectedDim] > 0) {
            const int coord = static_cast<int>((static_cast<int64_t>(i) / strides[selectedDim]) %
                                               std::max(dims[selectedDim], 1));
            inVector = coord == selectedToken;
        }
        if (inVector) {
            updateTensorValueStats(&vecStats, value, state.maxTensorValues);
        }
    }

    std::cerr << " selected_dim=" << selectedDim
              << " selected_token=" << selectedToken;
    printStatsFields("full", fullStats);
    printStatsFields("vec", vecStats);
    printFirstValuesField("vec_first", vecStats.firstValues);
    std::cerr << "\n";
}

static bool matchMnnLoraBranchPoint(const MnnActivationDebugState& state,
                                    const std::string& name, const std::string& phase,
                                    int tensorIndex, std::string* point) {
    if (phase == "before") {
        if (state.loraPropagation && isAttentionPropagationTarget(state) &&
            name == "/layers." + std::to_string(state.activeLayer) + "/self_attn/FusedAttention") {
            if (tensorIndex == 0) {
                *point = "attn_q_input";
                return true;
            }
            if (tensorIndex == 1) {
                *point = "attn_k_input";
                return true;
            }
            if (tensorIndex == 2) {
                *point = "attn_v_input";
                return true;
            }
        }
        if (endsWith(name, "/reshape_in") && tensorIndex == 0) {
            *point = "reshape_in_input";
            return true;
        }
        if (endsWith(name, "/matmul_A")) {
            if (tensorIndex == 0) {
                *point = "reshape_in";
                return true;
            }
            if (tensorIndex == 1) {
                *point = "A_const_scaled";
                return true;
            }
        }
        if (endsWith(name, "/matmul_B") && tensorIndex == 1) {
            *point = "B";
            return true;
        }
        if (endsWith(name, "/add")) {
            if (tensorIndex == 0) {
                *point = "add_base";
                return true;
            }
            if (tensorIndex == 1) {
                *point = "add_delta";
                return true;
            }
        }
        if (endsWith(name, "/pack_nc4hw4") && tensorIndex == 0) {
            *point = "reshape_out_before_pack";
            return true;
        }
    }
    if (phase == "after") {
        if (endsWith(name, "/post_convert") && tensorIndex == 0) {
            const std::string prefix = activeModulePointPrefix(state);
            *point = prefix.empty() ? "post_convert_after_lora" : prefix + "_post_convert";
            return true;
        }
        if (endsWith(name, "/post_reshape") && tensorIndex == 0) {
            const std::string prefix = activeModulePointPrefix(state);
            *point = prefix.empty() ? "post_reshape_after_lora" : prefix + "_post_reshape";
            return true;
        }
        if (state.loraPropagation && isAttentionPropagationTarget(state) && tensorIndex == 0) {
            const std::string layer = std::to_string(state.activeLayer);
            const std::string blockPrefix = "/blocks." + layer + "/self_attn/";
            const std::string layerPrefix = "/layers." + layer + "/self_attn/";
            if (name == blockPrefix + "Reshape_output_0") {
                *point = "q_block_reshape";
                return true;
            }
            if (name == blockPrefix + "Add_output_0") {
                *point = "q_rope";
                return true;
            }
            if (name == blockPrefix + "Reshape_1_output_0") {
                *point = "k_block_reshape";
                return true;
            }
            if (name == blockPrefix + "Add_1_output_0") {
                *point = "k_rope";
                return true;
            }
            if (name == blockPrefix + "Reshape_2_output_0") {
                *point = "v_block_reshape";
                return true;
            }
            if (name == layerPrefix + "FusedAttention") {
                *point = "attn_context";
                return true;
            }
        }
        if (endsWith(name, "/reshape_in") && tensorIndex == 0) {
            *point = "reshape_in";
            return true;
        }
        if (endsWith(name, "/matmul_A") && tensorIndex == 0) {
            *point = "matmul_A_scaled";
            return true;
        }
        if (endsWith(name, "/matmul_B") && tensorIndex == 0) {
            *point = "matmul_B_scaled";
            return true;
        }
        if (endsWith(name, "/reshape_out") && tensorIndex == 0) {
            *point = "reshape_out";
            return true;
        }
        if (endsWith(name, "/pack_nc4hw4") && tensorIndex == 0) {
            *point = "delta_pack_nc4hw4";
            return true;
        }
        if (endsWith(name, "/add") && tensorIndex == 0) {
            *point = "add_out";
            return true;
        }
    }
    return false;
}

static void printMnnLoraBranchTensorStats(const MnnActivationDebugState& state,
                                          const std::string& phase,
                                          const MNN::OperatorInfo* info,
                                          const MNN::Tensor* tensor,
                                          int tensorIndex,
                                          const std::string& point) {
    std::cerr << "diag_lora_branch_debug"
              << " framework=mnn"
              << " tag=" << state.tag
              << " forward_id=" << state.forwardId
              << " sample_ordinal=" << state.sampleOrdinal
              << " data_index=" << state.dataIndex
              << " label=" << state.label
              << " target=" << state.targetToken
              << " final_pos=" << state.finalPos
              << " canonical_key=" << (state.activeCanonicalKey.empty() ? "<missing>" : state.activeCanonicalKey)
              << " point=" << point
              << " phase=" << phase
              << " injection_point=" << (state.activeInjectionPoint.empty() ? "<missing>" : state.activeInjectionPoint)
              << " layout_note="
              << ((state.activeInjectionPoint == "pre_post_convert_nc4hw4" &&
                   (point == "add_base" || point == "add_delta" || point == "add_out" ||
                    point == "delta_pack_nc4hw4")) ? "nc4hw4_physical" : "logical_or_host")
              << " op_name=\"" << (info != nullptr ? info->name() : std::string("<null>")) << "\""
              << " tensor_index=" << tensorIndex;
    if (tensor == nullptr) {
        std::cerr << " tensor=null\n";
        return;
    }

    std::unique_ptr<MNN::Tensor> hostTensor(MNN::Tensor::createHostTensorFromDevice(tensor, true));
    const MNN::Tensor* readable = hostTensor ? hostTensor.get() : tensor;
    const auto type = readable->getType();
    const int elementCount = readable->elementSize();
    std::vector<int> dims;
    dims.reserve(readable->dimensions());
    for (int i = 0; i < readable->dimensions(); ++i) {
        dims.push_back(readable->length(i));
    }
    std::cerr << " shape=" << tensorShapeString(readable)
              << " dtype=" << tensorTypeString(readable)
              << " elements=" << elementCount
              << " seq_len=" << state.seqLen;

    if (type.code != halide_type_float || (type.bits != 32 && type.bits != 16)) {
        std::cerr << " stats=unsupported_dtype\n";
        return;
    }
    if (elementCount <= 0) {
        std::cerr << " stats=empty\n";
        return;
    }

    const float* ptr32 = nullptr;
    const half_float::half* ptr16 = nullptr;
    if (type.bits == 32) {
        ptr32 = readable->host<float>();
        if (ptr32 == nullptr) {
            std::cerr << " stats=null_host\n";
            return;
        }
    } else {
        ptr16 = readable->host<half_float::half>();
        if (ptr16 == nullptr) {
            std::cerr << " stats=null_host\n";
            return;
        }
    }

    auto readValue = [&](int index) -> float {
        return type.bits == 32 ? ptr32[index] : static_cast<float>(ptr16[index]);
    };

    TensorValueStats fullStats;
    TensorValueStats vecStats;
    const int selectedDim = chooseMnnTokenDim(dims, state.seqLen);
    int selectedToken = selectedDim >= 0 ? state.finalPos : -1;
    if (selectedDim >= 0 && (selectedToken < 0 || selectedToken >= dims[selectedDim])) {
        selectedToken = 0;
    }
    const std::vector<int64_t> strides = rowMajorStrides(dims);
    for (int i = 0; i < elementCount; ++i) {
        const float value = readValue(i);
        updateTensorValueStats(&fullStats, value, state.maxTensorValues);
        bool inVector = selectedDim < 0;
        if (selectedDim >= 0 && selectedDim < static_cast<int>(strides.size()) && strides[selectedDim] > 0) {
            const int coord = static_cast<int>((static_cast<int64_t>(i) / strides[selectedDim]) %
                                               std::max(dims[selectedDim], 1));
            inVector = coord == selectedToken;
        }
        if (inVector) {
            updateTensorValueStats(&vecStats, value, state.maxTensorValues);
        }
    }

    std::cerr << " selected_dim=" << selectedDim
              << " selected_token=" << selectedToken;
    printStatsFields("full", fullStats);
    printStatsFields("vec", vecStats);
    printFirstValuesField("vec_first", vecStats.firstValues);
    std::cerr << "\n";
}

static bool dumpMnnActivationCallback(MnnActivationDebugState* state,
                                      const std::vector<MNN::Tensor*>& tensors,
                                      const MNN::OperatorInfo* info,
                                      const char* phase) {
    if (state == nullptr || !state->enabled || info == nullptr) {
        return true;
    }
    std::lock_guard<std::mutex> guard(state->mutex);
    if (!state->active || state->printedOps >= state->maxOpsPerForward) {
        return true;
    }
    if (state->layerwise) {
        if (phase == nullptr || std::string(phase) != "after") {
            return true;
        }
        int layer = -1;
        std::string point;
        if (!matchMnnLayerwisePoint(*state, info->name(), &layer, &point)) {
            return true;
        }
        ++state->printedOps;
        const int n = std::min<int>(static_cast<int>(tensors.size()), 1);
        for (int i = 0; i < n; ++i) {
            printMnnLayerwiseActivationStats(*state, info, tensors[i], i, layer, point);
        }
        return true;
    }
    if (!shouldDumpMnnActivationOp(*state, info->name())) {
        return true;
    }
    if (state->loraBranch) {
        const std::string phaseString = phase != nullptr ? phase : "<null>";
        for (int i = 0; i < static_cast<int>(tensors.size()); ++i) {
            std::string point;
            if (!matchMnnLoraBranchPoint(*state, info->name(), phaseString, i, &point)) {
                continue;
            }
            ++state->printedOps;
            printMnnLoraBranchTensorStats(*state, phaseString, info, tensors[i], i, point);
            if (state->printedOps >= state->maxOpsPerForward) {
                break;
            }
        }
        return true;
    }
    ++state->printedOps;
    std::cerr << "mnn_activation_debug_op"
              << " tag=" << state->tag
              << " forward_id=" << state->forwardId
              << " data_index=" << state->dataIndex
              << " sample_ordinal=" << state->sampleOrdinal
              << " phase=" << (phase != nullptr ? phase : "<null>")
              << " op_name=\"" << info->name() << "\""
              << " op_type=\"" << info->type() << "\""
              << " tensor_count=" << tensors.size()
              << "\n";
    const int n = std::min<int>(static_cast<int>(tensors.size()), state->maxTensorSlots);
    for (int i = 0; i < n; ++i) {
        printMnnTensorActivationStats(*state, phase != nullptr ? phase : "<null>", info, tensors[i], i);
    }
    return true;
}

static void beginMnnActivationForward(MnnActivationDebugState* state, const std::string& tag,
                                      int dataIndex, int sampleOrdinal,
                                      int seqLen = -1, int finalPos = -1,
                                      int label = -1, int targetToken = -1) {
    if (state == nullptr || !state->enabled) {
        return;
    }
    std::lock_guard<std::mutex> guard(state->mutex);
    state->active = true;
    state->tag = tag;
    state->dataIndex = dataIndex;
    state->sampleOrdinal = sampleOrdinal;
    state->printedOps = 0;
    state->seqLen = seqLen;
    state->finalPos = finalPos;
    state->label = label;
    state->targetToken = targetToken;
    state->forwardId++;
    if (state->layerwise) {
        std::cerr << "activation_debug_begin"
                  << " framework=mnn"
                  << " tag=" << state->tag
                  << " forward_id=" << state->forwardId
                  << " data_index=" << state->dataIndex
                  << " sample_ordinal=" << state->sampleOrdinal
                  << " seq_len=" << state->seqLen
                  << " final_pos=" << state->finalPos
                  << " label=" << state->label
                  << " target=" << state->targetToken
                  << "\n";
    } else if (state->loraBranch) {
        std::cerr << "diag_lora_branch_debug_begin"
                  << " framework=mnn"
                  << " tag=" << state->tag
                  << " forward_id=" << state->forwardId
                  << " data_index=" << state->dataIndex
                  << " sample_ordinal=" << state->sampleOrdinal
                  << " canonical_key=" << (state->activeCanonicalKey.empty() ? "<missing>" : state->activeCanonicalKey)
                  << " active_layer=" << state->activeLayer
                  << " active_group=" << (state->activeGroup.empty() ? "<missing>" : state->activeGroup)
                  << " active_module=" << (state->activeModule.empty() ? "<missing>" : state->activeModule)
                  << " propagation=" << (state->loraPropagation ? "true" : "false")
                  << " seq_len=" << state->seqLen
                  << " final_pos=" << state->finalPos
                  << " label=" << state->label
                  << " target=" << state->targetToken
                  << "\n";
    } else {
        std::cerr << "mnn_activation_debug_begin"
                  << " tag=" << state->tag
                  << " forward_id=" << state->forwardId
                  << " data_index=" << state->dataIndex
                  << " sample_ordinal=" << state->sampleOrdinal
                  << " target_prefixes=" << state->targetPrefixes.size()
                  << "\n";
    }
}

static void endMnnActivationForward(MnnActivationDebugState* state) {
    if (state == nullptr || !state->enabled) {
        return;
    }
    std::lock_guard<std::mutex> guard(state->mutex);
    if (state->layerwise) {
        std::cerr << "activation_debug_end"
                  << " framework=mnn"
                  << " tag=" << state->tag
                  << " forward_id=" << state->forwardId
                  << " printed_ops=" << state->printedOps
                  << "\n";
    } else if (state->loraBranch) {
        std::cerr << "diag_lora_branch_debug_end"
                  << " framework=mnn"
                  << " tag=" << state->tag
                  << " forward_id=" << state->forwardId
                  << " printed_ops=" << state->printedOps
                  << "\n";
    } else {
        std::cerr << "mnn_activation_debug_end"
                  << " tag=" << state->tag
                  << " forward_id=" << state->forwardId
                  << " printed_ops=" << state->printedOps
                  << "\n";
    }
    state->active = false;
}

static std::string trim(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

static bool parseBoolValue(const std::string& value, bool* out) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower == "true" || lower == "1" || lower == "yes" || lower == "on") {
        *out = true;
        return true;
    }
    if (lower == "false" || lower == "0" || lower == "no" || lower == "off") {
        *out = false;
        return true;
    }
    return false;
}

static void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --config config_lora_fa.json --train-data sst2_train.tsv [--valid-data sst2_valid.tsv] [--lora-meta lora_fa.json]\n"
              << "  [--steps 2000] [--batch 4] [--seq-len 64] [--eval-step 50]\n"
              << "  [--epsilon 1e-2] [--lr 5e-5] [--threads 4, <=0 auto] [--seed 1337]\n"
              << "  [--precision low|normal|high] [--memory low|normal|high] [--b-param-precision fp16|fp32]\n"
              << "  [--b-storage-layout mnn|ggml]\n"
              << "  [--attention-mode 8] [--force-float-weight true|false]\n"
              << "  [--compact-normal-weight true|false]\n"
              << "  [--verbose] [--debug_MNN true|false] [--diag-zo-check]\n"
              << "  [--diag-repeat N] [--diag-b-readback] [--baseline-step N]\n"
              << "  [--diag-isolate-b N] [--diag-isolate-value 0.1]\n"
              << "  [--diag-prefill-decode N] [--force-full-causal-mask true|false]\n"
              << "  [--diag-layerwise N] [--diag-layerwise-layers 0,1,21] [--diag-layerwise-values 8]\n"
              << "  [--diag-lora-a N] [--diag-lora-branch N]\n"
              << "  [--diag-lora-perturb N] [--diag-lora-targets blk.0.attn_q.weight,...]\n"
              << "  [--diag-lora-values 8] [--diag-lora-pattern noise|ramp|constant]\n"
              << "  [--diag-lora-propagation true|false]\n"
              << "  [--diag-attn-internals true|false] [--diag-attn-heads 0,1,31] [--diag-attn-values 16]\n"
              << "    --debug_MNN true prints comparable token, embedding, and logits diagnostics\n"
              << "    --diag-zo-check repeats B0/+eps/-eps forwards and compares one\n"
              << "    ZO update with the opposite update, then exits\n"
              << "    --diag-repeat N repeats identical B/batch forwards to expose\n"
              << "    nondeterminism or stale cached inputs, then exits\n"
              << "    --diag-b-readback prints B VARP readback summaries after sync\n"
              << "    --b-storage-layout ggml interprets master B as llama.cpp flat order\n"
              << "    (out * rank + r) and transposes into MNN's [rank,out] input\n"
              << "    --baseline-step N logs the unperturbed post-update loss every N steps\n"
              << "    when --diag-isolate-b N>0, run a single-target B perturbation\n"
              << "    diagnostic for N evenly-spaced target indices, then exit\n"
              << "    --diag-prefill-decode N compares full prefill logits with\n"
              << "    token-by-token decode/KV logits for N warmup samples, then exits\n"
              << "    --diag-layerwise N dumps selected layer activations for N warmup samples, then exits\n"
              << "    --diag-lora-a N dumps selected LoRA-A constants and x@A for N warmup samples, then exits\n"
              << "    --diag-lora-branch N dumps B=0/+eps/-eps LoRA branch internals for N warmup samples, then exits\n"
              << "    --diag-lora-propagation true extends --diag-lora-branch to dump post-LoRA attention propagation\n"
              << "    --diag-attn-internals true extends --diag-lora-branch to dump QK/softmax/V attention internals\n"
              << "    --diag-lora-perturb N runs B=0/+eps/-eps single-target LoRA-B diagnostics, then exits\n"
              << "    --force-full-causal-mask true disables the CPU scalar causal\n"
              << "    mask fast path and materializes the full lower-triangular mask\n"
              << "    --force-float-weight true expands quantized CPU weights to float\n"
              << "    at load time, avoiding direct quant-weight GEMM for slower runs\n"
              << "    --compact-normal-weight true keeps Q4 normal-memory CPU weights\n"
              << "    packed while preserving the normal quant GEMM executor path\n"
              << "    --memory low|normal lets quantized CPU weights stay packed;\n"
              << "    --memory high expands more weights unless --force-float-weight changes it\n";
}

static bool parseArgs(int argc, const char* argv[], Args* args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        std::string inlineValue;
        auto eq = key.find('=');
        if (eq != std::string::npos && key.rfind("--", 0) == 0) {
            inlineValue = key.substr(eq + 1);
            key = key.substr(0, eq);
        }
        bool needValue = key != "--verbose" && key != "--diag-zo-check" &&
                         key != "--diag-b-readback" && key != "--help" && key != "-h";
        const char* value = nullptr;
        if (needValue && key.size() > 0 && key[0] == '-') {
            if (!inlineValue.empty()) {
                value = inlineValue.c_str();
            } else if (i + 1 >= argc) {
                return fail("missing value for " + key);
            } else {
                value = argv[++i];
            }
        }
        if (key == "--config") {
            args->config = value;
        } else if (key == "--lora-meta") {
            args->loraMeta = value;
        } else if (key == "--train-data") {
            args->trainData = value;
        } else if (key == "--valid-data" || key == "--eval-data") {
            args->validData = value;
        } else if (key == "--steps") {
            args->steps = std::atoi(value);
        } else if (key == "--batch" || key == "--batch-size") {
            args->batch = std::atoi(value);
        } else if (key == "--seq-len" || key == "--max-length") {
            args->seqLen = std::atoi(value);
        } else if (key == "--eval-step" || key == "--eval-interval") {
            args->evalStep = std::atoi(value);
        } else if (key == "--epsilon") {
            args->epsilon = static_cast<float>(std::atof(value));
        } else if (key == "--lr") {
            args->lr = static_cast<float>(std::atof(value));
        } else if (key == "--threads") {
            args->threads = std::atoi(value);
        } else if (key == "--seed") {
            args->seed = std::atoi(value);
        } else if (key == "--precision") {
            args->precision = value;
        } else if (key == "--memory" || key == "--runtime-memory") {
            args->runtimeMemory = value;
            std::transform(args->runtimeMemory.begin(), args->runtimeMemory.end(),
                           args->runtimeMemory.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (args->runtimeMemory != "low" && args->runtimeMemory != "normal" && args->runtimeMemory != "high") {
                return fail("--memory must be low, normal, or high");
            }
        } else if (key == "--b-param-precision") {
            args->bParamPrecision = value;
        } else if (key == "--b-storage-layout" || key == "--b-layout") {
            args->bStorageLayout = value;
            std::transform(args->bStorageLayout.begin(), args->bStorageLayout.end(),
                           args->bStorageLayout.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            if (args->bStorageLayout != "mnn" && args->bStorageLayout != "ggml") {
                return fail("--b-storage-layout must be mnn or ggml");
            }
        } else if (key == "--attention-mode") {
            args->attentionMode = std::atoi(value);
        } else if (key == "--force-float-weight") {
            if (value == nullptr || !parseBoolValue(value, &args->forceFloatWeight)) {
                return fail("--force-float-weight must be true or false");
            }
        } else if (key == "--compact-normal-weight") {
            if (value == nullptr || !parseBoolValue(value, &args->compactNormalWeight)) {
                return fail("--compact-normal-weight must be true or false");
            }
        } else if (key == "--verbose") {
            args->verbose = true;
        } else if (key == "--debug_MNN" || key == "--debug-mnn") {
            if (value == nullptr || !parseBoolValue(value, &args->debugMNN)) {
                return fail("--debug_MNN must be true or false");
            }
        } else if (key == "--diag-zo-check") {
            args->diagZoCheck = true;
        } else if (key == "--diag-repeat") {
            args->diagRepeat = std::atoi(value);
        } else if (key == "--diag-b-readback") {
            args->diagBReadback = true;
        } else if (key == "--baseline-step") {
            args->baselineStep = std::atoi(value);
        } else if (key == "--diag-isolate-b") {
            args->diagIsolateB = std::atoi(value);
        } else if (key == "--diag-isolate-value") {
            args->diagIsolateValue = static_cast<float>(std::atof(value));
        } else if (key == "--diag-prefill-decode" || key == "--diag-prefill-vs-decode") {
            args->diagPrefillDecode = std::atoi(value);
        } else if (key == "--diag-layerwise") {
            args->diagLayerwise = std::atoi(value);
        } else if (key == "--diag-layerwise-layers") {
            if (value == nullptr || !parseIntList(value, &args->diagLayerwiseLayers)) {
                return fail("--diag-layerwise-layers must be a comma-separated list of non-negative integers or all");
            }
        } else if (key == "--diag-layerwise-values") {
            args->diagLayerwiseValues = std::atoi(value);
        } else if (key == "--diag-lora-perturb") {
            args->diagLoraPerturb = std::atoi(value);
        } else if (key == "--diag-lora-a") {
            args->diagLoraA = std::atoi(value);
        } else if (key == "--diag-lora-branch") {
            args->diagLoraBranch = std::atoi(value);
        } else if (key == "--diag-lora-targets") {
            if (value == nullptr || !parseStringList(value, &args->diagLoraTargets)) {
                return fail("--diag-lora-targets must be a comma-separated canonical target list, or all");
            }
        } else if (key == "--diag-lora-values") {
            args->diagLoraValues = std::atoi(value);
        } else if (key == "--diag-lora-pattern") {
            args->diagLoraPattern = value;
            std::transform(args->diagLoraPattern.begin(), args->diagLoraPattern.end(),
                           args->diagLoraPattern.begin(), [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
        } else if (key == "--diag-lora-propagation") {
            if (value == nullptr || !parseBoolValue(value, &args->diagLoraPropagation)) {
                return fail("--diag-lora-propagation must be true or false");
            }
        } else if (key == "--diag-attn-internals") {
            if (value == nullptr || !parseBoolValue(value, &args->diagAttnInternals)) {
                return fail("--diag-attn-internals must be true or false");
            }
        } else if (key == "--diag-attn-heads") {
            if (value == nullptr || !parseIntList(value, &args->diagAttnHeads)) {
                return fail("--diag-attn-heads must be a comma-separated list of non-negative integers");
            }
        } else if (key == "--diag-attn-values") {
            args->diagAttnValues = std::atoi(value);
        } else if (key == "--force-full-causal-mask") {
            if (value == nullptr || !parseBoolValue(value, &args->forceFullCausalMask)) {
                return fail("--force-full-causal-mask must be true or false");
            }
        } else if (key == "--help" || key == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (!key.empty() && key[0] != '-' && args->config.empty()) {
            args->config = key;
        } else {
            return fail("unknown argument: " + key);
        }
    }
    if (args->config.empty()) {
        return fail("--config is required");
    }
    if (args->trainData.empty()) {
        return fail("--train-data is required");
    }
    if (args->loraMeta.empty()) {
        args->loraMeta = dirname(args->config) + "/lora_fa.json";
    }
    if (args->steps <= 0 || args->batch <= 0 || args->seqLen <= 0 || args->evalStep == 0 || args->evalStep < -1 ||
        args->epsilon <= 0.0f || args->lr <= 0.0f) {
        return fail("steps, batch, seq-len, epsilon and lr must be positive; eval-step must be positive or -1");
    }
    if (args->precision != "low" && args->precision != "normal" && args->precision != "high") {
        return fail("--precision must be one of: low, normal, high");
    }
    if (args->bParamPrecision != "fp16" && args->bParamPrecision != "fp32") {
        return fail("--b-param-precision must be one of: fp16, fp32");
    }
    if (args->attentionMode < 0) {
        return fail("--attention-mode must be non-negative");
    }
    if (args->diagRepeat < 0) {
        return fail("--diag-repeat must be non-negative");
    }
    if (args->baselineStep < 0) {
        return fail("--baseline-step must be non-negative");
    }
    if (args->diagPrefillDecode < 0) {
        return fail("--diag-prefill-decode must be non-negative");
    }
    if (args->diagLayerwise < 0) {
        return fail("--diag-layerwise must be non-negative");
    }
    if (args->diagLayerwiseValues <= 0) {
        return fail("--diag-layerwise-values must be positive");
    }
    if (args->diagLoraPerturb < 0) {
        return fail("--diag-lora-perturb must be non-negative");
    }
    if (args->diagLoraA < 0) {
        return fail("--diag-lora-a must be non-negative");
    }
    if (args->diagLoraBranch < 0) {
        return fail("--diag-lora-branch must be non-negative");
    }
    if (args->diagLoraValues <= 0) {
        return fail("--diag-lora-values must be positive");
    }
    if (args->diagLoraPattern != "noise" && args->diagLoraPattern != "ramp" &&
        args->diagLoraPattern != "constant") {
        return fail("--diag-lora-pattern must be one of: noise, ramp, constant");
    }
    if (args->diagLoraPropagation && args->diagLoraBranch <= 0) {
        return fail("--diag-lora-propagation requires --diag-lora-branch > 0");
    }
    if (args->diagAttnInternals && args->diagLoraBranch <= 0) {
        return fail("--diag-attn-internals requires --diag-lora-branch > 0");
    }
    if (args->diagAttnInternals && args->diagAttnHeads.empty()) {
        return fail("--diag-attn-heads must contain at least one head when --diag-attn-internals is true");
    }
    if (args->diagAttnValues <= 0) {
        return fail("--diag-attn-values must be positive");
    }
    if (args->evalStep != -1 && args->validData.empty()) {
        return fail("--valid-data is required unless --eval-step -1 is used");
    }
    return true;
}

static std::string readJsonString(const std::string& path, const char* key) {
    std::string content;
    if (!readFile(path, &content)) {
        return "";
    }
    rapidjson::Document doc;
    doc.Parse(content.c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember(key) || !doc[key].IsString()) {
        return "";
    }
    return doc[key].GetString();
}

static int getInt(const rapidjson::Value& obj, const char* key, int defaultValue) {
    if (!obj.HasMember(key) || !obj[key].IsInt()) {
        return defaultValue;
    }
    return obj[key].GetInt();
}

static size_t getSize(const rapidjson::Value& obj, const char* key, size_t defaultValue) {
    if (!obj.HasMember(key)) {
        return defaultValue;
    }
    if (obj[key].IsUint64()) {
        return static_cast<size_t>(obj[key].GetUint64());
    }
    if (obj[key].IsInt64()) {
        return static_cast<size_t>(obj[key].GetInt64());
    }
    return defaultValue;
}

static float getFloat(const rapidjson::Value& obj, const char* key, float defaultValue) {
    if (!obj.HasMember(key) || !obj[key].IsNumber()) {
        return defaultValue;
    }
    return static_cast<float>(obj[key].GetDouble());
}

static std::string getString(const rapidjson::Value& obj, const char* key, const std::string& defaultValue = "") {
    if (!obj.HasMember(key) || !obj[key].IsString()) {
        return defaultValue;
    }
    return obj[key].GetString();
}

static bool loadMetadata(const std::string& path, Metadata* meta) {
    std::string content;
    if (!readFile(path, &content)) {
        return false;
    }
    rapidjson::Document doc;
    doc.Parse(content.c_str());
    if (doc.HasParseError() || !doc.IsObject()) {
        return fail("invalid LoRA metadata JSON: " + path);
    }
    meta->rank = getInt(doc, "rank", 8);
    meta->alpha = getInt(doc, "alpha", meta->rank);
    meta->vocabSize = getInt(doc, "vocab_size", 32000);
    meta->totalBSize = getSize(doc, "total_b_size", 0);
    if (!doc.HasMember("targets") || !doc["targets"].IsArray()) {
        return fail("LoRA metadata has no targets array");
    }
    const auto& targets = doc["targets"];
    meta->targets.reserve(targets.Size());
    size_t computedTotal = 0;
    for (rapidjson::SizeType i = 0; i < targets.Size(); ++i) {
        const auto& item = targets[i];
        if (!item.IsObject() || !item.HasMember("b_input") || !item["b_input"].IsString()) {
            return fail("invalid target in LoRA metadata");
        }
        if (!item.HasMember("b_shape") || !item["b_shape"].IsArray() || item["b_shape"].Size() != 2) {
            return fail("invalid b_shape in LoRA metadata");
        }
        Target target;
        target.inputName = item["b_input"].GetString();
        target.canonicalKey = getString(item, "canonical_key");
        target.targetOp = getString(item, "target_op");
        target.injectionPoint = getString(item, "injection_point");
        target.hiddenInputSource = getString(item, "hidden_input_source");
        target.hiddenInputTensor = getInt(item, "hidden_input_tensor", -1);
        target.group = getString(item, "group");
        target.module = getString(item, "module");
        target.layer = getInt(item, "layer", -1);
        target.shape.push_back(item["b_shape"][0].GetInt());
        target.shape.push_back(item["b_shape"][1].GetInt());
        target.inputSize = getInt(item, "input_size", 0);
        target.outputSize = getInt(item, "output_size", target.shape[1]);
        target.scaleFoldedIntoA = getFloat(item, "lora_scale_folded_into_a",
                                           meta->rank != 0 ? static_cast<float>(meta->alpha) /
                                                             static_cast<float>(meta->rank) : 1.0f);
        target.offset = getSize(item, "offset", computedTotal);
        target.size = getSize(item, "size", static_cast<size_t>(target.shape[0]) * static_cast<size_t>(target.shape[1]));
        size_t shapeSize = static_cast<size_t>(target.shape[0]) * static_cast<size_t>(target.shape[1]);
        if (target.shape[0] <= 0 || target.shape[1] <= 0 || target.size != shapeSize) {
            return fail("invalid B tensor shape/size in LoRA metadata");
        }
        computedTotal = std::max(computedTotal, target.offset + target.size);
        meta->targets.push_back(target);
    }
    if (meta->targets.empty()) {
        return fail("LoRA metadata target list is empty");
    }
    if (meta->totalBSize == 0) {
        meta->totalBSize = computedTotal;
    }
    if (computedTotal > meta->totalBSize) {
        return fail("LoRA metadata total_b_size is smaller than target ranges");
    }
    return true;
}

static bool shouldPrintBDetail(size_t index, size_t count, int detailLimit) {
    return static_cast<int>(index) < detailLimit || index + 1 == count;
}

static uint32_t deriveStepSeed(uint32_t base, int step, uint32_t stream) {
    uint64_t x = static_cast<uint64_t>(base);
    x ^= static_cast<uint64_t>(stream) + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    x ^= static_cast<uint64_t>(static_cast<uint32_t>(step)) * 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<uint32_t>(x);
}

static uint32_t deriveTensorNoiseSeed(int seed, int step, size_t targetIndex) {
    return deriveStepSeed(static_cast<uint32_t>(seed), step, 0x5a4f544eU ^ static_cast<uint32_t>(targetIndex));
}

static uint32_t stableStringHash32(const std::string& text) {
    uint32_t h = 2166136261U;
    for (unsigned char c : text) {
        h ^= static_cast<uint32_t>(c);
        h *= 16777619U;
    }
    return h;
}

static uint32_t mixIndexHash32(uint32_t seed, size_t index) {
    uint64_t x = static_cast<uint64_t>(seed);
    x += 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(index) * 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<uint32_t>(x);
}

static float diagLoraPatternValue(const std::string& canonicalKey, const std::string& pattern,
                                  int seed, size_t mnnFlatIndex, int rank, int out) {
    (void)rank;
    const size_t r = out > 0 ? mnnFlatIndex / static_cast<size_t>(out) : 0;
    const size_t o = out > 0 ? mnnFlatIndex - r * static_cast<size_t>(out) : mnnFlatIndex;
    if (pattern == "constant") {
        return 1.0f;
    }
    if (pattern == "ramp") {
        return 0.1f * static_cast<float>(r + 1) + 0.00001f * static_cast<float>(o + 1);
    }
    const uint32_t stream = 0x4c4f5241U ^ stableStringHash32(canonicalKey);
    const uint32_t base = deriveStepSeed(static_cast<uint32_t>(seed), 0, stream);
    const uint32_t bits = mixIndexHash32(base, mnnFlatIndex);
    const float unit = (static_cast<float>(bits >> 8) + 0.5f) * (1.0f / 16777216.0f);
    return 2.0f * unit - 1.0f;
}

static int findTargetByCanonicalKey(const Metadata& meta, const std::string& key) {
    for (size_t i = 0; i < meta.targets.size(); ++i) {
        if (meta.targets[i].canonicalKey == key) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static VARPS createBVars(const Metadata& meta) {
    VARPS vars;
    vars.reserve(meta.targets.size());
    for (size_t i = 0; i < meta.targets.size(); ++i) {
        const auto& target = meta.targets[i];
        auto var = _Input(target.shape, NCHW, halide_type_of<float>());
        var->setName(target.inputName.c_str());
        vars.push_back(var);
    }
    return vars;
}

static size_t bStorageIndex(const Target& target, size_t mnnFlatIndex, const std::string& layout) {
    if (layout == "ggml") {
        const size_t rank = static_cast<size_t>(target.shape[0]);
        const size_t out = static_cast<size_t>(target.shape[1]);
        const size_t r = mnnFlatIndex / out;
        const size_t o = mnnFlatIndex - r * out;
        return target.offset + o * rank + r;
    }
    return target.offset + mnnFlatIndex;
}

static bool syncBVars(const Metadata& meta, const std::vector<float>& storage, const VARPS& vars,
                      bool snapToFp16, const std::string& bStorageLayout) {
    if (vars.size() != meta.targets.size()) {
        return fail("B VARP count does not match metadata");
    }
    for (size_t i = 0; i < meta.targets.size(); ++i) {
        const auto& target = meta.targets[i];
        if (target.offset + target.size > storage.size()) {
            return fail("B storage range is out of bounds");
        }
        auto ptr = vars[i]->writeMap<float>();
        if (ptr == nullptr) {
            return fail("failed to map B VARP for write");
        }
        for (size_t j = 0; j < target.size; ++j) {
            float value = storage[bStorageIndex(target, j, bStorageLayout)];
            ptr[j] = snapToFp16 ? static_cast<float>(half_float::half(value)) : value;
        }
    }
    return true;
}

static bool printBVarReadback(const Metadata& meta, const std::vector<float>& storage, const VARPS& vars,
                              bool snapToFp16, const char* tag, const std::string& bStorageLayout,
                              int detailLimit = 4) {
    if (vars.size() != meta.targets.size()) {
        return fail("B VARP count does not match metadata for readback");
    }
    double totalSumSq = 0.0;
    double totalExpectedSumSq = 0.0;
    double totalDiffSq = 0.0;
    float totalMaxAbs = 0.0f;
    float totalMaxDiff = 0.0f;
    size_t totalCount = 0;
    for (size_t i = 0; i < meta.targets.size(); ++i) {
        const auto& target = meta.targets[i];
        if (target.offset + target.size > storage.size()) {
            return fail("B storage range is out of bounds during readback");
        }
        const float* ptr = vars[i]->readMap<float>();
        if (ptr == nullptr) {
            return fail("failed to map B VARP for readback");
        }
        double sumSq = 0.0;
        double expectedSumSq = 0.0;
        double diffSq = 0.0;
        float maxAbs = 0.0f;
        float maxDiff = 0.0f;
        for (size_t j = 0; j < target.size; ++j) {
            float expected = storage[bStorageIndex(target, j, bStorageLayout)];
            expected = snapToFp16 ? static_cast<float>(half_float::half(expected)) : expected;
            float got = ptr[j];
            float absGot = std::fabs(got);
            float diff = std::fabs(got - expected);
            sumSq += static_cast<double>(got) * static_cast<double>(got);
            expectedSumSq += static_cast<double>(expected) * static_cast<double>(expected);
            diffSq += static_cast<double>(got - expected) * static_cast<double>(got - expected);
            if (absGot > maxAbs) {
                maxAbs = absGot;
            }
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        totalSumSq += sumSq;
        totalExpectedSumSq += expectedSumSq;
        totalDiffSq += diffSq;
        totalCount += target.size;
        if (maxAbs > totalMaxAbs) {
            totalMaxAbs = maxAbs;
        }
        if (maxDiff > totalMaxDiff) {
            totalMaxDiff = maxDiff;
        }
        if (static_cast<int>(i) < detailLimit || i + 1 == meta.targets.size()) {
            double rms = target.size == 0 ? 0.0 : std::sqrt(sumSq / static_cast<double>(target.size));
            double expectedRms = target.size == 0 ? 0.0 : std::sqrt(expectedSumSq / static_cast<double>(target.size));
            double diffRms = target.size == 0 ? 0.0 : std::sqrt(diffSq / static_cast<double>(target.size));
            std::cerr << "diag_b_readback_detail"
                      << " tag=" << (tag != nullptr ? tag : "B")
                      << " layout=" << bStorageLayout
                      << " target_index=" << i
                      << " input_name=" << target.inputName
                      << " rms=" << rms
                      << " expected_rms=" << expectedRms
                      << " diff_rms=" << diffRms
                      << " max_abs=" << maxAbs
                      << " max_diff=" << maxDiff
                      << " first_values=[";
            size_t n = std::min<size_t>(target.size, 4);
            for (size_t j = 0; j < n; ++j) {
                if (j > 0) {
                    std::cerr << ",";
                }
                std::cerr << ptr[j];
            }
            std::cerr << "]\n";
        }
    }
    double rms = totalCount == 0 ? 0.0 : std::sqrt(totalSumSq / static_cast<double>(totalCount));
    double expectedRms = totalCount == 0 ? 0.0 : std::sqrt(totalExpectedSumSq / static_cast<double>(totalCount));
    double diffRms = totalCount == 0 ? 0.0 : std::sqrt(totalDiffSq / static_cast<double>(totalCount));
    std::cerr << "diag_b_readback_summary"
              << " tag=" << (tag != nullptr ? tag : "B")
              << " layout=" << bStorageLayout
              << " tensors=" << vars.size()
              << " values=" << totalCount
              << " rms=" << rms
              << " expected_rms=" << expectedRms
              << " diff_rms=" << diffRms
              << " max_abs=" << totalMaxAbs
              << " max_diff=" << totalMaxDiff
              << "\n";
    return true;
}

static size_t storageIndexForGgmlFlat(const Target& target, size_t ggmlFlatIndex, const std::string& layout) {
    const size_t rank = static_cast<size_t>(target.shape[0]);
    const size_t out = static_cast<size_t>(target.shape[1]);
    const size_t o = ggmlFlatIndex / rank;
    const size_t r = ggmlFlatIndex - o * rank;
    if (layout == "ggml") {
        return target.offset + ggmlFlatIndex;
    }
    return target.offset + r * out + o;
}

static float maybeSnapFp16(float value, bool snapToFp16) {
    return snapToFp16 ? static_cast<float>(half_float::half(value)) : value;
}

static void printBStorageDebug(const Metadata& meta, const std::vector<float>& storage,
                               const char* tag, const std::string& bStorageLayout,
                               bool snapToFp16, int seed, int noiseStep,
                               int detailLimit = 6, size_t valueLimit = 8) {
    double totalSumAbs = 0.0;
    double totalSumSq = 0.0;
    float totalMaxAbs = 0.0f;
    size_t totalCount = 0;
    for (const auto& target : meta.targets) {
        if (target.offset + target.size > storage.size()) {
            continue;
        }
        for (size_t j = 0; j < target.size; ++j) {
            const float value = storage[target.offset + j];
            const float absValue = std::fabs(value);
            totalSumAbs += static_cast<double>(absValue);
            totalSumSq += static_cast<double>(value) * static_cast<double>(value);
            totalMaxAbs = std::max(totalMaxAbs, absValue);
            totalCount++;
        }
    }
    const double denom = totalCount > 0 ? static_cast<double>(totalCount) : 1.0;
    std::cerr << "lora_b_debug_summary"
              << " tag=" << (tag != nullptr ? tag : "B")
              << " layout=" << bStorageLayout
              << " tensors=" << meta.targets.size()
              << " values=" << totalCount
              << " mean_abs=" << (totalSumAbs / denom)
              << " rms=" << std::sqrt(totalSumSq / denom)
              << " max_abs=" << totalMaxAbs
              << "\n";

    for (size_t i = 0; i < meta.targets.size(); ++i) {
        const auto& target = meta.targets[i];
        if (!shouldPrintBDetail(i, meta.targets.size(), detailLimit)) {
            continue;
        }
        if (target.offset + target.size > storage.size()) {
            std::cerr << "lora_b_debug"
                      << " tag=" << (tag != nullptr ? tag : "B")
                      << " target_index=" << i
                      << " input_name=" << target.inputName
                      << " range=out_of_bounds\n";
            continue;
        }

        double sumAbs = 0.0;
        double sumSq = 0.0;
        float maxAbs = 0.0f;
        for (size_t j = 0; j < target.size; ++j) {
            const float value = storage[target.offset + j];
            const float absValue = std::fabs(value);
            sumAbs += static_cast<double>(absValue);
            sumSq += static_cast<double>(value) * static_cast<double>(value);
            maxAbs = std::max(maxAbs, absValue);
        }
        const double targetDenom = target.size > 0 ? static_cast<double>(target.size) : 1.0;

        std::cerr << "lora_b_debug"
                  << " tag=" << (tag != nullptr ? tag : "B")
                  << " layout=" << bStorageLayout
                  << " target_index=" << i
                  << " input_name=" << target.inputName
                  << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
                  << " target_op=" << (target.targetOp.empty() ? "<missing>" : target.targetOp)
                  << " shape_rank=" << target.shape[0]
                  << " shape_out=" << target.shape[1]
                  << " offset=" << target.offset
                  << " size=" << target.size
                  << " noise_seed_step=" << noiseStep
                  << " noise_seed=" << deriveTensorNoiseSeed(seed, noiseStep, i)
                  << " mean_abs=" << (sumAbs / targetDenom)
                  << " rms=" << std::sqrt(sumSq / targetDenom)
                  << " max_abs=" << maxAbs
                  << " first_storage=[";
        size_t n = std::min(target.size, valueLimit);
        for (size_t j = 0; j < n; ++j) {
            if (j > 0) {
                std::cerr << ",";
            }
            std::cerr << storage[target.offset + j];
        }
        std::cerr << "] first_mnn_flat=[";
        for (size_t j = 0; j < n; ++j) {
            if (j > 0) {
                std::cerr << ",";
            }
            std::cerr << maybeSnapFp16(storage[bStorageIndex(target, j, bStorageLayout)], snapToFp16);
        }
        std::cerr << "] first_ggml_flat=[";
        for (size_t j = 0; j < n; ++j) {
            if (j > 0) {
                std::cerr << ",";
            }
            std::cerr << maybeSnapFp16(storage[storageIndexForGgmlFlat(target, j, bStorageLayout)], snapToFp16);
        }
        std::cerr << "]\n";
    }
}

static void printTargetOrderDebug(const Metadata& meta, int seed, int noiseStep, int detailLimit = 12) {
    for (size_t i = 0; i < meta.targets.size(); ++i) {
        if (!shouldPrintBDetail(i, meta.targets.size(), detailLimit)) {
            continue;
        }
        const auto& target = meta.targets[i];
        std::cerr << "lora_b_target_order"
                  << " target_index=" << i
                  << " input_name=" << target.inputName
                  << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
                  << " target_op=" << (target.targetOp.empty() ? "<missing>" : target.targetOp)
                  << " shape_rank=" << target.shape[0]
                  << " shape_out=" << target.shape[1]
                  << " offset=" << target.offset
                  << " size=" << target.size
                  << " noise_seed_step=" << noiseStep
                  << " noise_seed=" << deriveTensorNoiseSeed(seed, noiseStep, i)
                  << "\n";
    }
}

static void configureMnnActivationDebug(MnnActivationDebugState* state, const Metadata& meta,
                                        bool layerwise, const std::vector<int>& layerwiseLayers,
                                        int maxTensorValues) {
    if (state == nullptr) {
        return;
    }
    state->enabled = true;
    state->layerwise = layerwise;
    state->loraBranch = false;
    state->activeCanonicalKey.clear();
    state->activeInjectionPoint.clear();
    state->activeTargetOp.clear();
    state->activeGroup.clear();
    state->activeModule.clear();
    state->activeLayer = -1;
    state->loraPropagation = false;
    state->layerwiseLayers = layerwiseLayers;
    state->maxTensorValues = maxTensorValues;
    if (state->layerwise) {
        state->targetPrefixes.clear();
        state->maxOpsPerForward = 512;
        state->maxTensorSlots = 1;
        std::cerr << "mnn_activation_debug_config enabled=true mode=layerwise layers=";
        if (state->layerwiseLayers.empty()) {
            std::cerr << "all";
        } else {
            for (size_t i = 0; i < state->layerwiseLayers.size(); ++i) {
                if (i > 0) {
                    std::cerr << ",";
                }
                std::cerr << state->layerwiseLayers[i];
            }
        }
        std::cerr << " values=" << state->maxTensorValues << "\n";
        return;
    }
    state->targetPrefixes.clear();
    auto addTarget = [&](size_t index) {
        if (index >= meta.targets.size()) {
            return;
        }
        const std::string prefix = meta.targets[index].targetOp + "/lora_fa/";
        if (std::find(state->targetPrefixes.begin(), state->targetPrefixes.end(), prefix) ==
            state->targetPrefixes.end()) {
            state->targetPrefixes.push_back(prefix);
        }
    };

    if (!meta.targets.empty()) {
        addTarget(0);
        addTarget(1);
        addTarget(meta.targets.size() / 2);
        if (meta.targets.size() > 2) {
            addTarget(meta.targets.size() - 2);
        }
        addTarget(meta.targets.size() - 1);
    }

    std::cerr << "mnn_activation_debug_config enabled=true mode=lora selected_targets=" << state->targetPrefixes.size();
    for (size_t i = 0; i < state->targetPrefixes.size(); ++i) {
        std::cerr << " prefix" << i << "=\"" << state->targetPrefixes[i] << "\"";
    }
    std::cerr << "\n";
}

static void configureMnnActivationDebugForTarget(MnnActivationDebugState* state, const Target& target,
                                                 int maxTensorValues, bool loraBranch = false,
                                                 bool loraPropagation = false) {
    if (state == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> guard(state->mutex);
    state->enabled = true;
    state->layerwise = false;
    state->loraBranch = loraBranch;
    state->activeCanonicalKey = target.canonicalKey;
    state->activeInjectionPoint = target.injectionPoint;
    state->activeTargetOp = target.targetOp;
    state->activeGroup = target.group;
    state->activeModule = target.module;
    state->activeLayer = target.layer;
    state->loraPropagation = loraBranch && loraPropagation;
    state->layerwiseLayers.clear();
    state->targetPrefixes.clear();
    if (!target.targetOp.empty()) {
        state->targetPrefixes.push_back(target.targetOp + "/lora_fa/");
    }
    state->maxTensorValues = maxTensorValues;
    state->maxOpsPerForward = loraBranch ? (state->loraPropagation ? 512 : 256) : 64;
    state->maxTensorSlots = 3;
}

static void setMnnAttentionInternalDebugEnv(const Args& args, const Target& target, const char* tag,
                                            int dataIndex, int sampleOrdinal, const Sample& sample) {
    if (!args.diagAttnInternals) {
        return;
    }
    setProcessEnv("MNN_ATTN_DEBUG_ENABLED", "1");
    setProcessEnv("MNN_ATTN_DEBUG_TAG", tag != nullptr ? tag : "");
    setProcessEnv("MNN_ATTN_DEBUG_TARGET", target.canonicalKey.empty() ? target.inputName : target.canonicalKey);
    setProcessEnv("MNN_ATTN_DEBUG_LAYER", std::to_string(target.layer));
    setProcessEnv("MNN_ATTN_DEBUG_QPOS",
                  std::to_string(sample.tokens.empty() ? -1 : static_cast<int>(sample.tokens.size()) - 1));
    setProcessEnv("MNN_ATTN_DEBUG_HEADS", joinIntList(args.diagAttnHeads));
    setProcessEnv("MNN_ATTN_DEBUG_VALUES", std::to_string(args.diagAttnValues));
    setProcessEnv("MNN_ATTN_DEBUG_DATA_INDEX", std::to_string(dataIndex));
    setProcessEnv("MNN_ATTN_DEBUG_SAMPLE_ORDINAL", std::to_string(sampleOrdinal));
}

static void clearMnnAttentionInternalDebugEnv() {
    unsetProcessEnv("MNN_ATTN_DEBUG_ENABLED");
    unsetProcessEnv("MNN_ATTN_DEBUG_TAG");
    unsetProcessEnv("MNN_ATTN_DEBUG_TARGET");
    unsetProcessEnv("MNN_ATTN_DEBUG_LAYER");
    unsetProcessEnv("MNN_ATTN_DEBUG_QPOS");
    unsetProcessEnv("MNN_ATTN_DEBUG_HEADS");
    unsetProcessEnv("MNN_ATTN_DEBUG_VALUES");
    unsetProcessEnv("MNN_ATTN_DEBUG_DATA_INDEX");
    unsetProcessEnv("MNN_ATTN_DEBUG_SAMPLE_ORDINAL");
}

static void printMnnLoraPerturbBTargetDebug(const Metadata& meta, const std::vector<float>& storage,
                                            size_t targetIndex, const char* tag,
                                            const std::string& pattern, float epsilon,
                                            const std::string& bStorageLayout, bool snapToFp16,
                                            int valueLimit) {
    if (targetIndex >= meta.targets.size()) {
        return;
    }
    const auto& target = meta.targets[targetIndex];
    if (target.offset + target.size > storage.size()) {
        std::cerr << "diag_lora_b_debug"
                  << " framework=mnn"
                  << " tag=" << (tag != nullptr ? tag : "B")
                  << " target_index=" << targetIndex
                  << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
                  << " range=out_of_bounds\n";
        return;
    }
    double sumAbs = 0.0;
    double sumSq = 0.0;
    float maxAbs = 0.0f;
    for (size_t j = 0; j < target.size; ++j) {
        const float value = maybeSnapFp16(storage[bStorageIndex(target, j, bStorageLayout)], snapToFp16);
        const float absValue = std::fabs(value);
        sumAbs += static_cast<double>(absValue);
        sumSq += static_cast<double>(value) * static_cast<double>(value);
        maxAbs = std::max(maxAbs, absValue);
    }
    const double denom = target.size > 0 ? static_cast<double>(target.size) : 1.0;
    const size_t n = std::min(target.size, static_cast<size_t>(std::max(0, valueLimit)));
    std::cerr << "diag_lora_b_debug"
              << " framework=mnn"
              << " tag=" << (tag != nullptr ? tag : "B")
              << " layout=" << bStorageLayout
              << " pattern=" << pattern
              << " epsilon=" << epsilon
              << " target_index=" << targetIndex
              << " input_name=" << target.inputName
              << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
              << " target_op=" << (target.targetOp.empty() ? "<missing>" : target.targetOp)
              << " shape_rank=" << target.shape[0]
              << " shape_out=" << target.shape[1]
              << " offset=" << target.offset
              << " size=" << target.size
              << " mean_abs=" << (sumAbs / denom)
              << " rms=" << std::sqrt(sumSq / denom)
              << " max_abs=" << maxAbs
              << " first_mnn_flat=[";
    for (size_t j = 0; j < n; ++j) {
        if (j > 0) {
            std::cerr << ",";
        }
        std::cerr << maybeSnapFp16(storage[bStorageIndex(target, j, bStorageLayout)], snapToFp16);
    }
    std::cerr << "] first_storage=[";
    for (size_t j = 0; j < n; ++j) {
        if (j > 0) {
            std::cerr << ",";
        }
        std::cerr << maybeSnapFp16(storage[target.offset + j], snapToFp16);
    }
    std::cerr << "] first_ggml_flat=[";
    for (size_t j = 0; j < n; ++j) {
        if (j > 0) {
            std::cerr << ",";
        }
        std::cerr << maybeSnapFp16(storage[storageIndexForGgmlFlat(target, j, bStorageLayout)], snapToFp16);
    }
    std::cerr << "]\n";
}

static void printMnnLoraTargetMetadataDebug(const Metadata& meta, size_t targetIndex,
                                            int sampleOrdinal, int dataIndex,
                                            const char* prefix) {
    if (targetIndex >= meta.targets.size()) {
        return;
    }
    const auto& target = meta.targets[targetIndex];
    std::cerr << prefix
              << " framework=mnn"
              << " sample_ordinal=" << sampleOrdinal
              << " data_index=" << dataIndex
              << " target_index=" << targetIndex
              << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
              << " target_op=" << (target.targetOp.empty() ? "<missing>" : target.targetOp)
              << " injection_point=" << (target.injectionPoint.empty() ? "<missing>" : target.injectionPoint)
              << " hidden_input_tensor=" << target.hiddenInputTensor
              << " hidden_input_source=" << (target.hiddenInputSource.empty() ? "<missing>" : target.hiddenInputSource)
              << " input_size=" << target.inputSize
              << " output_size=" << target.outputSize
              << " rank=" << meta.rank
              << " alpha=" << meta.alpha
              << " scale_folded_into_A=" << target.scaleFoldedIntoA
              << " b_shape_rank=" << target.shape[0]
              << " b_shape_out=" << target.shape[1]
              << " b_offset=" << target.offset
              << " b_size=" << target.size
              << "\n";
}

static void installMnnActivationDebugCallbacks(Llm* llm, MnnActivationDebugState* state) {
    if (llm == nullptr || state == nullptr || !state->enabled) {
        return;
    }
    MNN::TensorCallBackWithInfo before = [state](const std::vector<MNN::Tensor*>& tensors,
                                                 const MNN::OperatorInfo* info) {
        return dumpMnnActivationCallback(state, tensors, info, "before");
    };
    MNN::TensorCallBackWithInfo after = [state](const std::vector<MNN::Tensor*>& tensors,
                                                const MNN::OperatorInfo* info) {
        return dumpMnnActivationCallback(state, tensors, info, "after");
    };
    llm->setDebugCallback(std::move(before), std::move(after));
}

// Compute L1/L2/Linf norm of bStorage for diagnostic logging.
static void summarizeB(const std::vector<float>& storage, double* sumAbs, double* sumSq, float* maxAbs) {
    double sa = 0.0;
    double ss = 0.0;
    float mx = 0.0f;
    for (size_t i = 0; i < storage.size(); ++i) {
        float v = storage[i];
        float av = std::fabs(v);
        sa += av;
        ss += static_cast<double>(v) * static_cast<double>(v);
        if (av > mx) mx = av;
    }
    *sumAbs = sa;
    *sumSq = ss;
    *maxAbs = mx;
}

static bool parseLabel(const std::string& raw, int* label) {
    std::string value = trim(raw);
    if (value.size() != 1 || (value[0] != '0' && value[0] != '1')) {
        return false;
    }
    *label = value[0] - '0';
    return true;
}

static std::vector<int> stripTokenizerPrefix(Llm* llm, const std::string& text) {
    std::vector<int> tokens = llm->tokenizer_encode(text);
    std::vector<int> prefix = llm->tokenizer_encode("");
    if (!prefix.empty() && tokens.size() >= prefix.size() &&
        std::equal(prefix.begin(), prefix.end(), tokens.begin())) {
        tokens.erase(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(prefix.size()));
    }
    return tokens;
}

static bool getLabelSuffixes(Llm* llm, int vocabSize,
                             std::vector<std::vector<int>>* labelSuffixes,
                             std::vector<int>* clsIdx) {
    labelSuffixes->resize(2);
    (*labelSuffixes)[0] = stripTokenizerPrefix(llm, " It was terrible");
    (*labelSuffixes)[1] = stripTokenizerPrefix(llm, " It was great");
    clsIdx->resize(2);
    for (size_t cls = 0; cls < labelSuffixes->size(); ++cls) {
        const std::vector<int>& suffix = (*labelSuffixes)[cls];
        if (suffix.empty()) {
            return fail("failed to tokenize SST2 label suffix");
        }
        (*clsIdx)[cls] = suffix.back();
        if ((*clsIdx)[cls] < 0 || (*clsIdx)[cls] >= vocabSize) {
            return fail("SST2 class token is out of vocab range");
        }
    }
    if ((*clsIdx)[0] == (*clsIdx)[1]) {
        return fail("SST2 class verbalizers resolved to the same token");
    }
    std::cerr << "class_token_negative=" << (*clsIdx)[0]
              << " class_token_positive=" << (*clsIdx)[1]
              << " label_suffix_lens=" << (*labelSuffixes)[0].size()
              << "," << (*labelSuffixes)[1].size() << "\n";
    return true;
}

static void printTokenList(const char* name, const std::vector<int>& tokens, Llm* llm, size_t limit = 24) {
    std::cerr << name << "_size=" << tokens.size() << " " << name << "=[";
    size_t n = std::min(tokens.size(), limit);
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            std::cerr << ",";
        }
        std::cerr << tokens[i];
    }
    if (tokens.size() > n) {
        std::cerr << ",...";
    }
    std::cerr << "]";
    if (!tokens.empty() && llm != nullptr) {
        std::cerr << " decoded=\"";
        for (size_t i = 0; i < n; ++i) {
            std::string piece = llm->tokenizer_decode(tokens[i]);
            for (char& c : piece) {
                if (c == '\n' || c == '\r' || c == '\t') {
                    c = ' ';
                }
            }
            std::cerr << piece;
        }
        if (tokens.size() > n) {
            std::cerr << "...";
        }
        std::cerr << "\"";
    }
    std::cerr << "\n";
}

static void printFloatStats(const std::string& prefix, const float* data, size_t size, size_t limit = 8) {
    if (data == nullptr || size == 0) {
        std::cerr << prefix << " size=" << size << " data=<null-or-empty>\n";
        return;
    }
    double sum = 0.0;
    double sumAbs = 0.0;
    double sumSq = 0.0;
    float minValue = std::numeric_limits<float>::infinity();
    float maxValue = -std::numeric_limits<float>::infinity();
    float maxAbs = 0.0f;
    size_t finiteCount = 0;
    for (size_t i = 0; i < size; ++i) {
        float value = data[i];
        if (!std::isfinite(value)) {
            continue;
        }
        float absValue = std::fabs(value);
        sum += static_cast<double>(value);
        sumAbs += static_cast<double>(absValue);
        sumSq += static_cast<double>(value) * static_cast<double>(value);
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        maxAbs = std::max(maxAbs, absValue);
        finiteCount++;
    }
    double denom = finiteCount > 0 ? static_cast<double>(finiteCount) : 1.0;
    std::cerr << prefix
              << " size=" << size
              << " finite=" << finiteCount
              << " mean=" << (sum / denom)
              << " mean_abs=" << (sumAbs / denom)
              << " rms=" << std::sqrt(sumSq / denom)
              << " min=" << (finiteCount > 0 ? minValue : 0.0f)
              << " max=" << (finiteCount > 0 ? maxValue : 0.0f)
              << " max_abs=" << maxAbs
              << " first=[";
    size_t n = std::min(size, limit);
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            std::cerr << ",";
        }
        std::cerr << data[i];
    }
    if (size > n) {
        std::cerr << ",...";
    }
    std::cerr << "]\n";
}

static void printVarStats(const std::string& prefix, VARP var) {
    if (var.get() == nullptr || var->getInfo() == nullptr) {
        std::cerr << prefix << " dims=[] size=0 var=<null>\n";
        return;
    }
    const auto* info = var->getInfo();
    std::ostringstream os;
    os << prefix << " dims=[";
    for (size_t i = 0; i < info->dim.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << info->dim[i];
    }
    os << "]";
    const float* data = var->readMap<float>();
    printFloatStats(os.str(), data, static_cast<size_t>(info->size));
}

static void printTokenizerDiagnostics(Llm* llm, const std::vector<int>& clsIdx,
                                      const std::vector<std::vector<int>>& labelSuffixes,
                                      const std::vector<Sample>& trainData) {
    printTokenList("tokenizer_prefix_empty", llm->tokenizer_encode(""), llm);
    printTokenList("suffix_raw_It_was", llm->tokenizer_encode("It was"), llm);
    printTokenList("suffix_stripped_It_was", stripTokenizerPrefix(llm, "It was"), llm);
    printTokenList("suffix_stripped_leading_space_It_was", stripTokenizerPrefix(llm, " It was"), llm);
    printTokenList("negative_word_stripped", stripTokenizerPrefix(llm, "terrible"), llm);
    printTokenList("negative_word_leading_space_stripped", stripTokenizerPrefix(llm, " terrible"), llm);
    printTokenList("negative_fallback_stripped", stripTokenizerPrefix(llm, "It was terrible"), llm);
    printTokenList("positive_word_stripped", stripTokenizerPrefix(llm, "great"), llm);
    printTokenList("positive_word_leading_space_stripped", stripTokenizerPrefix(llm, " great"), llm);
    printTokenList("positive_fallback_stripped", stripTokenizerPrefix(llm, "It was great"), llm);
    if (labelSuffixes.size() >= 2) {
        printTokenList("negative_label_suffix", labelSuffixes[0], llm);
        printTokenList("positive_label_suffix", labelSuffixes[1], llm);
    }
    std::cerr << "class_token_negative_decode=\"" << llm->tokenizer_decode(clsIdx[0])
              << "\" class_token_positive_decode=\"" << llm->tokenizer_decode(clsIdx[1]) << "\"\n";
    if (!trainData.empty()) {
        const Sample& sample = trainData[0];
        std::cerr << "first_train_sample label=" << sample.label
                  << " target_token=" << sample.targetToken
                  << " target_decode=\"" << llm->tokenizer_decode(sample.targetToken) << "\"\n";
        printTokenList("first_train_tokens", sample.tokens, llm);
    }
}

static void printSampleDebug(const std::string& tag, size_t batchPos, int dataIndex,
                             const Sample& sample, Llm* llm) {
    std::cerr << "debug_mnn_sample tag=" << tag
              << " batch_pos=" << batchPos
              << " data_index=" << dataIndex
              << " token_count=" << sample.tokens.size()
              << " label=" << sample.label
              << " target_token=" << sample.targetToken
              << " target_decode=\"" << (llm != nullptr ? llm->tokenizer_decode(sample.targetToken) : std::string())
              << "\" last_token=" << (sample.tokens.empty() ? -1 : sample.tokens.back())
              << " last_decode=\"" << (llm != nullptr && !sample.tokens.empty() ? llm->tokenizer_decode(sample.tokens.back()) : std::string())
              << "\"\n";
    printTokenList("debug_mnn_tokens", sample.tokens, llm, 64);
}

static void printActivationSampleDebug(const std::string& framework, const std::string& tag,
                                       int sampleOrdinal, int dataIndex, const Sample& sample) {
    std::cerr << "activation_sample_debug"
              << " framework=" << framework
              << " tag=" << tag
              << " sample_ordinal=" << sampleOrdinal
              << " data_index=" << dataIndex
              << " label=" << sample.label
              << " target=" << sample.targetToken
              << " final_pos=" << (sample.tokens.empty() ? -1 : static_cast<int>(sample.tokens.size()) - 1)
              << " token_count=" << sample.tokens.size()
              << " tokens=[";
    for (size_t i = 0; i < sample.tokens.size(); ++i) {
        if (i > 0) {
            std::cerr << ",";
        }
        std::cerr << sample.tokens[i];
    }
    std::cerr << "]\n";
}

static bool loadSst2Tsv(const std::string& path, Llm* llm, int maxSamples, int maxLen,
                        const std::vector<std::vector<int>>& labelSuffixes,
                        std::vector<Sample>* samples) {
    std::ifstream fin(path.c_str());
    if (!fin.is_open()) {
        return fail("cannot open SST2 TSV file: " + path);
    }
    samples->clear();
    // Keep label-prefix tokens in the prompt; the final suffix token remains
    // the separate next-token target.
    if (labelSuffixes.size() != 2) {
        return fail("SST2 expects exactly two label suffixes");
    }
    for (size_t cls = 0; cls < labelSuffixes.size(); ++cls) {
        const std::vector<int>& labelSuffix = labelSuffixes[cls];
        int promptSuffixTokens = static_cast<int>(labelSuffix.size()) - 1;
        if (labelSuffix.empty() || promptSuffixTokens >= maxLen) {
            return fail("invalid SST2 label suffix for seq-len");
        }
    }
    std::string line;
    bool first = true;
    while (std::getline(fin, line) && static_cast<int>(samples->size()) < maxSamples) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        size_t tab = line.rfind('\t');
        if (tab == std::string::npos) {
            continue;
        }
        std::string sentence = trim(line.substr(0, tab));
        int label = 0;
        if (!parseLabel(line.substr(tab + 1), &label)) {
            if (first) {
                first = false;
                continue;
            }
            continue;
        }
        first = false;
        const std::vector<int>& labelSuffix = labelSuffixes[static_cast<size_t>(label)];
        int promptSuffixTokens = static_cast<int>(labelSuffix.size()) - 1;
        std::vector<int> tokens = llm->tokenizer_encode(sentence);
        int maxSentenceTokens = std::max(1, maxLen - promptSuffixTokens);
        if (static_cast<int>(tokens.size()) > maxSentenceTokens) {
            tokens.resize(maxSentenceTokens);
        }
        tokens.insert(tokens.end(), labelSuffix.begin(), labelSuffix.end() - 1);
        if (tokens.empty()) {
            continue;
        }
        Sample sample;
        sample.tokens = tokens;
        sample.label = label;
        sample.targetToken = labelSuffix.back();
        samples->push_back(sample);
    }
    if (samples->empty()) {
        return fail("no valid SST2 samples loaded from " + path);
    }
    return true;
}

static void printLogitDiagnostics(const char* tag, const float* row, int vocab, int targetToken,
                                  VARP logits, int validLogitStart, int validLogitSize,
                                  const std::vector<int>* classTokens, Llm* llm) {
    float maxValue = -std::numeric_limits<float>::infinity();
    int maxToken = 0;
    for (int v = 0; v < vocab; ++v) {
        if (row[v] > maxValue) {
            maxValue = row[v];
            maxToken = v;
        }
    }
    double sum = 0.0;
    for (int v = 0; v < vocab; ++v) {
        sum += std::exp(static_cast<double>(row[v] - maxValue));
    }
    double logsumexp = std::log(sum) + static_cast<double>(maxValue);
    double loss = logsumexp - static_cast<double>(row[targetToken]);

    std::cerr << "logit_debug tag=" << (tag != nullptr ? tag : "loss")
              << " output_dims=[";
    if (logits.get() != nullptr && logits->getInfo() != nullptr) {
        const auto& dims = logits->getInfo()->dim;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) {
                std::cerr << ",";
            }
            std::cerr << dims[i];
        }
        std::cerr << "] output_size=" << logits->getInfo()->size;
    } else {
        std::cerr << "] output_size=0";
    }
    std::cerr << " valid_start=" << validLogitStart
              << " valid_size=" << validLogitSize
              << " vocab=" << vocab
              << " target=" << targetToken
              << " target_decode=\"" << (llm != nullptr ? llm->tokenizer_decode(targetToken) : std::string())
              << "\" target_logit=" << row[targetToken]
              << " max_token=" << maxToken
              << " max_decode=\"" << (llm != nullptr ? llm->tokenizer_decode(maxToken) : std::string())
              << "\" max_logit=" << maxValue
              << " logsumexp=" << logsumexp
              << " loss=" << loss;
    if (classTokens != nullptr && classTokens->size() >= 2 &&
        (*classTokens)[0] >= 0 && (*classTokens)[0] < vocab &&
        (*classTokens)[1] >= 0 && (*classTokens)[1] < vocab) {
        const float negLogit = row[(*classTokens)[0]];
        const float posLogit = row[(*classTokens)[1]];
        const float classMax = std::max(negLogit, posLogit);
        const double classLogsumexp = std::log(std::exp(static_cast<double>(negLogit - classMax)) +
                                               std::exp(static_cast<double>(posLogit - classMax))) +
                                      static_cast<double>(classMax);
        const double binaryLoss = classLogsumexp - static_cast<double>(row[targetToken]);
        std::cerr << " neg_logit=" << row[(*classTokens)[0]]
                  << " pos_logit=" << row[(*classTokens)[1]]
                  << " class_margin_pos_minus_neg=" << (posLogit - negLogit)
                  << " binary_verbalizer_loss=" << binaryLoss;
    }
    std::cerr << "\n";
    printFloatStats(std::string("logit_row_stats tag=") + (tag != nullptr ? tag : "loss"), row, static_cast<size_t>(vocab));

    std::vector<int> top;
    top.reserve(5);
    for (int v = 0; v < vocab; ++v) {
        auto pos = std::lower_bound(top.begin(), top.end(), v, [&](int lhs, int rhs) {
            return row[lhs] > row[rhs];
        });
        top.insert(pos, v);
        if (top.size() > 5) {
            top.pop_back();
        }
    }
    for (size_t i = 0; i < top.size(); ++i) {
        int token = top[i];
        std::cerr << "logit_top" << (i + 1)
                  << " token=" << token
                  << " logit=" << row[token]
                  << " decode=\"" << (llm != nullptr ? llm->tokenizer_decode(token) : std::string())
                  << "\"\n";
    }
}

static bool getLastLogitRow(VARP logits, int expectedVocab, int validLogitStart, int validLogitSize,
                            const float** row, int* vocab) {
    if (logits.get() == nullptr || logits->getInfo() == nullptr) {
        return fail("logits is null");
    }
    const auto* info = logits->getInfo();
    const float* data = logits->readMap<float>();
    if (data == nullptr || info->size <= 0) {
        return fail("failed to read logits");
    }
    if (validLogitStart >= 0 && validLogitSize > 0) {
        if (validLogitStart + validLogitSize > info->size) {
            return fail("valid logits range is out of output tensor bounds");
        }
        *row = data + validLogitStart;
        *vocab = validLogitSize;
        return true;
    }
    int inferredVocab = expectedVocab;
    if (!info->dim.empty()) {
        int lastDim = info->dim.back();
        if (lastDim > 1 && info->size % lastDim == 0) {
            inferredVocab = lastDim;
        }
    }
    if (inferredVocab <= 1 || info->size % inferredVocab != 0) {
        return fail("cannot infer logits vocab dimension");
    }
    int positions = static_cast<int>(info->size / inferredVocab);
    if (positions <= 0) {
        return fail("no logits positions available");
    }
    *row = data + static_cast<size_t>(positions - 1) * static_cast<size_t>(inferredVocab);
    *vocab = inferredVocab;
    return true;
}

static bool copyLastLogitRow(VARP logits, int expectedVocab, int validLogitStart, int validLogitSize,
                             std::vector<float>* out, int* vocab) {
    const float* row = nullptr;
    int actualVocab = 0;
    if (!getLastLogitRow(logits, expectedVocab, validLogitStart, validLogitSize, &row, &actualVocab)) {
        return false;
    }
    out->assign(row, row + actualVocab);
    if (vocab != nullptr) {
        *vocab = actualVocab;
    }
    return true;
}

static int argmaxLogitRow(const std::vector<float>& row) {
    if (row.empty()) {
        return -1;
    }
    int best = 0;
    for (int i = 1; i < static_cast<int>(row.size()); ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return best;
}

struct LogitDiffStats {
    float maxAbsDiff = 0.0f;
    int maxAbsToken = -1;
    double meanAbsDiff = 0.0;
    double rmsDiff = 0.0;
    float targetDiff = std::numeric_limits<float>::quiet_NaN();
    float negDiff = std::numeric_limits<float>::quiet_NaN();
    float posDiff = std::numeric_limits<float>::quiet_NaN();
    int prefillTop1 = -1;
    int decodeTop1 = -1;
};

static bool computeLogitDiffStats(const std::vector<float>& prefill, const std::vector<float>& decode,
                                  int targetToken, const std::vector<int>* classTokens,
                                  LogitDiffStats* stats) {
    if (prefill.size() != decode.size() || prefill.empty()) {
        return fail("cannot compare logits with different or empty vocab sizes");
    }
    LogitDiffStats local;
    double absSum = 0.0;
    double sqSum = 0.0;
    for (int i = 0; i < static_cast<int>(prefill.size()); ++i) {
        const float diff = prefill[i] - decode[i];
        const float absDiff = std::fabs(diff);
        absSum += static_cast<double>(absDiff);
        sqSum += static_cast<double>(diff) * static_cast<double>(diff);
        if (absDiff > local.maxAbsDiff) {
            local.maxAbsDiff = absDiff;
            local.maxAbsToken = i;
        }
    }
    const double denom = static_cast<double>(prefill.size());
    local.meanAbsDiff = absSum / denom;
    local.rmsDiff = std::sqrt(sqSum / denom);
    if (targetToken >= 0 && targetToken < static_cast<int>(prefill.size())) {
        local.targetDiff = prefill[targetToken] - decode[targetToken];
    }
    if (classTokens != nullptr && classTokens->size() >= 2) {
        const int neg = (*classTokens)[0];
        const int pos = (*classTokens)[1];
        if (neg >= 0 && neg < static_cast<int>(prefill.size())) {
            local.negDiff = prefill[neg] - decode[neg];
        }
        if (pos >= 0 && pos < static_cast<int>(prefill.size())) {
            local.posDiff = prefill[pos] - decode[pos];
        }
    }
    local.prefillTop1 = argmaxLogitRow(prefill);
    local.decodeTop1 = argmaxLogitRow(decode);
    *stats = local;
    return true;
}

static void printLogitDiffDiagnostics(const std::string& tag, const LogitDiffStats& stats,
                                      int vocab, Llm* llm) {
    std::cerr << "logit_diff tag=" << tag
              << " vocab=" << vocab
              << " diff=prefill_minus_decode"
              << " max_abs_diff=" << stats.maxAbsDiff
              << " max_abs_token=" << stats.maxAbsToken
              << " max_abs_decode=\"" << (llm != nullptr && stats.maxAbsToken >= 0 ?
                                             llm->tokenizer_decode(stats.maxAbsToken) : std::string())
              << "\" mean_abs_diff=" << stats.meanAbsDiff
              << " rms_diff=" << stats.rmsDiff
              << " target_diff=" << stats.targetDiff
              << " neg_diff=" << stats.negDiff
              << " pos_diff=" << stats.posDiff
              << " top1_prefill=" << stats.prefillTop1
              << " top1_prefill_decode=\"" << (llm != nullptr && stats.prefillTop1 >= 0 ?
                                                 llm->tokenizer_decode(stats.prefillTop1) : std::string())
              << "\" top1_decode=" << stats.decodeTop1
              << " top1_decode_decode=\"" << (llm != nullptr && stats.decodeTop1 >= 0 ?
                                                llm->tokenizer_decode(stats.decodeTop1) : std::string())
              << "\"\n";
}

static bool tokenCrossEntropy(const float* row, int vocab, int targetToken, double* loss) {
    if (targetToken < 0 || targetToken >= vocab) {
        return fail("target token is out of logits range");
    }
    float maxValue = row[0];
    for (int v = 1; v < vocab; ++v) {
        maxValue = std::max(maxValue, row[v]);
    }
    double sum = 0.0;
    for (int v = 0; v < vocab; ++v) {
        sum += std::exp(static_cast<double>(row[v] - maxValue));
    }
    *loss = std::log(sum) + static_cast<double>(maxValue) - static_cast<double>(row[targetToken]);
    return true;
}

static bool evaluateBatchLoss(Llm* llm, const std::vector<Sample>& trainData, const std::vector<int>& batchIndices,
                              int vocabSize, int64_t* prefillUs, int64_t* prefillTokens, double* outLoss,
                              LossDebugState* debug = nullptr) {
    double lossSum = 0.0;
    int count = 0;
    for (size_t i = 0; i < batchIndices.size(); ++i) {
        const Sample& sample = trainData[batchIndices[i]];
        llm->reset();
        bool doDebugSample = debug != nullptr && debug->enabled && debug->printedSamples < debug->sampleLimit;
        if (doDebugSample) {
            printSampleDebug(debug->tag, i, batchIndices[i], sample, debug->llm);
            if (debug->printEmbedding) {
                auto embeds = llm->embedding(sample.tokens);
                printVarStats("debug_mnn_embedding tag=" + debug->tag, embeds);
            }
        }
        if (doDebugSample && debug->activationDebug != nullptr) {
            beginMnnActivationForward(debug->activationDebug, debug->tag, batchIndices[i], static_cast<int>(i),
                                      static_cast<int>(sample.tokens.size()),
                                      static_cast<int>(sample.tokens.empty() ? 0 : sample.tokens.size() - 1),
                                      sample.label, sample.targetToken);
        }
        llm->response(sample.tokens, nullptr, nullptr, 0);
        if (doDebugSample && debug->activationDebug != nullptr) {
            endMnnActivationForward(debug->activationDebug);
        }
        auto outputs = llm->getOutputs();
        if (outputs.empty()) {
            return fail("LLM forward produced no outputs");
        }
        const float* row = nullptr;
        int vocab = 0;
        if (!getLastLogitRow(outputs[0], vocabSize, llm->getValidLogitStart(), llm->getValidLogitSize(), &row, &vocab)) {
            return false;
        }
        if (doDebugSample) {
            printLogitDiagnostics(debug->tag.c_str(), row, vocab, sample.targetToken, outputs[0],
                                  llm->getValidLogitStart(), llm->getValidLogitSize(),
                                  debug->classTokens, debug->llm);
            debug->printed = true;
            debug->printedSamples++;
        }
        double loss = 0.0;
        if (!tokenCrossEntropy(row, vocab, sample.targetToken, &loss)) {
            return false;
        }
        lossSum += loss;
        count++;
        const auto* context = llm->getContext();
        if (context != nullptr) {
            *prefillUs += context->prefill_us;
        }
        *prefillTokens += static_cast<int64_t>(sample.tokens.size());
        auto scope = MNN::Express::ExecutorScope::Current();
        if (scope != nullptr) {
            scope->gc();
        }
    }
    if (count == 0) {
        return fail("empty training batch");
    }
    *outLoss = lossSum / static_cast<double>(count);
    return true;
}

static bool evaluateAccuracy(Llm* llm, const std::vector<Sample>& validData, const std::vector<int>& clsIdx,
                             int vocabSize, double* outAcc) {
    int correct = 0;
    int total = 0;
    for (size_t i = 0; i < validData.size(); ++i) {
        const Sample& sample = validData[i];
        llm->reset();
        llm->response(sample.tokens, nullptr, nullptr, 0);
        auto outputs = llm->getOutputs();
        if (outputs.empty()) {
            return fail("LLM forward produced no outputs during validation");
        }
        const float* row = nullptr;
        int vocab = 0;
        if (!getLastLogitRow(outputs[0], vocabSize, llm->getValidLogitStart(), llm->getValidLogitSize(), &row, &vocab)) {
            return false;
        }
        if (clsIdx[0] < 0 || clsIdx[0] >= vocab || clsIdx[1] < 0 || clsIdx[1] >= vocab) {
            return fail("class token is out of logits range");
        }
        int pred = row[clsIdx[1]] > row[clsIdx[0]] ? 1 : 0;
        if (pred == sample.label) {
            correct++;
        }
        total++;
        auto scope = MNN::Express::ExecutorScope::Current();
        if (scope != nullptr) {
            scope->gc();
        }
    }
    *outAcc = total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
    return true;
}

static bool runFullPrefillLogits(Llm* llm, const Sample& sample, int vocabSize,
                                 std::vector<float>* row, VARP* logits, int* vocab) {
    llm->reset();
    llm->response(sample.tokens, nullptr, nullptr, 0);
    auto outputs = llm->getOutputs();
    if (outputs.empty()) {
        return fail("full prefill forward produced no outputs");
    }
    if (logits != nullptr) {
        *logits = outputs[0];
    }
    return copyLastLogitRow(outputs[0], vocabSize, llm->getValidLogitStart(), llm->getValidLogitSize(), row, vocab);
}

static bool collectTokenDecodeRows(Llm* llm, const Sample& sample, int vocabSize,
                                   std::vector<std::vector<float>>* rows,
                                   VARP* finalLogits, int* finalVocab) {
    if (sample.tokens.empty()) {
        return fail("cannot run token decode check on an empty sample");
    }
    rows->clear();
    rows->reserve(sample.tokens.size());
    llm->reset();
    llm->generate_init(nullptr, "\n");
    for (size_t pos = 0; pos < sample.tokens.size(); ++pos) {
        std::vector<int> oneToken(1, sample.tokens[pos]);
        VARP logits = llm->forward(oneToken, false);
        if (logits.get() == nullptr) {
            return fail("token decode forward returned null logits");
        }
        std::vector<float> row;
        int vocab = 0;
        if (!copyLastLogitRow(logits, vocabSize, llm->getValidLogitStart(), llm->getValidLogitSize(), &row, &vocab)) {
            return false;
        }
        if (!rows->empty() && row.size() != rows->front().size()) {
            return fail("token decode logits vocab changed across prefix positions");
        }
        rows->push_back(std::move(row));
        if (finalLogits != nullptr) {
            *finalLogits = logits;
        }
        if (finalVocab != nullptr) {
            *finalVocab = vocab;
        }
    }
    return true;
}

static bool runPrefixSweep(Llm* llm, const Sample& sample, int sampleOrdinal, int dataIndex,
                           int vocabSize, const std::vector<int>& classTokens,
                           const std::vector<std::vector<float>>& decodeRows) {
    for (size_t pos = 0; pos < sample.tokens.size(); ++pos) {
        Sample prefixSample = sample;
        prefixSample.tokens.assign(sample.tokens.begin(), sample.tokens.begin() + static_cast<std::ptrdiff_t>(pos + 1));
        std::vector<float> prefillRow;
        VARP prefillLogits;
        int prefillVocab = 0;
        if (!runFullPrefillLogits(llm, prefixSample, vocabSize, &prefillRow, &prefillLogits, &prefillVocab)) {
            return false;
        }
        LogitDiffStats stats;
        if (!computeLogitDiffStats(prefillRow, decodeRows[pos], sample.targetToken, &classTokens, &stats)) {
            return false;
        }
        std::cerr << "diag_prefix_diff"
                  << " sample_ordinal=" << sampleOrdinal
                  << " data_index=" << dataIndex
                  << " pos=" << pos
                  << " prefix_len=" << (pos + 1)
                  << " token=" << sample.tokens[pos]
                  << " token_decode=\"" << llm->tokenizer_decode(sample.tokens[pos]) << "\""
                  << " target=" << sample.targetToken
                  << " max_abs_diff=" << stats.maxAbsDiff
                  << " max_abs_token=" << stats.maxAbsToken
                  << " target_diff=" << stats.targetDiff
                  << " neg_diff=" << stats.negDiff
                  << " pos_diff=" << stats.posDiff
                  << " top1_prefill=" << stats.prefillTop1
                  << " top1_decode=" << stats.decodeTop1
                  << "\n";
    }
    return true;
}

static bool runPrefillDecodeCheck(Llm* llm, const std::vector<Sample>& trainData,
                                  const std::vector<int>& batchIndices, int vocabSize,
                                  const std::vector<int>& classTokens, const Args& args) {
    if (batchIndices.empty()) {
        return fail("diag_prefill_decode requires a non-empty warmup batch");
    }
    const int sampleCount = std::min(args.diagPrefillDecode, static_cast<int>(batchIndices.size()));
    std::cerr << "diag_prefill_decode: batch_size=" << batchIndices.size()
              << " samples=" << sampleCount
              << " force_full_causal_mask=" << (args.forceFullCausalMask ? "true" : "false")
              << " attention_mode=" << args.attentionMode << "\n";
    for (int i = 0; i < sampleCount; ++i) {
        const int dataIndex = batchIndices[i];
        if (dataIndex < 0 || dataIndex >= static_cast<int>(trainData.size())) {
            return fail("warmup batch index is out of range");
        }
        const Sample& sample = trainData[dataIndex];
        printSampleDebug("diag_prefill_decode", static_cast<size_t>(i), dataIndex, sample, llm);

        std::vector<float> prefillRow;
        VARP prefillLogits;
        int prefillVocab = 0;
        if (!runFullPrefillLogits(llm, sample, vocabSize, &prefillRow, &prefillLogits, &prefillVocab)) {
            return false;
        }
        printLogitDiagnostics("diag_prefill_full", prefillRow.data(), prefillVocab, sample.targetToken,
                              prefillLogits, llm->getValidLogitStart(), llm->getValidLogitSize(),
                              &classTokens, llm);

        std::vector<std::vector<float>> decodeRows;
        VARP decodeLogits;
        int decodeVocab = 0;
        if (!collectTokenDecodeRows(llm, sample, vocabSize, &decodeRows, &decodeLogits, &decodeVocab)) {
            return false;
        }
        const std::vector<float>& decodeRow = decodeRows.back();
        printLogitDiagnostics("diag_prefill_decode", decodeRow.data(), decodeVocab, sample.targetToken,
                              decodeLogits, llm->getValidLogitStart(), llm->getValidLogitSize(),
                              &classTokens, llm);

        LogitDiffStats stats;
        if (!computeLogitDiffStats(prefillRow, decodeRow, sample.targetToken, &classTokens, &stats)) {
            return false;
        }
        printLogitDiffDiagnostics("diag_prefill_decode_final", stats, prefillVocab, llm);

        if (i == 0) {
            if (!runPrefixSweep(llm, sample, i, dataIndex, vocabSize, classTokens, decodeRows)) {
                return false;
            }
        }
        auto scope = MNN::Express::ExecutorScope::Current();
        if (scope != nullptr) {
            scope->gc();
        }
    }
    return true;
}

static bool runLayerwiseCheck(Llm* llm, const std::vector<Sample>& trainData,
                              const std::vector<int>& batchIndices, int vocabSize,
                              const std::vector<int>& classTokens, const Args& args,
                              MnnActivationDebugState* activationDebug) {
    if (batchIndices.empty()) {
        return fail("diag_layerwise requires a non-empty warmup batch");
    }
    if (activationDebug == nullptr || !activationDebug->enabled || !activationDebug->layerwise) {
        return fail("diag_layerwise requires an active layerwise debug callback");
    }
    const int sampleCount = std::min(args.diagLayerwise, static_cast<int>(batchIndices.size()));
    std::cerr << "diag_layerwise"
              << " framework=mnn"
              << " batch_size=" << batchIndices.size()
              << " samples=" << sampleCount
              << " force_full_causal_mask=" << (args.forceFullCausalMask ? "true" : "false")
              << " attention_mode=" << args.attentionMode
              << " values=" << args.diagLayerwiseValues
              << " layers=";
    if (args.diagLayerwiseLayers.empty()) {
        std::cerr << "all";
    } else {
        for (size_t i = 0; i < args.diagLayerwiseLayers.size(); ++i) {
            if (i > 0) {
                std::cerr << ",";
            }
            std::cerr << args.diagLayerwiseLayers[i];
        }
    }
    std::cerr << "\n";

    for (int i = 0; i < sampleCount; ++i) {
        const int dataIndex = batchIndices[i];
        if (dataIndex < 0 || dataIndex >= static_cast<int>(trainData.size())) {
            return fail("warmup batch index is out of range");
        }
        const Sample& sample = trainData[dataIndex];
        printActivationSampleDebug("mnn", "warmup_B0", i, dataIndex, sample);
        llm->reset();
        beginMnnActivationForward(activationDebug, "warmup_B0", dataIndex, i,
                                  static_cast<int>(sample.tokens.size()),
                                  static_cast<int>(sample.tokens.empty() ? 0 : sample.tokens.size() - 1),
                                  sample.label, sample.targetToken);
        llm->response(sample.tokens, nullptr, nullptr, 0);
        endMnnActivationForward(activationDebug);
        auto outputs = llm->getOutputs();
        if (outputs.empty()) {
            return fail("diag_layerwise forward produced no outputs");
        }
        const float* row = nullptr;
        int vocab = 0;
        if (!getLastLogitRow(outputs[0], vocabSize, llm->getValidLogitStart(), llm->getValidLogitSize(),
                             &row, &vocab)) {
            return false;
        }
        printLogitDiagnostics("diag_layerwise", row, vocab, sample.targetToken, outputs[0],
                              llm->getValidLogitStart(), llm->getValidLogitSize(),
                              &classTokens, llm);
        auto scope = MNN::Express::ExecutorScope::Current();
        if (scope != nullptr) {
            scope->gc();
        }
    }
    return true;
}

static void fillSingleTargetLoraPerturb(const Metadata& meta, size_t targetIndex,
                                        float signedEpsilon, const Args& args,
                                        std::vector<float>* storage) {
    std::fill(storage->begin(), storage->end(), 0.0f);
    if (targetIndex >= meta.targets.size()) {
        return;
    }
    const auto& target = meta.targets[targetIndex];
    const int rank = target.shape.size() > 0 ? target.shape[0] : 0;
    const int out = target.shape.size() > 1 ? target.shape[1] : 0;
    for (size_t j = 0; j < target.size; ++j) {
        const float z = diagLoraPatternValue(target.canonicalKey, args.diagLoraPattern,
                                             args.seed, j, rank, out);
        (*storage)[bStorageIndex(target, j, args.bStorageLayout)] = signedEpsilon * z;
    }
}

static bool selectDiagLoraTargets(const Metadata& meta, const std::vector<std::string>& requested,
                                  const char* diagName, std::vector<size_t>* selectedTargets) {
    selectedTargets->clear();
    if (requested.empty()) {
        selectedTargets->reserve(meta.targets.size());
        for (size_t i = 0; i < meta.targets.size(); ++i) {
            selectedTargets->push_back(i);
        }
    } else {
        for (const auto& key : requested) {
            const int index = findTargetByCanonicalKey(meta, key);
            if (index < 0) {
                std::cerr << diagName << "_missing_target"
                          << " framework=mnn"
                          << " canonical_key=" << key
                          << "\n";
                continue;
            }
            const size_t uindex = static_cast<size_t>(index);
            if (std::find(selectedTargets->begin(), selectedTargets->end(), uindex) == selectedTargets->end()) {
                selectedTargets->push_back(uindex);
            }
        }
    }
    if (selectedTargets->empty()) {
        return fail(std::string(diagName) + " found no matching LoRA targets");
    }
    return true;
}

static bool runLoraACheck(Llm* llm, const Metadata& meta, const std::vector<Sample>& trainData,
                          const std::vector<int>& batchIndices, int vocabSize,
                          const std::vector<int>& classTokens, VARPS& bVars,
                          std::vector<float>* bStorage, const Args& args,
                          bool snapBToFp16, MnnActivationDebugState* activationDebug) {
    if (bStorage == nullptr) {
        return fail("diag_lora_a received null B storage");
    }
    if (batchIndices.empty()) {
        return fail("diag_lora_a requires a non-empty warmup batch");
    }
    if (activationDebug == nullptr || !activationDebug->enabled) {
        return fail("diag_lora_a requires an active MNN debug callback");
    }

    std::vector<size_t> selectedTargets;
    if (!selectDiagLoraTargets(meta, args.diagLoraTargets, "diag_lora_a", &selectedTargets)) {
        return false;
    }

    const int sampleCount = std::min(args.diagLoraA, static_cast<int>(batchIndices.size()));
    std::cerr << "diag_lora_a"
              << " framework=mnn"
              << " batch_size=" << batchIndices.size()
              << " samples=" << sampleCount
              << " targets=" << selectedTargets.size()
              << " values=" << args.diagLoraValues
              << " b_storage_layout=" << args.bStorageLayout
              << " b_param_precision=" << args.bParamPrecision
              << " attn_internals=" << (args.diagAttnInternals ? "true" : "false")
              << " attn_heads=" << joinIntList(args.diagAttnHeads)
              << " attn_values=" << args.diagAttnValues
              << "\n";

    std::fill(bStorage->begin(), bStorage->end(), 0.0f);
    if (!syncBVars(meta, *bStorage, bVars, snapBToFp16, args.bStorageLayout)) {
        return false;
    }

    for (int s = 0; s < sampleCount; ++s) {
        const int dataIndex = batchIndices[s];
        if (dataIndex < 0 || dataIndex >= static_cast<int>(trainData.size())) {
            return fail("warmup batch index is out of range");
        }
        const Sample& sample = trainData[dataIndex];
        printActivationSampleDebug("mnn", "diag_lora_a", s, dataIndex, sample);
        std::vector<int> singleSample(1, dataIndex);

        for (size_t selectedOrdinal = 0; selectedOrdinal < selectedTargets.size(); ++selectedOrdinal) {
            const size_t targetIndex = selectedTargets[selectedOrdinal];
            const auto& target = meta.targets[targetIndex];
            printMnnLoraTargetMetadataDebug(meta, targetIndex, s, dataIndex, "diag_lora_a_target");
            configureMnnActivationDebugForTarget(activationDebug, target, args.diagLoraValues, true);

            LossDebugState debug;
            debug.enabled = true;
            debug.sampleLimit = 1;
            debug.printEmbedding = false;
            debug.tag = "diag_lora_A";
            debug.classTokens = &classTokens;
            debug.llm = llm;
            debug.activationDebug = activationDebug;

            int64_t prefillUs = 0;
            int64_t prefillTokens = 0;
            double loss = 0.0;
            if (!evaluateBatchLoss(llm, trainData, singleSample, vocabSize,
                                   &prefillUs, &prefillTokens, &loss, &debug)) {
                return false;
            }
            std::cerr << "diag_lora_a_eval"
                      << " framework=mnn"
                      << " sample_ordinal=" << s
                      << " data_index=" << dataIndex
                      << " selected_ordinal=" << selectedOrdinal
                      << " target_index=" << targetIndex
                      << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
                      << " loss=" << loss
                      << " prefill_tokens=" << prefillTokens
                      << " prefill_us=" << prefillUs
                      << "\n";
        }
    }

    std::fill(bStorage->begin(), bStorage->end(), 0.0f);
    return syncBVars(meta, *bStorage, bVars, snapBToFp16, args.bStorageLayout);
}

static bool runLoraPerturbCheck(Llm* llm, const Metadata& meta, const std::vector<Sample>& trainData,
                                const std::vector<int>& batchIndices, int vocabSize,
                                const std::vector<int>& classTokens, VARPS& bVars,
                                std::vector<float>* bStorage, const Args& args,
                                bool snapBToFp16, MnnActivationDebugState* activationDebug,
                                bool branchMode = false) {
    if (bStorage == nullptr) {
        return fail(branchMode ? "diag_lora_branch received null B storage" :
                                 "diag_lora_perturb received null B storage");
    }
    if (batchIndices.empty()) {
        return fail(branchMode ? "diag_lora_branch requires a non-empty warmup batch" :
                                 "diag_lora_perturb requires a non-empty warmup batch");
    }

    const char* diagName = branchMode ? "diag_lora_branch" : "diag_lora_perturb";
    std::vector<size_t> selectedTargets;
    if (!selectDiagLoraTargets(meta, args.diagLoraTargets, diagName, &selectedTargets)) {
        return false;
    }

    const int requestedSamples = branchMode ? args.diagLoraBranch : args.diagLoraPerturb;
    const int sampleCount = std::min(requestedSamples, static_cast<int>(batchIndices.size()));
    std::cerr << diagName
              << " framework=mnn"
              << " batch_size=" << batchIndices.size()
              << " samples=" << sampleCount
              << " targets=" << selectedTargets.size()
              << " pattern=" << args.diagLoraPattern
              << " epsilon=" << args.epsilon
              << " values=" << args.diagLoraValues
              << " b_storage_layout=" << args.bStorageLayout
              << " b_param_precision=" << args.bParamPrecision
              << "\n";

    auto evalCurrentB = [&](const std::vector<int>& singleSample,
                            const Target& target,
                            size_t targetIndex,
                            int sampleOrdinal,
                            int dataIndex,
                            const Sample& sample,
                            const char* tag,
                            double* loss) -> bool {
        if (!syncBVars(meta, *bStorage, bVars, snapBToFp16, args.bStorageLayout)) {
            return false;
        }
        printMnnLoraPerturbBTargetDebug(meta, *bStorage, targetIndex, tag, args.diagLoraPattern,
                                        args.epsilon, args.bStorageLayout, snapBToFp16,
                                        args.diagLoraValues);
        if (args.diagBReadback) {
            if (!printBVarReadback(meta, *bStorage, bVars, snapBToFp16, tag, args.bStorageLayout)) {
                return false;
            }
        }
        if (activationDebug != nullptr && activationDebug->enabled) {
            configureMnnActivationDebugForTarget(activationDebug, target, args.diagLoraValues, branchMode,
                                                 branchMode && args.diagLoraPropagation);
        }
        LossDebugState debug;
        debug.enabled = true;
        debug.sampleLimit = 1;
        debug.printEmbedding = false;
        debug.tag = tag;
        debug.classTokens = &classTokens;
        debug.llm = llm;
        debug.activationDebug = activationDebug;
        int64_t prefillUs = 0;
        int64_t prefillTokens = 0;
        if (args.diagAttnInternals) {
            setMnnAttentionInternalDebugEnv(args, target, tag, dataIndex, sampleOrdinal, sample);
        }
        const bool ok = evaluateBatchLoss(llm, trainData, singleSample, vocabSize,
                                          &prefillUs, &prefillTokens, loss, &debug);
        if (args.diagAttnInternals) {
            clearMnnAttentionInternalDebugEnv();
        }
        if (!ok) {
            return false;
        }
        std::cerr << diagName << "_eval"
                  << " framework=mnn"
                  << " tag=" << tag
                  << " target_index=" << targetIndex
                  << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
                  << " loss=" << *loss
                  << " prefill_tokens=" << prefillTokens
                  << " prefill_us=" << prefillUs
                  << "\n";
        return true;
    };

    for (int s = 0; s < sampleCount; ++s) {
        const int dataIndex = batchIndices[s];
        if (dataIndex < 0 || dataIndex >= static_cast<int>(trainData.size())) {
            return fail("warmup batch index is out of range");
        }
        const Sample& sample = trainData[dataIndex];
        printActivationSampleDebug("mnn", diagName, s, dataIndex, sample);
        std::vector<int> singleSample(1, dataIndex);

        for (size_t selectedOrdinal = 0; selectedOrdinal < selectedTargets.size(); ++selectedOrdinal) {
            const size_t targetIndex = selectedTargets[selectedOrdinal];
            const auto& target = meta.targets[targetIndex];
            printMnnLoraTargetMetadataDebug(meta, targetIndex, s, dataIndex,
                                            branchMode ? "diag_lora_branch_target_meta" :
                                                         "diag_lora_perturb_target_meta");
            std::cerr << diagName << "_target"
                      << " framework=mnn"
                      << " sample_ordinal=" << s
                      << " data_index=" << dataIndex
                      << " selected_ordinal=" << selectedOrdinal
                      << " target_index=" << targetIndex
                      << " input_name=" << target.inputName
                      << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
                      << " target_op=" << (target.targetOp.empty() ? "<missing>" : target.targetOp)
                      << " shape_rank=" << target.shape[0]
                      << " shape_out=" << target.shape[1]
                      << " size=" << target.size
                      << "\n";

            double lossB0 = 0.0;
            double lossPlus = 0.0;
            double lossMinus = 0.0;

            std::fill(bStorage->begin(), bStorage->end(), 0.0f);
            if (!evalCurrentB(singleSample, target, targetIndex, s, dataIndex, sample,
                              "diag_lora_B0", &lossB0)) {
                return false;
            }

            fillSingleTargetLoraPerturb(meta, targetIndex, args.epsilon, args, bStorage);
            if (!evalCurrentB(singleSample, target, targetIndex, s, dataIndex, sample,
                              "diag_lora_plus", &lossPlus)) {
                return false;
            }

            fillSingleTargetLoraPerturb(meta, targetIndex, -args.epsilon, args, bStorage);
            if (!evalCurrentB(singleSample, target, targetIndex, s, dataIndex, sample,
                              "diag_lora_minus", &lossMinus)) {
                return false;
            }

            const double g = (lossPlus - lossMinus) / (2.0 * static_cast<double>(args.epsilon));
            std::cerr << diagName << "_summary"
                      << " framework=mnn"
                      << " sample_ordinal=" << s
                      << " data_index=" << dataIndex
                      << " target_index=" << targetIndex
                      << " canonical_key=" << (target.canonicalKey.empty() ? "<missing>" : target.canonicalKey)
                      << " pattern=" << args.diagLoraPattern
                      << " epsilon=" << args.epsilon
                      << " loss_b0=" << lossB0
                      << " loss_plus=" << lossPlus
                      << " loss_minus=" << lossMinus
                      << " plus_minus_delta=" << (lossPlus - lossMinus)
                      << " g=" << g
                      << "\n";
        }
    }

    std::fill(bStorage->begin(), bStorage->end(), 0.0f);
    return syncBVars(meta, *bStorage, bVars, snapBToFp16, args.bStorageLayout);
}

static bool readPssKbFromFile(const std::string& path, double* pssKb) {
    std::ifstream is(path.c_str());
    if (!is.is_open()) {
        return false;
    }
    std::string line;
    double totalKb = 0.0;
    bool found = false;
    while (std::getline(is, line)) {
        if (line.find("Pss:") != 0) {
            continue;
        }
        std::istringstream ss(line.substr(4));
        double kb = 0.0;
        ss >> kb;
        totalKb += kb;
        found = true;
    }
    if (!found) {
        return false;
    }
    *pssKb = totalKb;
    return true;
}

static double readCurrentPssMemoryMb() {
    double kb = 0.0;
    if (readPssKbFromFile("/proc/self/smaps_rollup", &kb)) {
        return kb / 1024.0;
    }
    if (readPssKbFromFile("/proc/self/smaps", &kb)) {
        return kb / 1024.0;
    }
    return 0.0;
}

class PssPeakSampler {
public:
    void start() {
        observeOnce();
        mWorker = std::thread([this]() {
            while (!mStop.load(std::memory_order_relaxed)) {
                observeOnce();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }

    double peakMb() {
        observeOnce();
        int64_t kb = mPeakKb.load(std::memory_order_relaxed);
        return kb >= 0 ? static_cast<double>(kb) / 1024.0 : 0.0;
    }

    double currentMb() {
        observeOnce();
        return readCurrentPssMemoryMb();
    }

    void shutdown() {
        mStop.store(true, std::memory_order_relaxed);
        if (mWorker.joinable()) {
            mWorker.join();
        }
    }

    ~PssPeakSampler() {
        shutdown();
    }

private:
    void observeOnce() {
        double mb = readCurrentPssMemoryMb();
        if (mb <= 0.0) {
            return;
        }
        int64_t kb = static_cast<int64_t>(mb * 1024.0);
        int64_t cur = mPeakKb.load(std::memory_order_relaxed);
        while (kb > cur && !mPeakKb.compare_exchange_weak(cur, kb, std::memory_order_relaxed)) {
        }
    }

    std::atomic<bool> mStop{false};
    std::atomic<int64_t> mPeakKb{-1};
    std::thread mWorker;
};

static double readPeakMemoryMb() {
    std::ifstream is("/proc/self/status");
    if (!is.is_open()) {
        return 0.0;
    }
    std::string line;
    while (std::getline(is, line)) {
        if (line.find("VmHWM:") != 0) {
            continue;
        }
        std::istringstream ss(line.substr(6));
        double kb = 0.0;
        ss >> kb;
        return kb / 1024.0;
    }
    return 0.0;
}

static int resolveThreadCount(int threads) {
    if (threads > 0) {
        return threads;
    }
    unsigned int autoThreads = std::thread::hardware_concurrency();
    return autoThreads > 0 ? static_cast<int>(autoThreads) : 4;
}

static std::string runtimeConfigJson(int threads, const std::string& precision, const std::string& memory,
                                     int attentionMode, bool enableDebug,
                                     bool forceFullCausalMask, bool forceFloatWeight,
                                     bool compactNormalWeight, bool mixedFp32Attention) {
    std::ostringstream os;
    // memory=high avoids MNN's KleidiAI / quant-weight paths for the
    // convergence-sensitive FP16 baseline. Passing memory=low/normal is useful
    // when explicitly benchmarking packed low-bit weight execution.
    //
    // The old hard-coded memory=high (was low) avoided MNN's INT8 GEMM path which would
    // re-quantize FP16 weights to INT8 at runtime. The INT8 quantization is
    // saving memory + speed but injects ~1e-3 quantization noise per forward,
    // which floors ZO's gradient estimate and prevents convergence.
    os << "{"
       << "\"backend_type\":\"cpu\","
       << "\"thread_num\":" << threads << ","
       << "\"precision\":\"" << precision << "\","
       << "\"memory\":\"" << memory << "\","
       << "\"attention_mode\":" << attentionMode << ","
       << "\"all_logits\":false,"
       << "\"reuse_kv\":false,"
       << "\"async\":false,"
       << "\"use_mmap\":false,"
       << "\"use_cached_mmap\":false,"
       << "\"use_template\":false,"
       << "\"force_full_causal_mask\":" << (forceFullCausalMask ? "true" : "false") << ","
       << "\"force_float_weight\":" << (forceFloatWeight ? "true" : "false") << ","
       << "\"compact_normal_weight\":" << (compactNormalWeight ? "true" : "false") << ","
       << "\"mixed_fp32_attention\":" << (mixedFp32Attention ? "true" : "false") << ","
       << "\"mixed_fp32_attention_context\":" << (mixedFp32Attention ? "true" : "false") << ","
       << "\"enable_debug\":" << (enableDebug ? "true" : "false")
       << "}";
    return os.str();
}

static void fillBatchIndices(std::vector<int>* batchIndices, int trainSize, int batch, int step, int seed) {
    batchIndices->clear();
    batchIndices->reserve(batch);
    std::mt19937 rng(deriveStepSeed(static_cast<uint32_t>(seed), step, 0x44415441U));
    std::uniform_int_distribution<int> dist(0, trainSize - 1);
    for (int b = 0; b < batch; ++b) {
        batchIndices->push_back(dist(rng));
    }
}

static void fillNoiseForStep(const Metadata& meta, int step, int seed, std::vector<float>* z) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (size_t t = 0; t < meta.targets.size(); ++t) {
        const auto& target = meta.targets[t];
        std::mt19937 rng(deriveTensorNoiseSeed(seed, step, t));
        for (size_t i = 0; i < target.size; ++i) {
            (*z)[target.offset + i] = normal(rng);
        }
    }
}

static bool runZoCheck(Llm* llm, const Metadata& meta, const std::vector<Sample>& trainData,
                       const std::vector<int>& batchIndices, int vocabSize, VARPS& bVars,
                       std::vector<float>* bStorage, const Args& args, bool snapBToFp16) {
    if (bStorage == nullptr) {
        return fail("diag_zo_check received null B storage");
    }
    std::vector<float> z(meta.totalBSize, 0.0f);
    std::vector<float> bWork(meta.totalBSize, 0.0f);
    fillNoiseForStep(meta, 0, args.seed, &z);

    auto evalWithB = [&](const std::vector<float>& values, const char* tag, double* loss) -> bool {
        if (!syncBVars(meta, values, bVars, snapBToFp16, args.bStorageLayout)) {
            return false;
        }
        if (args.diagBReadback) {
            if (!printBVarReadback(meta, values, bVars, snapBToFp16, tag, args.bStorageLayout)) {
                return false;
            }
        }
        int64_t prefillUs = 0;
        int64_t prefillTokens = 0;
        if (!evaluateBatchLoss(llm, trainData, batchIndices, vocabSize, &prefillUs, &prefillTokens, loss, nullptr)) {
            return false;
        }
        std::cerr << "diag_zo_check_eval tag=" << tag
                  << " loss=" << *loss
                  << " prefill_tokens=" << prefillTokens
                  << " prefill_us=" << prefillUs << "\n";
        return true;
    };

    double b0A = 0.0;
    double b0B = 0.0;
    double plusA = 0.0;
    double minusA = 0.0;
    double plusB = 0.0;
    double minusB = 0.0;
    std::fill(bStorage->begin(), bStorage->end(), 0.0f);
    if (!evalWithB(*bStorage, "B0_a", &b0A) ||
        !evalWithB(*bStorage, "B0_b", &b0B)) {
        return false;
    }

    for (size_t i = 0; i < z.size(); ++i) {
        bWork[i] = (*bStorage)[i] + args.epsilon * z[i];
    }
    if (!evalWithB(bWork, "plus_a", &plusA)) {
        return false;
    }
    for (size_t i = 0; i < z.size(); ++i) {
        bWork[i] = (*bStorage)[i] - args.epsilon * z[i];
    }
    if (!evalWithB(bWork, "minus_a", &minusA)) {
        return false;
    }
    for (size_t i = 0; i < z.size(); ++i) {
        bWork[i] = (*bStorage)[i] + args.epsilon * z[i];
    }
    if (!evalWithB(bWork, "plus_b", &plusB)) {
        return false;
    }
    for (size_t i = 0; i < z.size(); ++i) {
        bWork[i] = (*bStorage)[i] - args.epsilon * z[i];
    }
    if (!evalWithB(bWork, "minus_b", &minusB)) {
        return false;
    }

    double g = (plusA - minusA) / (2.0 * static_cast<double>(args.epsilon));
    std::vector<float> bNew(meta.totalBSize, 0.0f);
    std::vector<float> bOpp(meta.totalBSize, 0.0f);
    for (size_t i = 0; i < z.size(); ++i) {
        bNew[i] = (*bStorage)[i] - static_cast<float>(args.lr * g * z[i]);
        bOpp[i] = (*bStorage)[i] + static_cast<float>(args.lr * g * z[i]);
    }

    double newLoss = 0.0;
    double oppLoss = 0.0;
    if (!evalWithB(bNew, "update_new", &newLoss) ||
        !evalWithB(bOpp, "update_opposite", &oppLoss)) {
        return false;
    }

    double bSumAbs = 0.0;
    double bSumSq = 0.0;
    float bMaxAbs = 0.0f;
    summarizeB(bNew, &bSumAbs, &bSumSq, &bMaxAbs);
    double bNewRms = bNew.empty() ? 0.0 : std::sqrt(bSumSq / static_cast<double>(bNew.size()));
    std::cerr << "diag_zo_check_summary"
              << " b0_repeat_abs_diff=" << std::fabs(b0A - b0B)
              << " plus_repeat_abs_diff=" << std::fabs(plusA - plusB)
              << " minus_repeat_abs_diff=" << std::fabs(minusA - minusB)
              << " plus_minus_delta=" << (plusA - minusA)
              << " g=" << g
              << " update_new_loss=" << newLoss
              << " update_opposite_loss=" << oppLoss
              << " update_new_minus_b0=" << (newLoss - b0A)
              << " update_opposite_minus_b0=" << (oppLoss - b0A)
              << " update_new_b_rms=" << bNewRms
              << " update_new_b_max_abs=" << bMaxAbs
              << "\n";

    std::fill(bStorage->begin(), bStorage->end(), 0.0f);
    return syncBVars(meta, *bStorage, bVars, snapBToFp16, args.bStorageLayout);
}

static bool runRepeatCheck(Llm* llm, const Metadata& meta, const std::vector<Sample>& trainData,
                           const std::vector<int>& batchIndices, int vocabSize, VARPS& bVars,
                           std::vector<float>* bStorage, const Args& args, bool snapBToFp16) {
    if (bStorage == nullptr) {
        return fail("diag_repeat received null B storage");
    }
    int repeat = std::max(1, args.diagRepeat);
    std::vector<float> z(meta.totalBSize, 0.0f);
    std::vector<float> bWork(meta.totalBSize, 0.0f);
    fillNoiseForStep(meta, 0, args.seed, &z);

    auto repeatWithB = [&](const std::vector<float>& values, const char* tag) -> bool {
        if (!syncBVars(meta, values, bVars, snapBToFp16, args.bStorageLayout)) {
            return false;
        }
        if (args.diagBReadback) {
            if (!printBVarReadback(meta, values, bVars, snapBToFp16, tag, args.bStorageLayout)) {
                return false;
            }
        }
        double firstLoss = 0.0;
        double minLoss = std::numeric_limits<double>::infinity();
        double maxLoss = -std::numeric_limits<double>::infinity();
        for (int r = 0; r < repeat; ++r) {
            int64_t prefillUs = 0;
            int64_t prefillTokens = 0;
            double loss = 0.0;
            if (!evaluateBatchLoss(llm, trainData, batchIndices, vocabSize, &prefillUs, &prefillTokens, &loss, nullptr)) {
                return false;
            }
            if (r == 0) {
                firstLoss = loss;
            }
            minLoss = std::min(minLoss, loss);
            maxLoss = std::max(maxLoss, loss);
            std::cerr << "diag_repeat_eval"
                      << " tag=" << tag
                      << " repeat=" << r
                      << " loss=" << loss
                      << " diff_from_first=" << (loss - firstLoss)
                      << " prefill_tokens=" << prefillTokens
                      << " prefill_us=" << prefillUs << "\n";
        }
        std::cerr << "diag_repeat_summary"
                  << " tag=" << tag
                  << " repeat=" << repeat
                  << " first_loss=" << firstLoss
                  << " min_loss=" << minLoss
                  << " max_loss=" << maxLoss
                  << " range=" << (maxLoss - minLoss)
                  << "\n";
        return true;
    };

    std::fill(bStorage->begin(), bStorage->end(), 0.0f);
    if (!repeatWithB(*bStorage, "B0")) {
        return false;
    }
    for (size_t i = 0; i < z.size(); ++i) {
        bWork[i] = (*bStorage)[i] + args.epsilon * z[i];
    }
    if (!repeatWithB(bWork, "plus")) {
        return false;
    }
    for (size_t i = 0; i < z.size(); ++i) {
        bWork[i] = (*bStorage)[i] - args.epsilon * z[i];
    }
    if (!repeatWithB(bWork, "minus")) {
        return false;
    }
    std::fill(bStorage->begin(), bStorage->end(), 0.0f);
    return syncBVars(meta, *bStorage, bVars, snapBToFp16, args.bStorageLayout);
}

int main(int argc, const char* argv[]) {
    Args args;
    if (!parseArgs(argc, argv, &args)) {
        printUsage(argv[0]);
        return 1;
    }

    Metadata meta;
    if (!loadMetadata(args.loraMeta, &meta)) {
        return 1;
    }
    std::string tokenizerFile = readJsonString(args.config, "tokenizer_file");
    std::string tokenizerPath = joinPath(dirname(args.config), tokenizerFile);
    if (!tokenizerFile.empty() && endsWith(tokenizerFile, ".txt")) {
        std::cerr << "warning: config uses tokenizer_file=" << tokenizerFile
                  << "; for TinyLlama SST2, tokenizer.mtok is preferred for llama.cpp parity\n";
    }
    PssPeakSampler pssSampler;
    pssSampler.start();

    std::vector<std::string> extraInputNames;
    extraInputNames.reserve(meta.targets.size());
    for (size_t i = 0; i < meta.targets.size(); ++i) {
        extraInputNames.push_back(meta.targets[i].inputName);
    }

    std::unique_ptr<Llm, void (*)(Llm*)> llm(Llm::createLLM(args.config), Llm::destroy);
    if (!llm) {
        fail("failed to create LLM");
        return 1;
    }
    int runtimeThreads = resolveThreadCount(args.threads);
    if (args.verbose || args.threads <= 0) {
        std::cout << "threads=" << runtimeThreads << "\n";
    }
    const bool mixedFp32Attention = args.precision == "normal";
    const bool mixedFp32AttentionScore = mixedFp32Attention;
    const bool mixedFp32AttentionContext = mixedFp32Attention;
    const std::string effectiveBackendPrecision = mixedFp32Attention ? "low" : args.precision;
    const bool anyDiagLora = args.diagLoraPerturb > 0 || args.diagLoraA > 0 || args.diagLoraBranch > 0;
    if (args.verbose || args.debugMNN || args.diagLayerwise > 0 || anyDiagLora) {
        std::cerr << "runtime_precision=" << args.precision
                  << " effective_backend_precision=" << effectiveBackendPrecision
                  << " mixed_fp32_attention=" << (mixedFp32Attention ? "true" : "false")
                  << " mixed_fp32_attention_score=" << (mixedFp32AttentionScore ? "true" : "false")
                  << " mixed_fp32_attention_context=" << (mixedFp32AttentionContext ? "true" : "false") << "\n";
        std::cerr << "b_param_precision=" << args.bParamPrecision
                  << " b_storage_layout=" << args.bStorageLayout
                  << " attention_mode=" << args.attentionMode
                  << " runtime_memory=" << args.runtimeMemory
                  << " force_full_causal_mask=" << (args.forceFullCausalMask ? "true" : "false")
                  << " force_float_weight=" << (args.forceFloatWeight ? "true" : "false")
                  << " compact_normal_weight=" << (args.compactNormalWeight ? "true" : "false")
                  << " debug_MNN=" << (args.debugMNN ? "true" : "false") << "\n";
        std::cerr << "diag_repeat=" << args.diagRepeat
                  << " diag_b_readback=" << (args.diagBReadback ? "true" : "false")
                  << " baseline_step=" << args.baselineStep
                  << " diag_prefill_decode=" << args.diagPrefillDecode
                  << " diag_layerwise=" << args.diagLayerwise
                  << " diag_layerwise_values=" << args.diagLayerwiseValues
                  << " diag_lora_perturb=" << args.diagLoraPerturb
                  << " diag_lora_a=" << args.diagLoraA
                  << " diag_lora_branch=" << args.diagLoraBranch
                  << " diag_lora_propagation=" << (args.diagLoraPropagation ? "true" : "false")
                  << " diag_attn_internals=" << (args.diagAttnInternals ? "true" : "false")
                  << " diag_attn_values=" << args.diagAttnValues
                  << " diag_attn_heads=" << joinIntList(args.diagAttnHeads)
                  << " diag_lora_values=" << args.diagLoraValues
                  << " diag_lora_pattern=" << args.diagLoraPattern << "\n";
        std::cerr << "config_tokenizer_file=" << (tokenizerFile.empty() ? "<missing>" : tokenizerFile)
                  << " tokenizer_exists=" << (fileExists(tokenizerPath) ? "true" : "false") << "\n";
        std::string mtokPath = joinPath(dirname(args.config), "tokenizer.mtok");
        std::cerr << "tokenizer_mtok_exists=" << (fileExists(mtokPath) ? "true" : "false") << "\n";
    }
    llm->setExtraInputNames(extraInputNames);
    const bool enableMnnDebugRuntime = args.debugMNN || args.diagLayerwise > 0 || anyDiagLora;
    llm->set_config(runtimeConfigJson(runtimeThreads, args.precision, args.runtimeMemory,
                                      args.attentionMode, enableMnnDebugRuntime,
                                      args.forceFullCausalMask, args.forceFloatWeight,
                                      args.compactNormalWeight, mixedFp32Attention));

    MnnActivationDebugState activationDebug;
    if (enableMnnDebugRuntime) {
        configureMnnActivationDebug(&activationDebug, meta, args.diagLayerwise > 0,
                                    args.diagLayerwiseLayers, args.diagLayerwiseValues);
        installMnnActivationDebugCallbacks(llm.get(), &activationDebug);
    }

    VARPS bVars = createBVars(meta);
    std::vector<float> bStorage(meta.totalBSize, 0.0f);
    const bool snapBToFp16 = args.bParamPrecision == "fp16";
    if (!syncBVars(meta, bStorage, bVars, snapBToFp16, args.bStorageLayout)) {
        return 1;
    }
    if (args.diagBReadback) {
        if (!printBVarReadback(meta, bStorage, bVars, snapBToFp16, "initial_B0", args.bStorageLayout)) {
            return 1;
        }
    }
    if (args.debugMNN || anyDiagLora) {
        printTargetOrderDebug(meta, args.seed, 0);
        printBStorageDebug(meta, bStorage, "initial_master_B0", args.bStorageLayout, snapBToFp16, args.seed, 0);
    }
    llm->setExtraInputs(bVars);

    if (!llm->load()) {
        fail("failed to load LLM");
        return 1;
    }

    std::vector<int> clsIdx;
    std::vector<std::vector<int>> labelSuffixes;
    if (!getLabelSuffixes(llm.get(), meta.vocabSize, &labelSuffixes, &clsIdx)) {
        return 1;
    }

    bool doEval = args.evalStep != -1;
    std::vector<Sample> trainData;
    std::vector<Sample> validData;
    if (!loadSst2Tsv(args.trainData, llm.get(), 1000, args.seqLen, labelSuffixes, &trainData)) {
        return 1;
    }
    if (doEval) {
        if (!loadSst2Tsv(args.validData, llm.get(), 1000, args.seqLen, labelSuffixes, &validData)) {
            return 1;
        }
    }
    std::cerr << "train_samples=" << trainData.size();
    if (doEval) {
        std::cerr << " valid_samples=" << validData.size();
    } else {
        std::cerr << " validation=disabled";
    }
    std::cerr << " max_seq_len=" << args.seqLen << "\n";
    if (args.verbose || args.debugMNN || args.diagLayerwise > 0 || anyDiagLora) {
        std::cerr << "lora_targets=" << meta.targets.size()
                  << " rank=" << meta.rank
                  << " alpha=" << meta.alpha
                  << " total_b_size=" << meta.totalBSize << "\n";
        printTokenizerDiagnostics(llm.get(), clsIdx, labelSuffixes, trainData);
    }

    std::cout << std::fixed << std::setprecision(3);

    std::vector<int> warmupBatch;
    fillBatchIndices(&warmupBatch, static_cast<int>(trainData.size()), args.batch, 0, args.seed);
    if (args.diagLayerwise > 0) {
        return runLayerwiseCheck(llm.get(), trainData, warmupBatch, meta.vocabSize,
                                 clsIdx, args, &activationDebug) ? 0 : 1;
    }
    if (args.diagLoraA > 0) {
        return runLoraACheck(llm.get(), meta, trainData, warmupBatch, meta.vocabSize,
                             clsIdx, bVars, &bStorage, args, snapBToFp16,
                             &activationDebug) ? 0 : 1;
    }
    if (args.diagLoraBranch > 0) {
        return runLoraPerturbCheck(llm.get(), meta, trainData, warmupBatch, meta.vocabSize,
                                   clsIdx, bVars, &bStorage, args, snapBToFp16,
                                   &activationDebug, true) ? 0 : 1;
    }
    if (args.diagLoraPerturb > 0) {
        return runLoraPerturbCheck(llm.get(), meta, trainData, warmupBatch, meta.vocabSize,
                                   clsIdx, bVars, &bStorage, args, snapBToFp16,
                                   &activationDebug) ? 0 : 1;
    }

    int64_t warmupPrefillUs = 0;
    int64_t warmupPrefillTokens = 0;
    double warmupLoss = 0.0;
    LossDebugState warmupDebug;
    warmupDebug.enabled = args.verbose || args.debugMNN;
    warmupDebug.sampleLimit = args.debugMNN ? std::min(args.batch, 2) : 1;
    warmupDebug.printEmbedding = args.debugMNN;
    warmupDebug.tag = "warmup_B0";
    warmupDebug.classTokens = &clsIdx;
    warmupDebug.llm = llm.get();
    warmupDebug.activationDebug = args.debugMNN ? &activationDebug : nullptr;
    if (!evaluateBatchLoss(llm.get(), trainData, warmupBatch, meta.vocabSize,
                           &warmupPrefillUs, &warmupPrefillTokens, &warmupLoss, &warmupDebug)) {
        return 1;
    }
    std::cerr << "warmup_loss=" << warmupLoss
              << " warmup_prefill_tokens=" << warmupPrefillTokens
              << " warmup_prefill_us=" << warmupPrefillUs << "\n";

    if (args.diagPrefillDecode > 0) {
        return runPrefillDecodeCheck(llm.get(), trainData, warmupBatch, meta.vocabSize,
                                     clsIdx, args) ? 0 : 1;
    }

    if (args.diagRepeat > 0) {
        std::cerr << "diag_repeat: batch_size=" << warmupBatch.size()
                  << " repeat=" << args.diagRepeat
                  << " epsilon=" << args.epsilon
                  << " seed=" << args.seed
                  << " b_param_precision=" << args.bParamPrecision
                  << " b_storage_layout=" << args.bStorageLayout
                  << " attention_mode=" << args.attentionMode << "\n";
        return runRepeatCheck(llm.get(), meta, trainData, warmupBatch, meta.vocabSize,
                              bVars, &bStorage, args, snapBToFp16) ? 0 : 1;
    }

    if (args.diagZoCheck) {
        std::cerr << "diag_zo_check: batch_size=" << warmupBatch.size()
                  << " epsilon=" << args.epsilon
                  << " lr=" << args.lr
                  << " seed=" << args.seed
                  << " b_param_precision=" << args.bParamPrecision
                  << " b_storage_layout=" << args.bStorageLayout
                  << " attention_mode=" << args.attentionMode << "\n";
        return runZoCheck(llm.get(), meta, trainData, warmupBatch, meta.vocabSize,
                          bVars, &bStorage, args, snapBToFp16) ? 0 : 1;
    }

    // --diag-isolate-b N: for each of N selected target indices, perturb ONLY that
    // target's B (to a uniform constant), zero everything else, run a forward, and
    // log the resulting loss. If the B-input-to-graph binding is correct, every
    // target should produce a DIFFERENT loss. Duplicates → wrong layer is being
    // perturbed (binding bug). Same loss as baseline → that B isn't connected to
    // the graph at all.
    if (args.diagIsolateB > 0) {
        const int total = static_cast<int>(meta.targets.size());
        const int n = std::min(args.diagIsolateB, total);
        std::cerr << "diag_isolate_b: running " << n << " single-target B perturbations"
                  << " (value=" << args.diagIsolateValue
                  << ", total_targets=" << total << ")\n";

        // First, confirm the in-tool data structures are consistent: print each
        // bVar's setName vs meta.targets[i].inputName side-by-side for the
        // selected indices. Mismatch here means the tool's own bookkeeping is
        // wrong before we even talk to MNN.
        for (int k = 0; k < n; ++k) {
            const int t = (n >= total) ? k : (k * total / n);
            const auto* info = bVars[t]->getInfo();
            std::cerr << "diag_isolate_b_check"
                      << " k=" << k
                      << " target_index=" << t
                      << " meta_name=" << meta.targets[t].inputName
                      << " varp_dims=[";
            if (info != nullptr) {
                for (size_t d = 0; d < info->dim.size(); ++d) {
                    if (d > 0) std::cerr << ",";
                    std::cerr << info->dim[d];
                }
                std::cerr << "] varp_size=" << info->size;
            } else {
                std::cerr << "]";
            }
            std::cerr << " expected_shape=[" << meta.targets[t].shape[0]
                      << "," << meta.targets[t].shape[1] << "]"
                      << " offset=" << meta.targets[t].offset
                      << " size=" << meta.targets[t].size << "\n";
        }

        // Baseline forward (all B = 0). Use just the FIRST sample of the warmup
        // batch so per-sample loss is comparable across isolations.
        std::vector<int> singleSample(1, warmupBatch.empty() ? 0 : warmupBatch[0]);

        std::fill(bStorage.begin(), bStorage.end(), 0.0f);
        if (!syncBVars(meta, bStorage, bVars, snapBToFp16, args.bStorageLayout)) {
            return 1;
        }
        int64_t diagPrefillUs = 0;
        int64_t diagPrefillTokens = 0;
        double baselineLoss = 0.0;
        if (!evaluateBatchLoss(llm.get(), trainData, singleSample, meta.vocabSize,
                               &diagPrefillUs, &diagPrefillTokens, &baselineLoss, nullptr)) {
            return 1;
        }
        std::cerr << "diag_isolate_b_baseline sample_idx=" << singleSample[0]
                  << " loss=" << baselineLoss << "\n";

        // Per-target isolated perturbations.
        std::vector<double> isolatedLosses;
        std::vector<int> isolatedTargets;
        for (int k = 0; k < n; ++k) {
            const int t = (n >= total) ? k : (k * total / n);
            std::fill(bStorage.begin(), bStorage.end(), 0.0f);
            const auto& target = meta.targets[t];
            for (size_t j = 0; j < target.size; ++j) {
                bStorage[target.offset + j] = args.diagIsolateValue;
            }
            if (!syncBVars(meta, bStorage, bVars, snapBToFp16, args.bStorageLayout)) {
                return 1;
            }
            double isoLoss = 0.0;
            if (!evaluateBatchLoss(llm.get(), trainData, singleSample, meta.vocabSize,
                                   &diagPrefillUs, &diagPrefillTokens, &isoLoss, nullptr)) {
                return 1;
            }
            isolatedLosses.push_back(isoLoss);
            isolatedTargets.push_back(t);
            std::cerr << "diag_isolate_b_result"
                      << " k=" << k
                      << " target_index=" << t
                      << " input_name=" << target.inputName
                      << " loss=" << isoLoss
                      << " delta_from_baseline=" << (isoLoss - baselineLoss) << "\n";
        }

        // Summary: count distinct losses and look for duplicates.
        int distinct = 0;
        int duplicates = 0;
        const double tol = 1e-6;
        for (size_t a = 0; a < isolatedLosses.size(); ++a) {
            bool isFirst = true;
            for (size_t b = 0; b < a; ++b) {
                if (std::fabs(isolatedLosses[a] - isolatedLosses[b]) < tol) {
                    isFirst = false;
                    std::cerr << "diag_isolate_b_DUPLICATE target_a=" << isolatedTargets[a]
                              << " (name=" << meta.targets[isolatedTargets[a]].inputName << ")"
                              << " target_b=" << isolatedTargets[b]
                              << " (name=" << meta.targets[isolatedTargets[b]].inputName << ")"
                              << " loss=" << isolatedLosses[a] << "\n";
                    duplicates++;
                    break;
                }
            }
            if (isFirst) distinct++;
        }
        int matchesBaseline = 0;
        for (size_t a = 0; a < isolatedLosses.size(); ++a) {
            if (std::fabs(isolatedLosses[a] - baselineLoss) < tol) {
                std::cerr << "diag_isolate_b_NO_EFFECT target_index=" << isolatedTargets[a]
                          << " (name=" << meta.targets[isolatedTargets[a]].inputName << ")"
                          << " loss equals baseline=" << baselineLoss << "\n";
                matchesBaseline++;
            }
        }
        std::cerr << "diag_isolate_b_summary"
                  << " n_tested=" << isolatedLosses.size()
                  << " distinct=" << distinct
                  << " duplicates=" << duplicates
                  << " matches_baseline=" << matchesBaseline << "\n";

        // Restore B = 0 for any subsequent code paths, even though we're exiting.
        std::fill(bStorage.begin(), bStorage.end(), 0.0f);
        if (!syncBVars(meta, bStorage, bVars, snapBToFp16, args.bStorageLayout)) {
            return 1;
        }
        return 0;
    }

    if (doEval) {
        double initAcc = 0.0;
        if (!evaluateAccuracy(llm.get(), validData, clsIdx, meta.vocabSize, &initAcc)) {
            return 1;
        }
        std::cout << "initial_accuracy=" << initAcc << "\n";
        std::cout << "pss_memory_mb=" << pssSampler.currentMb() << "\n";
    }

    std::vector<int> batchIndices;
    std::vector<float> z(meta.totalBSize, 0.0f);
    // Match llama.cpp's master/work split: bStorage is the pristine FP32 master
    // (never touched between plus/minus forwards), bWork is the perturbed copy
    // that gets snapped to FP16 and pushed to the model. Avoids cumulative FP32
    // round-off from the previous in-place += εz / -= 2εz / += εz pattern, and
    // keeps the master update at the end a single clean B -= lr*g*z.
    std::vector<float> bWork(meta.totalBSize, 0.0f);
    std::vector<double> stepTimesMs;
    stepTimesMs.reserve(args.steps);
    int64_t totalPrefillUs = 0;
    int64_t totalPrefillTokens = 0;
    double trainTimeS = 0.0;

    for (int step = 0; step < args.steps; ++step) {
        fillBatchIndices(&batchIndices, static_cast<int>(trainData.size()), args.batch, step, args.seed);

        auto stepStart = std::chrono::steady_clock::now();
        fillNoiseForStep(meta, step, args.seed, &z);

        // Plus: work = master + ε·z
        for (size_t i = 0; i < z.size(); ++i) {
            bWork[i] = bStorage[i] + args.epsilon * z[i];
        }
        if (!syncBVars(meta, bWork, bVars, snapBToFp16, args.bStorageLayout)) {
            return 1;
        }
        if (args.diagBReadback && step == 0) {
            if (!printBVarReadback(meta, bWork, bVars, snapBToFp16, "step1_plus", args.bStorageLayout)) {
                return 1;
            }
        }
        if (args.debugMNN && step == 0) {
            printBStorageDebug(meta, bWork, "step1_plus_work", args.bStorageLayout, snapBToFp16, args.seed, step);
        }

        double lossPlus = 0.0;
        LossDebugState plusDebug;
        plusDebug.enabled = (args.verbose || args.debugMNN) && step == 0;
        plusDebug.sampleLimit = args.debugMNN ? std::min(args.batch, 2) : 1;
        plusDebug.printEmbedding = args.debugMNN;
        plusDebug.tag = "step1_plus";
        plusDebug.classTokens = &clsIdx;
        plusDebug.llm = llm.get();
        plusDebug.activationDebug = args.debugMNN ? &activationDebug : nullptr;
        if (!evaluateBatchLoss(llm.get(), trainData, batchIndices, meta.vocabSize,
                               &totalPrefillUs, &totalPrefillTokens, &lossPlus, &plusDebug)) {
            return 1;
        }

        // Minus: work = master - ε·z (master untouched)
        for (size_t i = 0; i < z.size(); ++i) {
            bWork[i] = bStorage[i] - args.epsilon * z[i];
        }
        if (!syncBVars(meta, bWork, bVars, snapBToFp16, args.bStorageLayout)) {
            return 1;
        }
        if (args.diagBReadback && step == 0) {
            if (!printBVarReadback(meta, bWork, bVars, snapBToFp16, "step1_minus", args.bStorageLayout)) {
                return 1;
            }
        }
        if (args.debugMNN && step == 0) {
            printBStorageDebug(meta, bWork, "step1_minus_work", args.bStorageLayout, snapBToFp16, args.seed, step);
        }

        double lossMinus = 0.0;
        LossDebugState minusDebug;
        minusDebug.enabled = (args.verbose || args.debugMNN) && step == 0;
        minusDebug.sampleLimit = args.debugMNN ? std::min(args.batch, 2) : 1;
        minusDebug.printEmbedding = args.debugMNN;
        minusDebug.tag = "step1_minus";
        minusDebug.classTokens = &clsIdx;
        minusDebug.llm = llm.get();
        minusDebug.activationDebug = args.debugMNN ? &activationDebug : nullptr;
        if (!evaluateBatchLoss(llm.get(), trainData, batchIndices, meta.vocabSize,
                               &totalPrefillUs, &totalPrefillTokens, &lossMinus, &minusDebug)) {
            return 1;
        }

        // Update master: master -= lr·g·z
        double g = (lossPlus - lossMinus) / (2.0 * static_cast<double>(args.epsilon));
        for (size_t i = 0; i < z.size(); ++i) {
            bStorage[i] -= static_cast<float>(args.lr * g * z[i]);
        }
        // Push the new master to the model so the next step's first forward
        // (before its own +εz) would see the correct baseline if needed.
        if (!syncBVars(meta, bStorage, bVars, snapBToFp16, args.bStorageLayout)) {
            return 1;
        }
        if (args.debugMNN && step == 0) {
            printBStorageDebug(meta, bStorage, "step1_master_after_update", args.bStorageLayout, snapBToFp16, args.seed, step);
        }

        double baselineLoss = std::numeric_limits<double>::quiet_NaN();
        bool printedBaselineLoss = false;
        if (args.baselineStep > 0 && (((step + 1) % args.baselineStep == 0) || step == 0 || (step + 1 == args.steps))) {
            int64_t baselinePrefillUs = 0;
            int64_t baselinePrefillTokens = 0;
            if (!evaluateBatchLoss(llm.get(), trainData, batchIndices, meta.vocabSize,
                                   &baselinePrefillUs, &baselinePrefillTokens, &baselineLoss, nullptr)) {
                return 1;
            }
            printedBaselineLoss = true;
        }

        auto stepEnd = std::chrono::steady_clock::now();
        double stepMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
        trainTimeS += stepMs / 1000.0;
        stepTimesMs.push_back(stepMs);

        if (args.verbose) {
            double bSumAbs = 0.0;
            double bSumSq = 0.0;
            float bMaxAbs = 0.0f;
            summarizeB(bStorage, &bSumAbs, &bSumSq, &bMaxAbs);
            double bMeanAbs = bStorage.empty() ? 0.0 : bSumAbs / static_cast<double>(bStorage.size());
            double bRms = bStorage.empty() ? 0.0 : std::sqrt(bSumSq / static_cast<double>(bStorage.size()));
            std::cerr << "step=" << (step + 1)
                      << " loss_plus=" << lossPlus
                      << " loss_minus=" << lossMinus
                      << " g=" << g
                      << " b_rms=" << bRms
                      << " b_mean_abs=" << bMeanAbs
                      << " b_max_abs=" << bMaxAbs
                      << " step_time_ms=" << stepMs;
            if (printedBaselineLoss) {
                std::cerr << " baseline_loss=" << baselineLoss;
            }
            std::cerr << "\n";
        } else if (printedBaselineLoss) {
            std::cerr << "step=" << (step + 1)
                      << " baseline_loss=" << baselineLoss << "\n";
        }
        if (!std::isfinite(lossPlus) || !std::isfinite(lossMinus) || !std::isfinite(g)) {
            fail("non-finite loss or gradient estimate");
            return 1;
        }

        if (doEval && (((step + 1) % args.evalStep == 0) || (step + 1 == args.steps))) {
            double acc = 0.0;
            if (!evaluateAccuracy(llm.get(), validData, clsIdx, meta.vocabSize, &acc)) {
                return 1;
            }
            std::cout << "step=" << (step + 1)
                      << " accuracy=" << acc
                      << " train_time_s=" << trainTimeS
                      << " pss_memory_mb=" << pssSampler.currentMb()
                      << "\n";
        }
    }

    double speedPp = totalPrefillUs > 0 ? static_cast<double>(totalPrefillTokens) * 1000000.0 / static_cast<double>(totalPrefillUs) : 0.0;
    double avgStepMs = 0.0;
    for (size_t i = 0; i < stepTimesMs.size(); ++i) {
        avgStepMs += stepTimesMs[i];
    }
    avgStepMs /= static_cast<double>(stepTimesMs.size());
    double peakMemoryMb = readPeakMemoryMb();

    std::cout << "speed_pp=" << speedPp << "\n";
    std::cout << "avg_step_time_ms=" << avgStepMs << "\n";
    std::cout << "peak_memory_mb=" << peakMemoryMb << "\n";
    std::cout << "peak_pss_memory_mb=" << pssSampler.peakMb() << "\n";
    return 0;
}
