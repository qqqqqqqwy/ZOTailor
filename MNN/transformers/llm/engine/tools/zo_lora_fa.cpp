#include "llm/llm.hpp"

#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <rapidjson/document.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using MNN::Express::NCHW;
using MNN::Express::VARP;
using MNN::Express::VARPS;
using MNN::Express::_Input;
using MNN::Transformer::Llm;

struct Args {
    std::string config;
    std::string loraMeta;
    int steps = 5;
    int batch = 4;
    int seqLen = 32;
    float epsilon = 1.0e-2f;
    float lr = 5.0e-5f;
    int threads = -1;
    int seed = 42;
    bool verbose = false;
};

struct Target {
    std::string inputName;
    std::vector<int> shape;
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

struct Samples {
    std::vector<std::vector<int> > tokens;
    std::vector<std::vector<int> > labels;
};

static bool fail(const std::string& message) {
    std::cerr << "zo_lora_fa error: " << message << "\n";
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

static void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --config config_lora_fa.json [--lora-meta lora_fa.json]\n"
              << "  [--steps 5] [--batch 4] [--seq-len 32] [--epsilon 1e-2]\n"
              << "  [--lr 5e-5] [--threads -1, <=0 auto] [--seed 42] [--verbose]\n";
}

static bool parseArgs(int argc, const char* argv[], Args* args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        bool needValue = key != "--verbose" && key != "--help" && key != "-h";
        const char* value = nullptr;
        if (needValue && key.size() > 0 && key[0] == '-') {
            if (i + 1 >= argc) {
                return fail("missing value for " + key);
            }
            value = argv[++i];
        }
        if (key == "--config") {
            args->config = value;
        } else if (key == "--lora-meta") {
            args->loraMeta = value;
        } else if (key == "--steps") {
            args->steps = std::atoi(value);
        } else if (key == "--batch" || key == "--batch-size") {
            args->batch = std::atoi(value);
        } else if (key == "--seq-len") {
            args->seqLen = std::atoi(value);
        } else if (key == "--epsilon") {
            args->epsilon = static_cast<float>(std::atof(value));
        } else if (key == "--lr") {
            args->lr = static_cast<float>(std::atof(value));
        } else if (key == "--threads") {
            args->threads = std::atoi(value);
        } else if (key == "--seed") {
            args->seed = std::atoi(value);
        } else if (key == "--verbose") {
            args->verbose = true;
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
    if (args->loraMeta.empty()) {
        args->loraMeta = dirname(args->config) + "/lora_fa.json";
    }
    if (args->steps <= 0 || args->batch <= 0 || args->seqLen <= 0 || args->epsilon <= 0.0f || args->lr <= 0.0f) {
        return fail("steps, batch, seq-len, epsilon and lr must be positive");
    }
    return true;
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
        target.shape.push_back(item["b_shape"][0].GetInt());
        target.shape.push_back(item["b_shape"][1].GetInt());
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

static bool makeSamples(int batch, int seqLen, int vocabSize, int seed, Samples* samples) {
    if (vocabSize <= 1) {
        return fail("metadata vocab_size must be greater than 1");
    }
    samples->tokens.resize(batch);
    samples->labels.resize(batch);
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> dist(0, vocabSize - 1);
    for (int b = 0; b < batch; ++b) {
        samples->tokens[b].resize(seqLen);
        samples->labels[b].resize(seqLen);
        for (int i = 0; i < seqLen; ++i) {
            samples->tokens[b][i] = dist(rng);
            samples->labels[b][i] = dist(rng);
        }
    }
    return true;
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

static bool syncBVars(const Metadata& meta, const std::vector<float>& storage, const VARPS& vars) {
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
        ::memcpy(ptr, storage.data() + target.offset, target.size * sizeof(float));
    }
    return true;
}

static bool crossEntropy(VARP logits, const std::vector<int>& labels, int seqLen, int expectedVocab, double* outLoss) {
    if (logits.get() == nullptr || logits->getInfo() == nullptr) {
        return fail("logits is null");
    }
    const auto* info = logits->getInfo();
    const float* data = logits->readMap<float>();
    if (data == nullptr || info->size <= 0) {
        return fail("failed to read logits");
    }
    int vocab = expectedVocab;
    if (!info->dim.empty()) {
        int lastDim = info->dim.back();
        if (lastDim > 1 && info->size % lastDim == 0) {
            vocab = lastDim;
        }
    }
    if (vocab <= 1 || info->size % vocab != 0) {
        return fail("cannot infer logits vocab dimension");
    }
    int positions = static_cast<int>(info->size / vocab);
    int usePositions = std::min(seqLen, positions);
    if (usePositions <= 0) {
        return fail("no logits positions available");
    }

    double loss = 0.0;
    for (int p = 0; p < usePositions; ++p) {
        const float* row = data + static_cast<size_t>(p) * static_cast<size_t>(vocab);
        float maxValue = row[0];
        for (int v = 1; v < vocab; ++v) {
            maxValue = std::max(maxValue, row[v]);
        }
        double sum = 0.0;
        for (int v = 0; v < vocab; ++v) {
            sum += std::exp(static_cast<double>(row[v] - maxValue));
        }
        int label = labels[p] % vocab;
        loss += std::log(sum) + static_cast<double>(maxValue) - static_cast<double>(row[label]);
    }
    *outLoss = loss / static_cast<double>(usePositions);
    return true;
}

static bool evaluateLoss(Llm* llm, const Samples& samples, int seqLen, int vocabSize, int64_t* prefillUs, double* outLoss) {
    double lossSum = 0.0;
    int count = 0;
    for (size_t i = 0; i < samples.tokens.size(); ++i) {
        llm->response(samples.tokens[i], nullptr, nullptr, 0);
        auto outputs = llm->getOutputs();
        if (outputs.empty()) {
            return fail("LLM forward produced no outputs");
        }
        double loss = 0.0;
        if (!crossEntropy(outputs[0], samples.labels[i], seqLen, vocabSize, &loss)) {
            return false;
        }
        lossSum += loss * static_cast<double>(seqLen);
        count += seqLen;
        const auto* context = llm->getContext();
        if (context != nullptr) {
            *prefillUs += context->prefill_us;
        }
        auto scope = MNN::Express::ExecutorScope::Current();
        if (scope != nullptr) {
            scope->gc();
        }
    }
    if (count == 0) {
        return fail("empty loss batch");
    }
    *outLoss = lossSum / static_cast<double>(count);
    return true;
}

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

static std::string runtimeConfigJson(int threads) {
    std::ostringstream os;
    os << "{"
       << "\"backend_type\":\"cpu\","
       << "\"thread_num\":" << threads << ","
       << "\"precision\":\"low\","
       << "\"memory\":\"low\","
       << "\"all_logits\":true,"
       << "\"reuse_kv\":false,"
       << "\"async\":false,"
       << "\"use_mmap\":false,"
       << "\"use_cached_mmap\":false,"
       << "\"use_template\":false"
       << "}";
    return os.str();
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
    if (meta.alpha != meta.rank) {
        fail("LoRA metadata must have alpha == rank for this LoRA-FA runner");
        return 1;
    }

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
    llm->setExtraInputNames(extraInputNames);
    llm->set_config(runtimeConfigJson(runtimeThreads));

    VARPS bVars = createBVars(meta);
    std::vector<float> bStorage(meta.totalBSize, 0.0f);
    if (!syncBVars(meta, bStorage, bVars)) {
        return 1;
    }
    llm->setExtraInputs(bVars);

    if (!llm->load()) {
        fail("failed to load LLM");
        return 1;
    }

    Samples samples;
    if (!makeSamples(args.batch, args.seqLen, meta.vocabSize, args.seed, &samples)) {
        return 1;
    }

    std::mt19937 rng(static_cast<uint32_t>(args.seed + 1));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> z(meta.totalBSize, 0.0f);
    std::vector<double> stepTimesMs;
    stepTimesMs.reserve(args.steps);
    int64_t totalPrefillUs = 0;

    for (int step = 0; step < args.steps; ++step) {
        auto stepStart = std::chrono::steady_clock::now();
        for (size_t i = 0; i < z.size(); ++i) {
            z[i] = normal(rng);
            bStorage[i] += args.epsilon * z[i];
        }
        if (!syncBVars(meta, bStorage, bVars)) {
            return 1;
        }
        double lossPlus = 0.0;
        if (!evaluateLoss(llm.get(), samples, args.seqLen, meta.vocabSize, &totalPrefillUs, &lossPlus)) {
            return 1;
        }

        for (size_t i = 0; i < z.size(); ++i) {
            bStorage[i] -= 2.0f * args.epsilon * z[i];
        }
        if (!syncBVars(meta, bStorage, bVars)) {
            return 1;
        }
        double lossMinus = 0.0;
        if (!evaluateLoss(llm.get(), samples, args.seqLen, meta.vocabSize, &totalPrefillUs, &lossMinus)) {
            return 1;
        }

        double g = (lossPlus - lossMinus) / (2.0 * static_cast<double>(args.epsilon));
        for (size_t i = 0; i < z.size(); ++i) {
            bStorage[i] += args.epsilon * z[i] - static_cast<float>(args.lr * g * z[i]);
        }
        if (!syncBVars(meta, bStorage, bVars)) {
            return 1;
        }

        auto stepEnd = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
        stepTimesMs.push_back(ms);
        if (args.verbose) {
            std::cerr << "step=" << step
                      << " loss_plus=" << lossPlus
                      << " loss_minus=" << lossMinus
                      << " g=" << g
                      << " step_time_ms=" << ms << "\n";
        }
        if (!std::isfinite(lossPlus) || !std::isfinite(lossMinus) || !std::isfinite(g)) {
            fail("non-finite loss or gradient estimate");
            return 1;
        }
    }

    double totalTokens = static_cast<double>(args.steps) * 2.0 * static_cast<double>(args.batch) * static_cast<double>(args.seqLen);
    double speedPp = totalPrefillUs > 0 ? totalTokens * 1000000.0 / static_cast<double>(totalPrefillUs) : 0.0;
    double avgStepMs = 0.0;
    for (size_t i = 0; i < stepTimesMs.size(); ++i) {
        avgStepMs += stepTimesMs[i];
    }
    avgStepMs /= static_cast<double>(stepTimesMs.size());
    double peakMemoryMb = readPeakMemoryMb();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "speed_pp=" << speedPp << "\n";
    std::cout << "avg_step_time_ms=" << avgStepMs << "\n";
    std::cout << "peak_memory_mb=" << peakMemoryMb << "\n";
    return 0;
}
