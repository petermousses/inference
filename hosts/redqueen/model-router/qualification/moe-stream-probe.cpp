// Short resident/SSD parity fixture. Run only inside the host maintenance window.
// SLOTS=0 and BUDGET=0 means resident; either positive value enables streaming.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

uint64_t parse_unsigned(const char * text, const char * name, uint64_t maximum) {
    if (!text || !*text) {
        throw std::runtime_error(std::string(name) + " must be an unsigned decimal integer");
    }
    uint64_t value = 0;
    for (const char * p = text; *p; ++p) {
        if (*p < '0' || *p > '9') {
            throw std::runtime_error(std::string(name) + " must be an unsigned decimal integer");
        }
        const uint64_t digit = static_cast<uint64_t>(*p - '0');
        if (value > (maximum - digit) / 10) {
            throw std::runtime_error(std::string(name) + " is out of range");
        }
        value = value * 10 + digit;
    }
    return value;
}

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
    Backend(const Backend &) = delete;
    Backend & operator=(const Backend &) = delete;
};

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 7) {
            throw std::runtime_error("usage: probe MODEL GPU_LAYERS SLOTS BUDGET_BYTES STEPS LOGITS_FILE; both zero for resident, either positive for streaming");
        }
        const uint64_t layer_value = parse_unsigned(argv[2], "GPU_LAYERS", std::numeric_limits<int>::max());
        const uint64_t slots_value = parse_unsigned(argv[3], "SLOTS", std::numeric_limits<uint32_t>::max());
        const uint64_t budget = parse_unsigned(argv[4], "BUDGET_BYTES", std::numeric_limits<uint64_t>::max());
        const uint64_t step_value = parse_unsigned(argv[5], "STEPS", 256);
        const int layers = static_cast<int>(layer_value);
        const uint32_t slots = static_cast<uint32_t>(slots_value);
        const uint32_t steps = static_cast<uint32_t>(step_value);
        if (steps == 0 || steps > 256) {
            throw std::runtime_error("steps must be in 1..256");
        }
        if (argv[1][0] == '\0' || layers <= 0) {
            throw std::runtime_error("MODEL must be nonempty and GPU_LAYERS must be positive");
        }
        uint64_t max_cache_budget = 120ull * 1024 * 1024 * 1024;
        if (const char * raw_max_cache = std::getenv("MOE_PROBE_MAX_CACHE_BYTES")) {
            max_cache_budget = parse_unsigned(raw_max_cache, "MOE_PROBE_MAX_CACHE_BYTES", std::numeric_limits<uint64_t>::max());
        }
        if (max_cache_budget == 0) {
            throw std::runtime_error("MOE_PROBE_MAX_CACHE_BYTES must be positive");
        }
        uint64_t resize_bytes = 0;
        if (const char * raw_resize = std::getenv("MOE_PROBE_RESIZE_BYTES")) {
            resize_bytes = parse_unsigned(raw_resize, "MOE_PROBE_RESIZE_BYTES", std::numeric_limits<uint64_t>::max());
            if (resize_bytes == 0) {
                throw std::runtime_error("MOE_PROBE_RESIZE_BYTES must be positive");
            }
        }
        uint64_t context_value = 512;
        if (const char * raw_context = std::getenv("MOE_PROBE_CONTEXT_TOKENS")) {
            context_value = parse_unsigned(raw_context, "MOE_PROBE_CONTEXT_TOKENS", 220000);
        }
        if (context_value == 0) {
            throw std::runtime_error("MOE_PROBE_CONTEXT_TOKENS must be positive");
        }
        const uint32_t context_tokens = static_cast<uint32_t>(context_value);
        ggml_backend_load_all();
        Backend backend;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = layers;
        mp.moe_stream = slots != 0 || budget != 0;
        mp.moe_stream_slots = slots;
        mp.moe_stream_budget = budget;
        mp.moe_stream_max_budget = max_cache_budget;
        mp.moe_stream_io_threads = 1;
        mp.moe_stream_direct = mp.moe_stream;
        using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
        Model model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) { throw std::runtime_error("model load failed"); }
        auto cp = llama_context_default_params();
        cp.n_ctx = context_tokens;
        cp.n_batch = 1;
        cp.n_ubatch = 1;
        cp.n_seq_max = 1;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.no_perf = true;
        Context context(llama_init_from_model(model.get(), cp), llama_free);
        if (!context) { throw std::runtime_error("context creation failed"); }
        if (resize_bytes != 0) {
            if (!mp.moe_stream) {
                throw std::runtime_error("MOE_PROBE_RESIZE_BYTES requires streamed MoE mode");
            }
            llama_moe_stream_info before{};
            if (!llama_moe_stream_get_info(model.get(), &before) || before.maximum_capacity_bytes == 0) {
                throw std::runtime_error("model does not expose an adaptive expert-cache range");
            }
            if (resize_bytes < before.minimum_capacity_bytes || resize_bytes > before.maximum_capacity_bytes) {
                throw std::runtime_error("MOE_PROBE_RESIZE_BYTES is outside the model's expert-cache range");
            }
            llama_moe_stream_info resized{};
            if (!llama_moe_stream_resize_cache(context.get(), resize_bytes, nullptr, nullptr, &resized)) {
                throw std::runtime_error("expert-cache resize was rejected");
            }
            if (resized.cache_bytes > resize_bytes ||
                (resize_bytes < before.maximum_capacity_bytes && resized.cache_bytes >= before.maximum_capacity_bytes)) {
                throw std::runtime_error("expert-cache resize exceeded its requested capacity");
            }
            fprintf(stderr, "resize target=%llu actual=%llu maximum=%llu slots=%u\n",
                (unsigned long long) resize_bytes, (unsigned long long) resized.cache_bytes,
                (unsigned long long) resized.maximum_capacity_bytes, resized.slots_per_layer);
        }
        const auto * vocab = llama_model_get_vocab(model.get());
        const std::string prompt = "A scientist compares exact arithmetic with a disk-backed mixture of experts. Explain why moving unchanged weights should preserve model outputs.";
        const auto text_length = static_cast<int32_t>(prompt.size());
        int count = -llama_tokenize(vocab, prompt.data(), text_length, nullptr, 0, true, true);
        if (count <= 0) { throw std::runtime_error("token sizing failed"); }
        std::vector<llama_token> tokens(static_cast<std::size_t>(count));
        count = llama_tokenize(vocab, prompt.data(), text_length, tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
        if (count <= 0) { throw std::runtime_error("tokenization failed"); }
        tokens.resize(static_cast<std::size_t>(count));
        const int32_t vocab_count = llama_vocab_n_tokens(vocab);
        if (vocab_count <= 0) { throw std::runtime_error("vocabulary size is invalid"); }
        const uint32_t n_vocab = static_cast<uint32_t>(vocab_count);
        using File = std::unique_ptr<FILE, decltype(&fclose)>;
        File output(fopen(argv[6], "wb"), fclose);
        if (!output) { throw std::runtime_error("cannot open logits file"); }
        const uint32_t header[] = {0x4d4f4531, n_vocab, steps};
        if (fwrite(header, sizeof(header), 1, output.get()) != 1) {
            throw std::runtime_error("cannot write header");
        }
        if (const char * barrier_path = std::getenv("MOE_PROBE_BARRIER")) {
            if (*barrier_path == '\0') {
                throw std::runtime_error("MOE_PROBE_BARRIER must be a nonempty path");
            }
            fprintf(stderr, "READY\n");
            fflush(stderr);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(15);
            while (true) {
                std::ifstream barrier(barrier_path);
                if (barrier.good()) {
                    break;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("timed out waiting for concurrent decode barrier");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        llama_token next = tokens[0];
        for (uint32_t pos = 0; pos < steps; ++pos) {
            llama_token input = pos < tokens.size() ? tokens[pos] : next;
            llama_batch batch = llama_batch_get_one(&input, 1);
            if (llama_decode(context.get(), batch) != 0) {
                throw std::runtime_error("decode failed");
            }
            const float * logits = llama_get_logits_ith(context.get(), -1);
            if (!logits || !std::all_of(logits, logits + n_vocab, [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("nonfinite or missing logits");
            }
            const auto next_index = std::distance(logits, std::max_element(logits, logits + n_vocab));
            next = static_cast<llama_token>(next_index);
            if (fwrite(logits, sizeof(float), n_vocab, output.get()) != n_vocab) {
                throw std::runtime_error("cannot write logits");
            }
        }
        if (fflush(output.get()) || ferror(output.get())) { throw std::runtime_error("logits write failed"); }
        llama_moe_stream_info info{};
        const bool streamed = llama_moe_stream_get_info(model.get(), &info);
        if (mp.moe_stream && (!streamed || !info.direct_io_verified || info.pending_reads || info.load_failed || !info.misses)) {
            throw std::runtime_error("stream completion contract failed");
        }
        printf("{\"steps\":%u,\"context_tokens\":%u,\"vocab\":%u,\"streamed\":%s,\"cache_bytes\":%llu,\"minimum_capacity_bytes\":%llu,\"maximum_capacity_bytes\":%llu,\"host_overhead_bytes\":%llu,\"slots_per_layer\":%u,\"hits\":%llu,\"misses\":%llu,\"pending_reads\":%u,\"direct_io_verified\":%s}\n",
            steps, context_tokens, n_vocab, streamed ? "true" : "false", (unsigned long long) info.cache_bytes,
            (unsigned long long) info.minimum_capacity_bytes, (unsigned long long) info.maximum_capacity_bytes,
            (unsigned long long) info.host_overhead_bytes, info.slots_per_layer, (unsigned long long) info.hits,
            (unsigned long long) info.misses, info.pending_reads, info.direct_io_verified ? "true" : "false");
        context.reset();
        model.reset();
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "probe: %s\n", error.what());
        return 1;
    }
}
