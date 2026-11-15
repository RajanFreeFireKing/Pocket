#ifndef AUTOREGRESSIVE_ENGINE_HPP
#define AUTOREGRESSIVE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <atomic>
#include <chrono>

#include "../io/layer_streamer.hpp"
#include "../parser/gguf_reader.hpp"
#include "../resource/resource_intelligence.hpp"
#include "../memory/hierarchical_memory.hpp"
#include "../scheduler/compute_scheduler.hpp"
#include "llama.h"

namespace streamvex {

/**
 * Live inference telemetry collected during authentic autoregressive generation.
 */
struct GenerationTelemetry {
    int total_tokens = 0;
    float time_to_first_token_ms = 0.0f;
    float total_duration_sec = 0.0f;
    float tokens_per_sec = 0.0f;
    float prefill_tok_per_sec = 0.0f;
    double baseline_ram_mb = 0.0f;
    double peak_ram_mb = 0.0f;
    double current_ram_mb = 0.0f;
    double kv_cache_ram_mb = 0.0f;
    int layers_streamed = 0;
    double disk_throughput_mb_s = 0.0f;
    bool is_ram_bounded = true;
    std::string model_name;
    std::string architecture;
};

/**
 * Production-Grade Autoregressive Engine for StreamVex.
 * 
 * Executes authentic GGUF Transformer inference powered by the integrated
 * llama.cpp / ggml backend and double-buffered layer streaming.
 * Zero synthetic weights, zero hardcoded responses.
 */
class AutoregressiveEngine {
public:
    AutoregressiveEngine();
    ~AutoregressiveEngine();

    AutoregressiveEngine(const AutoregressiveEngine&) = delete;
    AutoregressiveEngine& operator=(const AutoregressiveEngine&) = delete;

    /**
     * Initializes the engine with a local GGUF model file.
     * Detects model architecture, allocates context, configures memory mapping,
     * and sets up the KV cache.
     * 
     * @param model_path Path to the local .gguf model file.
     * @param n_ctx Context window capacity (default 2048, or auto-budgeted).
     * @param n_threads Compute thread count (0 = auto-detect via ResourceIntelligence).
     * @return true on success, false on failure.
     */
    bool initialize(const std::string& model_path, int n_ctx = 2048, int n_threads = 0);

    /**
     * Streams generated tokens for the prompt to the callback in real time.
     * 
     * @param prompt Null-terminated user prompt.
     * @param callback Function invoked per generated token string piece.
     * @param max_tokens Maximum number of tokens to generate (default 256).
     * @param temperature Sampling temperature (default 0.7).
     * @param top_k Top-K filtering (default 40).
     * @param top_p Top-P nucleus sampling (default 0.9).
     * @return 0 on success, non-zero error code otherwise.
     */
    int generate_stream(
        const std::string& prompt,
        std::function<void(const char* token)> callback,
        int max_tokens = 256,
        float temperature = 0.7f,
        int top_k = 40,
        float top_p = 0.9f
    );

    /**
     * Aborts ongoing token generation asynchronously.
     */
    void abort_generation();

    /**
     * Executes a single forward pass step on a single token.
     * 
     * @param token_id Input token ID.
     * @param pos Sequence position index.
     * @param out_next_token Output pointer to receive sampled next token ID.
     * @param out_ms Output pointer to receive measured step latency in ms.
     * @return 0 on success, non-zero on error.
     */
    int forward_pass_step(
        int token_id,
        int pos,
        int* out_next_token,
        float* out_ms
    );

    /**
     * Queries current resident memory (RSS) in Megabytes.
     */
    double get_memory_usage() const;

    /**
     * Checks if a model is currently loaded and ready for inference.
     */
    bool is_ready() const;

    /**
     * Unloads model weights, context, and frees KV cache.
     */
    void release();

    /**
     * Resets KV cache memory and cached prefix sequence without unloading model weights.
     */
    void reset_context();

    /**
     * Retrieves the latest generation telemetry.
     */
    GenerationTelemetry get_telemetry() const;

    /**
     * Formats current telemetry as JSON.
     */
    std::string get_telemetry_json() const;

    const std::string& model_path() const { return m_model_path; }

    llama_model* raw_model() const { return m_model; }
    llama_context* raw_context() const { return m_ctx; }

private:
    mutable std::mutex m_engine_mtx;
    std::atomic<bool> m_is_initialized{false};
    std::atomic<bool> m_abort_requested{false};
    std::string m_model_path;

    llama_model* m_model = nullptr;
    llama_context* m_ctx = nullptr;
    std::unique_ptr<GgufReader> m_gguf_reader;
    std::unique_ptr<LayerStreamer> m_streamer;

    std::vector<llama_token> m_cached_tokens;

    GenerationTelemetry m_telemetry;
    int m_n_ctx = 2048;
    int m_n_threads = 4;
};

} // namespace streamvex

#endif // AUTOREGRESSIVE_ENGINE_HPP
