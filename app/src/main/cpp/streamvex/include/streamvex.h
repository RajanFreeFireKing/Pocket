#ifndef STREAMVEX_H
#define STREAMVEX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef STREAMVEX_BUILDING_DLL
        #define STREAMVEX_API __declspec(dllexport)
    #else
        #define STREAMVEX_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
        #define STREAMVEX_API __attribute__((visibility("default")))
    #else
        #define STREAMVEX_API
    #endif
#endif

// ============================================================================
// Original StreamVex Core C-ABI (Preserved & Fully Backwards-Compatible)
// ============================================================================

/**
 * Token callback function type.
 * Invoked synchronously or from worker thread for each generated token string.
 */
typedef void (*TokenCallback)(const char* token);

/**
 * Initializes the local LLM engine with the given model path (e.g. .gguf file).
 * 
 * @param model_path Null-terminated string pointing to local model weights.
 * @return 0 on success, non-zero error code if initialization fails.
 */
STREAMVEX_API int init_engine(const char* model_path);

/**
 * Streams tokens for the given prompt to the provided callback function.
 * Executes authentic autoregressive Transformer prefill and decode loops.
 * 
 * @param prompt Null-terminated user prompt / conversation string.
 * @param callback Function pointer called per generated token chunk.
 * @return 0 on completion, non-zero if generation encounters an error or was aborted.
 */
STREAMVEX_API int stream_token(const char* prompt, TokenCallback callback);

/**
 * Retrieves the current resident set / RAM memory usage of the engine in Megabytes (MB).
 * Queries /proc/self/statm for true OS resident set size.
 * 
 * @return RAM usage in MB.
 */
STREAMVEX_API double get_memory_usage(void);

/**
 * Checks whether the engine has been successfully loaded with a model.
 * 
 * @return 1 if ready, 0 otherwise.
 */
STREAMVEX_API int is_engine_ready(void);

/**
 * Releases model weights and frees allocated engine memory.
 */
STREAMVEX_API void release_engine(void);

/**
 * Clears KV cache and resets conversation context without unloading weights.
 */
STREAMVEX_API void reset_context(void);

/**
 * Retrieves the latest generation telemetry summary (TTFT, tokens/sec, peak RAM) in JSON format.
 * 
 * @param out_json Buffer to receive JSON string.
 * @param max_len Maximum capacity of out_json buffer.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int get_last_generation_telemetry(char* out_json, size_t max_len);

/**
 * Runs the asynchronous double-buffering layer streaming benchmark.
 * 
 * @param file_path Path to model file, or NULL / empty for storage benchmarking.
 * @param chunk_size_mb Size of each layer chunk in Megabytes (e.g. 32, 64).
 * @param total_layers Total number of layers to stream sequentially.
 * @param out_json_stats Buffer to receive formatted JSON performance metrics.
 * @param max_len Maximum length of out_json_stats buffer.
 * @return 0 on success, non-zero on failure.
 */
STREAMVEX_API int run_streaming_benchmark(
    const char* file_path,
    size_t chunk_size_mb,
    int total_layers,
    char* out_json_stats,
    size_t max_len
);

/**
 * Parses GGUF model metadata and tensor catalog from disk without loading tensor data.
 * Outputs formatted JSON into out_json_info.
 * 
 * @param model_path Path to the local .gguf file.
 * @param out_json_info Output buffer for JSON summary.
 * @param max_len Maximum length of out_json_info buffer.
 * @return 0 on success, non-zero on failure.
 */
STREAMVEX_API int parse_gguf_metadata(
    const char* model_path,
    char* out_json_info,
    size_t max_len
);

/**
 * Streams a specific layer tensor from disk in bounded, page-aligned chunks
 * and dequantizes it directly into out_buffer without allocating a duplicate full-tensor buffer.
 * Supports F32, F16, Q4_0, Q8_0, and Q4_K formats.
 * 
 * @param model_path Path to the local .gguf file.
 * @param tensor_name Name of the target tensor (e.g. "blk.0.attn_q.weight").
 * @param out_buffer Output float array to receive dequantized weights.
 * @param max_floats Maximum number of float elements out_buffer can store.
 * @return Number of dequantized floats on success, negative error code on failure.
 */
STREAMVEX_API int stream_and_dequant_layer(
    const char* model_path,
    const char* tensor_name,
    float* out_buffer,
    size_t max_floats
);

/**
 * Runs high-performance Matrix-Vector Multiplication (GEMV) benchmark comparing
 * single-precision float32 vs direct 4-bit (Q4_0) and 8-bit (Q8_0) execution.
 * 
 * @param rows Number of matrix rows.
 * @param cols Number of matrix columns.
 * @param out_json_stats Output buffer for JSON summary containing GFLOPS and bandwidth.
 * @param max_len Maximum length of out_json_stats buffer.
 * @return 0 on success, non-zero on failure.
 */
STREAMVEX_API int compute_gemv_benchmark(
    int rows,
    int cols,
    char* out_json_stats,
    size_t max_len
);

/**
 * Executes an authentic single autoregressive Transformer forward pass step:
 * Evaluates the input token through the active model context, computes logits,
 * samples the next token, and records true step execution latency.
 * 
 * @param token_id Current input token ID.
 * @param pos Current sequence position index.
 * @param out_next_token Output pointer to receive predicted next token ID.
 * @param out_ms Output pointer to receive forward pass step execution latency in milliseconds.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int run_forward_pass_step(
    int token_id,
    int pos,
    int* out_next_token,
    float* out_ms
);

// ============================================================================
// Version 2.0 Extended Adaptive Multi-Model & Hardware Intelligence APIs
// ============================================================================

/**
 * Asynchronously aborts any active autoregressive token generation.
 */
STREAMVEX_API void streamvex_abort_generation(void);

/**
 * Retrieves real-time device resource telemetry (System RAM, Process RSS,
 * CPU utilization, Big/Little core topology, Thermal status, Battery state,
 * and Adaptive Resource Budget) formatted as JSON.
 * 
 * @param out_json Buffer to receive formatted JSON string.
 * @param max_len Capacity of out_json buffer.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int streamvex_get_resource_telemetry(char* out_json, size_t max_len);

/**
 * Retrieves the mobile device hardware classification profile (Low, Mid, Flagship)
 * and recommended model sizing as JSON.
 * 
 * @param out_json Buffer to receive formatted JSON string.
 * @param max_len Capacity of out_json buffer.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int streamvex_get_device_profile(char* out_json, size_t max_len);

/**
 * Validates whether a candidate GGUF model can safely fit in memory
 * on the current device without risking Android OOM foreground application kills.
 * 
 * @param model_path Path to candidate .gguf file.
 * @param context_len Desired context length (e.g. 2048).
 * @param out_json_result Buffer to receive compatibility verdict and recommendations as JSON.
 * @param max_len Capacity of out_json_result buffer.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int streamvex_validate_model(
    const char* model_path,
    int context_len,
    char* out_json_result,
    size_t max_len
);

/**
 * Delivers Android OS memory trim notification to trigger memory recovery.
 * 
 * @param trim_level Level constant from Android ComponentCallbacks2 (e.g. TRIM_MEMORY_RUNNING_CRITICAL = 15).
 */
STREAMVEX_API void streamvex_on_trim_memory(int trim_level);

/**
 * Configures background execution mode for zero-stutter adaptive operation.
 * 
 * @param is_background 1 if running in background (throttles compute, reduces threads), 0 for foreground interactive.
 */
STREAMVEX_API void streamvex_set_background_mode(int is_background);

/**
 * Processes an audio frame for Voice Activity Detection (VAD).
 * 
 * @param pcm16 Array of 16-bit mono PCM samples (16kHz).
 * @param num_samples Number of samples in frame (e.g. 160 = 10ms, 320 = 20ms).
 * @param out_is_speech Pointer to receive speech flag (1 = speech, 0 = silence/noise).
 * @param out_prob Pointer to receive speech probability [0.0, 1.0].
 * @param out_energy_db Pointer to receive frame energy in dB.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int streamvex_voice_process_vad(
    const int16_t* pcm16,
    size_t num_samples,
    int* out_is_speech,
    float* out_prob,
    float* out_energy_db
);

/**
 * Synthesizes text into offline 16kHz 16-bit PCM speech audio.
 * 
 * @param text Null-terminated text to synthesize.
 * @param out_pcm16 Caller-allocated buffer for PCM16 audio samples.
 * @param max_samples Maximum capacity of out_pcm16 buffer.
 * @param out_actual_samples Output pointer receiving actual count of synthesized samples.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int streamvex_voice_synthesize(
    const char* text,
    int16_t* out_pcm16,
    size_t max_samples,
    int* out_actual_samples
);

/**
 * Transcribes spoken audio into text.
 * 
 * @param pcm16 Input 16-bit PCM audio samples (16kHz).
 * @param num_samples Number of audio samples.
 * @param out_text Buffer to receive recognized text string.
 * @param max_len Maximum length of out_text buffer.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int streamvex_voice_transcribe(
    const int16_t* pcm16,
    size_t num_samples,
    char* out_text,
    size_t max_len
);

/**
 * Benchmarks Double-Buffering vs Triple-Buffering streaming on current mobile storage.
 * 
 * @param model_path Path to model file.
 * @param chunk_size_mb Chunk size in MB.
 * @param total_layers Total layers to stream.
 * @param out_json Output buffer receiving comparative JSON results.
 * @param max_len Capacity of out_json buffer.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int streamvex_benchmark_triple_buffering(
    const char* model_path,
    size_t chunk_size_mb,
    int total_layers,
    char* out_json,
    size_t max_len
);

#ifdef __cplusplus
}
#endif

#endif // STREAMVEX_H
