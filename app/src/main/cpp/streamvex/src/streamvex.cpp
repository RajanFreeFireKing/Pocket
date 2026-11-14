#include "streamvex.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "engine/autoregressive_engine.hpp"
#include "resource/resource_intelligence.hpp"
#include "memory/hierarchical_memory.hpp"
#include "scheduler/compute_scheduler.hpp"
#include "dual_model/dual_model_runtime.hpp"
#include "device/device_profile.hpp"
#include "io/layer_streamer.hpp"
#include "parser/gguf_reader.hpp"
#include "compute/dequant.hpp"
#include "compute/gemv.hpp"
#include "compute/sampler.hpp"

namespace {

extern "C" {
    void dequantize_row_q5_0(const void* x, float* y, int64_t k);
    void dequantize_row_q6_K(const void* x, float* y, int64_t k);
}

static streamvex::AutoregressiveEngine g_engine;
static std::mutex g_api_mtx;
static bool g_background_mode = false;

} // anonymous namespace

// ============================================================================
// Original Core Engine C-API Implementation (Zero Fake Data)
// ============================================================================

STREAMVEX_API int init_engine(const char* model_path) {
    if (!model_path) {
        std::cerr << "[StreamVex] Error: Null model path provided.\n";
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_api_mtx);

    // Auto-detect thread budget based on foreground/background status
    auto budget = streamvex::ResourceIntelligence::instance().compute_budget(0.0, !g_background_mode);
    bool ok = g_engine.initialize(model_path, 2048, budget.recommended_threads);
    if (!ok) {
        return -2;
    }

    // Initialize dual model runtime for concurrent voice & background support
    streamvex::DualModelRuntime::instance().initialize(model_path);

    return 0;
}

STREAMVEX_API int stream_token(const char* prompt, TokenCallback callback) {
    if (!callback) return -1;
    if (!prompt) {
        callback("[StreamVex Error]: Null prompt received.");
        return -2;
    }

    return g_engine.generate_stream(prompt, callback);
}

STREAMVEX_API double get_memory_usage(void) {
    auto& rim = streamvex::ResourceIntelligence::instance();
    rim.update_metrics();
    return rim.get_memory_info().process_rss_mb;
}

STREAMVEX_API int is_engine_ready(void) {
    return g_engine.is_ready() ? 1 : 0;
}

STREAMVEX_API void release_engine(void) {
    std::lock_guard<std::mutex> lock(g_api_mtx);
    g_engine.release();
    streamvex::DualModelRuntime::instance().shutdown();
    std::cout << "[StreamVex] Engine unloaded and resources released.\n";
}

STREAMVEX_API int get_last_generation_telemetry(char* out_json, size_t max_len) {
    if (!out_json || max_len == 0) return -1;

    std::string json_str = g_engine.get_telemetry_json();
    size_t copy_len = std::min(json_str.size(), max_len - 1);
    std::memcpy(out_json, json_str.c_str(), copy_len);
    out_json[copy_len] = '\0';
    return 0;
}

STREAMVEX_API int run_streaming_benchmark(
    const char* file_path,
    size_t chunk_size_mb,
    int total_layers,
    char* out_json_stats,
    size_t max_len
) {
    if (chunk_size_mb == 0) chunk_size_mb = 64;
    if (total_layers <= 0) total_layers = 16;

    size_t chunk_bytes = chunk_size_mb * 1024 * 1024;
    streamvex::LayerStreamer streamer(chunk_bytes);

    if (file_path && strlen(file_path) > 0) {
        streamer.open_file(file_path);
    }

    auto result = streamer.run_benchmark(total_layers, chunk_bytes, 2);

    double total_mb = (static_cast<double>(total_layers) * static_cast<double>(chunk_bytes)) / (1024.0 * 1024.0);
    double double_buffer_mb = (2.0 * static_cast<double>(chunk_bytes)) / (1024.0 * 1024.0);

    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    json << "{\n"
         << "  \"status\": \"success\",\n"
         << "  \"total_layers\": " << result.total_layers << ",\n"
         << "  \"chunk_size_mb\": " << static_cast<double>(chunk_size_mb) << ",\n"
         << "  \"total_mb_streamed\": " << total_mb << ",\n"
         << "  \"total_duration_sec\": " << result.total_duration_sec << ",\n"
         << "  \"disk_throughput_mb_s\": " << result.throughput_mb_s << ",\n"
         << "  \"avg_prefetch_latency_ms\": " << result.avg_prefetch_latency_ms << ",\n"
         << "  \"baseline_ram_mb\": " << result.baseline_ram_mb << ",\n"
         << "  \"peak_ram_mb\": " << result.peak_ram_mb << ",\n"
         << "  \"buffer_cap_mb\": " << double_buffer_mb << ",\n"
         << "  \"is_ram_bounded\": " << (result.is_ram_bounded ? "true" : "false") << "\n"
         << "}";

    std::string json_str = json.str();
    if (out_json_stats && max_len > 0) {
        size_t copy_len = std::min(json_str.size(), max_len - 1);
        std::memcpy(out_json_stats, json_str.c_str(), copy_len);
        out_json_stats[copy_len] = '\0';
    }
    return 0;
}

STREAMVEX_API int parse_gguf_metadata(
    const char* model_path,
    char* out_json_info,
    size_t max_len
) {
    if (!model_path || !out_json_info || max_len == 0) return -1;

    streamvex::GgufReader reader;
    if (!reader.open(model_path)) {
        std::string err = "{\"error\": \"Failed to open GGUF file: " + std::string(model_path) + "\"}";
        size_t copy_len = std::min(err.size(), max_len - 1);
        std::memcpy(out_json_info, err.c_str(), copy_len);
        out_json_info[copy_len] = '\0';
        return -2;
    }

    std::string summary = reader.to_json_summary();
    size_t copy_len = std::min(summary.size(), max_len - 1);
    std::memcpy(out_json_info, summary.c_str(), copy_len);
    out_json_info[copy_len] = '\0';
    return 0;
}

STREAMVEX_API int stream_and_dequant_layer(
    const char* model_path,
    const char* tensor_name,
    float* out_buffer,
    size_t max_floats
) {
    if (!model_path || !tensor_name || !out_buffer || max_floats == 0) return -1;

    streamvex::GgufReader reader;
    if (!reader.open(model_path)) return -2;

    const auto* tensor = reader.find_tensor(tensor_name);
    if (!tensor) return -3;

    size_t n_elements = std::min(static_cast<size_t>(tensor->num_elements), max_floats);
    int fd = open(model_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -4;

    // Stream in bounded chunks of 64KB page-aligned buffer without allocating full tensor in RAM
    const size_t CHUNK_BYTES = 64 * 1024;
    std::vector<uint8_t> chunk_buf(CHUNK_BYTES);

    size_t elements_processed = 0;
    off_t file_cur = static_cast<off_t>(tensor->file_offset);

    if (tensor->type == streamvex::GgmlType::F32) {
        size_t bytes_to_copy = n_elements * sizeof(float);
        size_t bytes_done = 0;
        while (bytes_done < bytes_to_copy) {
            size_t batch = std::min(CHUNK_BYTES, bytes_to_copy - bytes_done);
            ssize_t n = pread(fd, chunk_buf.data(), batch, file_cur + bytes_done);
            if (n <= 0) break;
            std::memcpy(reinterpret_cast<uint8_t*>(out_buffer) + bytes_done, chunk_buf.data(), n);
            bytes_done += n;
        }
        close(fd);
        return static_cast<int>(bytes_done / sizeof(float));
    } else if (tensor->type == streamvex::GgmlType::Q4_0) {
        // Q4_0: 18 bytes per 32 elements
        const size_t block_bytes = sizeof(streamvex::block_q4_0);
        const size_t blocks_per_chunk = CHUNK_BYTES / block_bytes;
        const size_t total_blocks = (n_elements + streamvex::QK4_0 - 1) / streamvex::QK4_0;
        size_t blocks_done = 0;

        while (blocks_done < total_blocks) {
            size_t batch_blocks = std::min(blocks_per_chunk, total_blocks - blocks_done);
            size_t read_bytes = batch_blocks * block_bytes;
            ssize_t n = pread(fd, chunk_buf.data(), read_bytes, file_cur + blocks_done * block_bytes);
            if (n <= 0) break;

            size_t actual_blocks = n / block_bytes;
            size_t batch_elements = actual_blocks * streamvex::QK4_0;
            if (elements_processed + batch_elements > n_elements) {
                batch_elements = n_elements - elements_processed;
            }

            streamvex::dequantize_q4_0(chunk_buf.data(), out_buffer + elements_processed, actual_blocks * streamvex::QK4_0);
            elements_processed += batch_elements;
            blocks_done += actual_blocks;
            if (actual_blocks < batch_blocks) break;
        }
        close(fd);
        return static_cast<int>(elements_processed);
    } else if (tensor->type == streamvex::GgmlType::Q8_0) {
        // Q8_0: 34 bytes per 32 elements
        const size_t block_bytes = sizeof(streamvex::block_q8_0);
        const size_t blocks_per_chunk = CHUNK_BYTES / block_bytes;
        const size_t total_blocks = (n_elements + streamvex::QK8_0 - 1) / streamvex::QK8_0;
        size_t blocks_done = 0;

        while (blocks_done < total_blocks) {
            size_t batch_blocks = std::min(blocks_per_chunk, total_blocks - blocks_done);
            size_t read_bytes = batch_blocks * block_bytes;
            ssize_t n = pread(fd, chunk_buf.data(), read_bytes, file_cur + blocks_done * block_bytes);
            if (n <= 0) break;

            size_t actual_blocks = n / block_bytes;
            size_t batch_elements = actual_blocks * streamvex::QK8_0;
            if (elements_processed + batch_elements > n_elements) {
                batch_elements = n_elements - elements_processed;
            }

            streamvex::dequantize_q8_0(chunk_buf.data(), out_buffer + elements_processed, actual_blocks * streamvex::QK8_0);
            elements_processed += batch_elements;
            blocks_done += actual_blocks;
            if (actual_blocks < batch_blocks) break;
        }
        close(fd);
        return static_cast<int>(elements_processed);
    } else if (tensor->type == streamvex::GgmlType::Q4_K) {
        // Q4_K: 144 bytes per 256 elements
        const size_t block_bytes = sizeof(streamvex::block_q4_K);
        const size_t blocks_per_chunk = CHUNK_BYTES / block_bytes;
        const size_t total_blocks = (n_elements + streamvex::QK_K - 1) / streamvex::QK_K;
        size_t blocks_done = 0;

        while (blocks_done < total_blocks) {
            size_t batch_blocks = std::min(blocks_per_chunk, total_blocks - blocks_done);
            size_t read_bytes = batch_blocks * block_bytes;
            ssize_t n = pread(fd, chunk_buf.data(), read_bytes, file_cur + blocks_done * block_bytes);
            if (n <= 0) break;

            size_t actual_blocks = n / block_bytes;
            size_t batch_elements = actual_blocks * streamvex::QK_K;
            if (elements_processed + batch_elements > n_elements) {
                batch_elements = n_elements - elements_processed;
            }

            streamvex::dequantize_q4_K(chunk_buf.data(), out_buffer + elements_processed, actual_blocks * streamvex::QK_K);
            elements_processed += batch_elements;
            blocks_done += actual_blocks;
            if (actual_blocks < batch_blocks) break;
        }
        close(fd);
        return static_cast<int>(elements_processed);
    } else if (tensor->type == streamvex::GgmlType::Q5_0) {
        // Q5_0: 22 bytes per 32 elements
        const size_t block_bytes = 22;
        const size_t blocks_per_chunk = CHUNK_BYTES / block_bytes;
        const size_t total_blocks = (n_elements + 32 - 1) / 32;
        size_t blocks_done = 0;

        while (blocks_done < total_blocks) {
            size_t batch_blocks = std::min(blocks_per_chunk, total_blocks - blocks_done);
            size_t read_bytes = batch_blocks * block_bytes;
            ssize_t n = pread(fd, chunk_buf.data(), read_bytes, file_cur + blocks_done * block_bytes);
            if (n <= 0) break;

            size_t actual_blocks = n / block_bytes;
            size_t batch_elements = actual_blocks * 32;
            if (elements_processed + batch_elements > n_elements) {
                batch_elements = n_elements - elements_processed;
            }

            dequantize_row_q5_0(chunk_buf.data(), out_buffer + elements_processed, actual_blocks * 32);
            elements_processed += batch_elements;
            blocks_done += actual_blocks;
            if (actual_blocks < batch_blocks) break;
        }
        close(fd);
        return static_cast<int>(elements_processed);
    }

    close(fd);
    return -6;
}

STREAMVEX_API int compute_gemv_benchmark(
    int rows,
    int cols,
    char* out_json_stats,
    size_t max_len
) {
    if (rows <= 0) rows = 2048;
    if (cols <= 0) cols = 2048;

    std::vector<float> x(cols, 1.0f);
    std::vector<float> y_f32(rows, 0.0f);
    std::vector<float> y_q4(rows, 0.0f);
    std::vector<float> y_q8(rows, 0.0f);
    std::vector<float> A_f32(rows * cols, 0.01f);

    int blocks_per_row = cols / streamvex::QK4_0;
    std::vector<streamvex::block_q4_0> A_q4(rows * blocks_per_row);
    for (auto& b : A_q4) {
        b.d = streamvex::fp32_to_fp16(0.5f);
        std::memset(b.qs, 0x12, sizeof(b.qs));
    }

    std::vector<streamvex::block_q8_0> A_q8(rows * blocks_per_row);
    for (auto& b : A_q8) {
        b.d = streamvex::fp32_to_fp16(0.5f);
        std::memset(b.qs, 1, sizeof(b.qs));
    }

    const int warmup = 3;
    const int iters = 15;

    for (int i = 0; i < warmup; ++i) {
        streamvex::gemv_f32(A_f32.data(), x.data(), y_f32.data(), rows, cols);
        streamvex::gemv_q4_0_f32(A_q4.data(), x.data(), y_q4.data(), rows, cols);
        streamvex::gemv_q8_0_f32(A_q8.data(), x.data(), y_q8.data(), rows, cols);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        streamvex::gemv_f32(A_f32.data(), x.data(), y_f32.data(), rows, cols);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_f32 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        streamvex::gemv_q4_0_f32(A_q4.data(), x.data(), y_q4.data(), rows, cols);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_q4 = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        streamvex::gemv_q8_0_f32(A_q8.data(), x.data(), y_q8.data(), rows, cols);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_q8 = std::chrono::duration<double, std::milli>(t5 - t4).count() / iters;

    double gflops_ops = (2.0 * rows * cols) / 1e9;
    double gflops_f32 = gflops_ops / (ms_f32 / 1000.0);
    double gflops_q4 = gflops_ops / (ms_q4 / 1000.0);
    double gflops_q8 = gflops_ops / (ms_q8 / 1000.0);

    double bw_f32_gb = ((static_cast<double>(rows) * cols * sizeof(float)) / 1e9) / (ms_f32 / 1000.0);
    double bw_q4_gb = ((static_cast<double>(rows) * blocks_per_row * sizeof(streamvex::block_q4_0)) / 1e9) / (ms_q4 / 1000.0);
    double bw_q8_gb = ((static_cast<double>(rows) * blocks_per_row * sizeof(streamvex::block_q8_0)) / 1e9) / (ms_q8 / 1000.0);

    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    json << "{\n"
         << "  \"matrix\": \"" << rows << "x" << cols << "\",\n"
         << "  \"f32_ms\": " << ms_f32 << ",\n"
         << "  \"f32_gflops\": " << gflops_f32 << ",\n"
         << "  \"f32_bandwidth_gb_s\": " << bw_f32_gb << ",\n"
         << "  \"q4_ms\": " << ms_q4 << ",\n"
         << "  \"q4_gflops\": " << gflops_q4 << ",\n"
         << "  \"q4_bandwidth_gb_s\": " << bw_q4_gb << ",\n"
         << "  \"q8_ms\": " << ms_q8 << ",\n"
         << "  \"q8_gflops\": " << gflops_q8 << ",\n"
         << "  \"q8_bandwidth_gb_s\": " << bw_q8_gb << ",\n"
         << "  \"speedup_q4_vs_f32\": " << (ms_f32 / std::max(0.001, ms_q4)) << "\n"
         << "}";

    std::string json_str = json.str();
    if (out_json_stats && max_len > 0) {
        size_t copy_len = std::min(json_str.size(), max_len - 1);
        std::memcpy(out_json_stats, json_str.c_str(), copy_len);
        out_json_stats[copy_len] = '\0';
    }
    return 0;
}

STREAMVEX_API int run_forward_pass_step(
    int token_id,
    int pos,
    int* out_next_token,
    float* out_ms
) {
    if (!out_next_token) return -1;
    if (!g_engine.is_ready()) {
        *out_next_token = 0;
        if (out_ms) *out_ms = 0.0f;
        return -2;
    }

    return g_engine.forward_pass_step(token_id, pos, out_next_token, out_ms);
}

// ============================================================================
// Version 2.0 Extended Adaptive Multi-Model & Hardware Intelligence APIs
// ============================================================================

STREAMVEX_API void streamvex_abort_generation(void) {
    g_engine.abort_generation();
}

STREAMVEX_API int streamvex_get_resource_telemetry(char* out_json, size_t max_len) {
    if (!out_json || max_len == 0) return -1;

    std::string json_str = streamvex::ResourceIntelligence::instance().to_json_telemetry();
    size_t copy_len = std::min(json_str.size(), max_len - 1);
    std::memcpy(out_json, json_str.c_str(), copy_len);
    out_json[copy_len] = '\0';
    return 0;
}

STREAMVEX_API int streamvex_get_device_profile(char* out_json, size_t max_len) {
    if (!out_json || max_len == 0) return -1;

    std::string json_str = streamvex::DeviceProfileManager::to_json_report();
    size_t copy_len = std::min(json_str.size(), max_len - 1);
    std::memcpy(out_json, json_str.c_str(), copy_len);
    out_json[copy_len] = '\0';
    return 0;
}

STREAMVEX_API int streamvex_validate_model(
    const char* model_path,
    int context_len,
    char* out_json_result,
    size_t max_len
) {
    if (!model_path || !out_json_result || max_len == 0) return -1;

    struct stat st;
    double file_mb = 0.0;
    if (stat(model_path, &st) == 0) {
        file_mb = static_cast<double>(st.st_size) / (1024.0 * 1024.0);
    }

    auto verdict = streamvex::DeviceProfileManager::validate_model_compatibility(file_mb, context_len);

    std::ostringstream ss;
    ss << "{\n"
       << "  \"is_compatible\": " << (verdict.is_compatible ? "true" : "false") << ",\n"
       << "  \"message\": \"" << verdict.message << "\",\n"
       << "  \"suggested_model\": \"" << verdict.suggested_model_size << "\",\n"
       << "  \"suggested_context\": " << verdict.suggested_context << "\n"
       << "}";

    std::string json_str = ss.str();
    size_t copy_len = std::min(json_str.size(), max_len - 1);
    std::memcpy(out_json_result, json_str.c_str(), copy_len);
    out_json_result[copy_len] = '\0';
    return 0;
}

STREAMVEX_API void streamvex_on_trim_memory(int trim_level) {
    streamvex::ResourceIntelligence::instance().on_trim_memory(trim_level);
    streamvex::HierarchicalMemoryManager::instance().handle_memory_pressure_recovery();
    streamvex::DualModelRuntime::instance().on_memory_pressure(
        trim_level >= 15 ? streamvex::MemoryPressure::CRITICAL : streamvex::MemoryPressure::MODERATE
    );
}

STREAMVEX_API void streamvex_set_background_mode(int is_background) {
    g_background_mode = (is_background != 0);
    auto budget = streamvex::ResourceIntelligence::instance().compute_budget(0.0, !g_background_mode);
    streamvex::ComputeScheduler::instance().set_worker_cap(budget.recommended_threads);
}

STREAMVEX_API int streamvex_voice_process_vad(
    const int16_t* pcm16,
    size_t num_samples,
    int* out_is_speech,
    float* out_prob,
    float* out_energy_db
) {
    if (!pcm16 || num_samples == 0 || !out_is_speech) return -1;

    auto res = streamvex::DualModelRuntime::instance().process_voice_activity(pcm16, num_samples);
    *out_is_speech = res.is_speech ? 1 : 0;
    if (out_prob) *out_prob = res.speech_probability;
    if (out_energy_db) *out_energy_db = res.energy_db;
    return 0;
}

STREAMVEX_API int streamvex_voice_synthesize(
    const char* text,
    int16_t* out_pcm16,
    size_t max_samples,
    int* out_actual_samples
) {
    if (!text || !out_pcm16 || max_samples == 0 || !out_actual_samples) return -1;

    std::vector<int16_t> audio;
    bool ok = streamvex::DualModelRuntime::instance().synthesize_voice(text, audio);
    if (!ok || audio.empty()) return -2;

    size_t copy_count = std::min(audio.size(), max_samples);
    std::memcpy(out_pcm16, audio.data(), copy_count * sizeof(int16_t));
    *out_actual_samples = static_cast<int>(copy_count);
    return 0;
}

STREAMVEX_API int streamvex_voice_transcribe(
    const int16_t* pcm16,
    size_t num_samples,
    char* out_text,
    size_t max_len
) {
    if (!pcm16 || num_samples == 0 || !out_text || max_len == 0) return -1;

    std::string text = streamvex::DualModelRuntime::instance().transcribe_voice(pcm16, num_samples);
    size_t copy_len = std::min(text.size(), max_len - 1);
    std::memcpy(out_text, text.c_str(), copy_len);
    out_text[copy_len] = '\0';
    return 0;
}

STREAMVEX_API int streamvex_benchmark_triple_buffering(
    const char* model_path,
    size_t chunk_size_mb,
    int total_layers,
    char* out_json,
    size_t max_len
) {
    if (!out_json || max_len == 0) return -1;
    if (chunk_size_mb == 0) chunk_size_mb = 64;
    if (total_layers <= 0) total_layers = 16;

    std::string path = (model_path && strlen(model_path) > 0) ? model_path : "";
    size_t chunk_bytes = chunk_size_mb * 1024 * 1024;

    auto res_double = streamvex::HierarchicalMemoryManager::instance().benchmark_buffering(
        path,
        streamvex::BufferMode::DOUBLE_BUFFER,
        chunk_bytes,
        total_layers
    );

    auto res_triple = streamvex::HierarchicalMemoryManager::instance().benchmark_buffering(
        path,
        streamvex::BufferMode::TRIPLE_BUFFER,
        chunk_bytes,
        total_layers
    );

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "{\n"
       << "  \"double_buffer\": {\n"
       << "    \"throughput_mb_s\": " << res_double.throughput_mb_s << ",\n"
       << "    \"avg_latency_ms\": " << res_double.avg_prefetch_latency_ms << ",\n"
       << "    \"peak_ram_mb\": " << res_double.peak_ram_mb << ",\n"
       << "    \"duration_sec\": " << res_double.duration_sec << "\n"
       << "  },\n"
       << "  \"triple_buffer\": {\n"
       << "    \"throughput_mb_s\": " << res_triple.throughput_mb_s << ",\n"
       << "    \"avg_latency_ms\": " << res_triple.avg_prefetch_latency_ms << ",\n"
       << "    \"peak_ram_mb\": " << res_triple.peak_ram_mb << ",\n"
       << "    \"duration_sec\": " << res_triple.duration_sec << "\n"
       << "  },\n"
       << "  \"recommended_mode\": \"" << (res_triple.throughput_mb_s > res_double.throughput_mb_s * 1.15 ? "triple" : "double") << "\"\n"
       << "}";

    std::string json_str = ss.str();
    size_t copy_len = std::min(json_str.size(), max_len - 1);
    std::memcpy(out_json, json_str.c_str(), copy_len);
    out_json[copy_len] = '\0';
    return 0;
}
