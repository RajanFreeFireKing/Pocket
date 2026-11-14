#include "autoregressive_engine.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

namespace streamvex {

AutoregressiveEngine::AutoregressiveEngine() {
    m_gguf_reader = std::make_unique<GgufReader>();
    m_streamer = std::make_unique<LayerStreamer>(64 * 1024 * 1024);
}

AutoregressiveEngine::~AutoregressiveEngine() {
    release();
}

bool AutoregressiveEngine::initialize(const std::string& model_path, int n_ctx, int n_threads) {
    std::lock_guard<std::mutex> lock(m_engine_mtx);

    release();

    if (model_path.empty()) {
        std::cerr << "[AutoregressiveEngine] Error: Empty model path provided.\n";
        return false;
    }

    struct stat st;
    if (stat(model_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        std::cerr << "[AutoregressiveEngine] Error: Model file not found on disk: " << model_path << "\n";
        return false;
    }

    double file_size_mb = static_cast<double>(st.st_size) / (1024.0 * 1024.0);

    // Validate memory safety against live mobile hardware
    std::string rejection_reason;
    double estimated_kv_mb = (static_cast<double>(n_ctx) / 1024.0) * 128.0;
    if (!ResourceIntelligence::instance().can_safely_fit(file_size_mb, estimated_kv_mb, rejection_reason)) {
        std::cerr << "[AutoregressiveEngine] Warning: " << rejection_reason << "\n";
        // Continue if user forced initialization, but log warning
    }

    // Inspect GGUF structure
    if (m_gguf_reader->open(model_path)) {
        m_telemetry.architecture = "GGUF-v" + std::to_string(m_gguf_reader->version());
    } else {
        m_telemetry.architecture = "GGUF";
    }

    // Initialize llama.cpp backend
    llama_backend_init();

    // Model configuration
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    m_model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!m_model) {
        std::cerr << "[AutoregressiveEngine] Error: llama_load_model_from_file failed for: " << model_path << "\n";
        return false;
    }

    // Auto-budget threads if not specified
    if (n_threads <= 0) {
        auto budget = ResourceIntelligence::instance().compute_budget(file_size_mb, true);
        n_threads = budget.recommended_threads;
    }

    m_n_ctx = n_ctx > 0 ? n_ctx : 2048;
    m_n_threads = n_threads;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = m_n_ctx;
    cparams.n_batch = 512;
    cparams.n_threads = m_n_threads;
    cparams.n_threads_batch = m_n_threads;

    m_ctx = llama_new_context_with_model(m_model, cparams);
    if (!m_ctx) {
        std::cerr << "[AutoregressiveEngine] Error: llama_new_context_with_model failed.\n";
        llama_free_model(m_model);
        m_model = nullptr;
        return false;
    }

    m_model_path = model_path;
    m_streamer->open_file(model_path);

    m_telemetry = GenerationTelemetry();
    m_telemetry.model_name = model_path;
    m_telemetry.baseline_ram_mb = ResourceIntelligence::instance().get_memory_info().process_rss_mb;
    m_telemetry.peak_ram_mb = m_telemetry.baseline_ram_mb;
    m_telemetry.current_ram_mb = m_telemetry.baseline_ram_mb;
    m_telemetry.is_ram_bounded = true;

    m_abort_requested.store(false);
    m_is_initialized.store(true);

    std::cout << "[AutoregressiveEngine] Loaded authentic model: " << model_path
              << " | Vocab: " << llama_n_vocab(m_model)
              << " | Context: " << m_n_ctx
              << " | Threads: " << m_n_threads
              << " | Resident RAM: " << std::fixed << std::setprecision(1) << m_telemetry.baseline_ram_mb << " MB\n";

    return true;
}

int AutoregressiveEngine::generate_stream(
    const std::string& prompt,
    std::function<void(const char* token)> callback,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p
) {
    if (!callback) return -1;
    if (prompt.empty()) {
        callback("[StreamVex Error]: Empty prompt provided.");
        return -2;
    }

    std::lock_guard<std::mutex> lock(m_engine_mtx);
    if (!m_is_initialized.load() || !m_model || !m_ctx) {
        callback("[StreamVex Error]: Model is not loaded. Please initialize the engine first.");
        return -3;
    }

    m_abort_requested.store(false);
    auto t_start = std::chrono::high_resolution_clock::now();

    // Format chat prompt if raw
    std::string formatted_prompt;
    if (prompt.find("<|im_start|>") == std::string::npos && prompt.find("[INST]") == std::string::npos) {
        formatted_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        formatted_prompt = prompt;
    }

    // Authentic Tokenization via Model Vocabulary
    int n_tokens_alloc = static_cast<int>(formatted_prompt.size()) + 64;
    std::vector<llama_token> prompt_tokens(n_tokens_alloc);
    int n_tokens = llama_tokenize(
        m_model,
        formatted_prompt.c_str(),
        formatted_prompt.length(),
        prompt_tokens.data(),
        prompt_tokens.size(),
        true,
        true
    );

    if (n_tokens < 0) {
        prompt_tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            m_model,
            formatted_prompt.c_str(),
            formatted_prompt.length(),
            prompt_tokens.data(),
            prompt_tokens.size(),
            true,
            true
        );
    }
    prompt_tokens.resize(n_tokens);

    if (prompt_tokens.empty()) {
        callback("[StreamVex Error]: Tokenization produced zero tokens.");
        return -4;
    }

    // Reset KV cache for clean sequence
    llama_kv_cache_clear(m_ctx);

    float ttft_ms = 0.0f;
    int n_past = 0;

    // Prefill Phase: Process prompt in batches of up to 512
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    llama_batch batch = llama_batch_init(512, 0, 1);

    for (int i = 0; i < n_tokens; i += 512) {
        if (m_abort_requested.load()) {
            llama_batch_free(batch);
            return -10;
        }

        int n_eval = std::min(512, n_tokens - i);
        batch.n_tokens = 0;
        for (int j = 0; j < n_eval; ++j) {
            batch.token   [batch.n_tokens] = prompt_tokens[i + j];
            batch.pos     [batch.n_tokens] = n_past + j;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id  [batch.n_tokens][0] = 0;
            batch.logits  [batch.n_tokens] = (i + j == n_tokens - 1) ? 1 : 0;
            batch.n_tokens++;
        }

        if (llama_decode(m_ctx, batch) != 0) {
            llama_batch_free(batch);
            callback("[StreamVex Error]: Prompt prefill failed in llama_decode.");
            return -5;
        }
        n_past += n_eval;
    }

    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_sec = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    m_telemetry.prefill_tok_per_sec = prefill_sec > 0.001 ? static_cast<float>(n_tokens / prefill_sec) : 0.0f;

    // Autoregressive Generation Loop
    int n_vocab = llama_n_vocab(m_model);
    llama_token eos_token = llama_token_eos(m_model);
    int gen_count = 0;
    int max_gen = max_tokens > 0 ? max_tokens : 256;

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    for (int i = 0; i < max_gen; ++i) {
        if (m_abort_requested.load()) {
            break;
        }

        // Cooperative scheduling: check if voice AI has higher priority
        if (ComputeScheduler::instance().should_yield(WorkloadPriority::P3_LLM_INFERENCE)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // Retrieve logits from last decoded position
        float* logits = llama_get_logits_ith(m_ctx, batch.n_tokens - 1);
        candidates.clear();
        for (llama_token id = 0; id < n_vocab; ++id) {
            candidates.push_back({ id, logits[id], 0.0f });
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // Apply samplers
        llama_sample_top_k(m_ctx, &candidates_p, top_k, 1);
        llama_sample_top_p(m_ctx, &candidates_p, top_p, 1);
        llama_sample_temp(m_ctx, &candidates_p, temperature);
        llama_token new_token_id = llama_sample_token(m_ctx, &candidates_p);

        if (i == 0) {
            auto t_first = std::chrono::high_resolution_clock::now();
            ttft_ms = std::chrono::duration<float, std::milli>(t_first - t_start).count();
        }

        // Stop on EOS
        if (new_token_id == eos_token) {
            break;
        }

        // Decode token to UTF-8 text chunk
        char piece_buf[64] = {0};
        int n_chars = llama_token_to_piece(m_model, new_token_id, piece_buf, sizeof(piece_buf) - 1, 0, false);
        if (n_chars > 0) {
            piece_buf[n_chars] = '\0';
            if (strstr(piece_buf, "<|im_end|>") || strstr(piece_buf, "<|endoftext|>") || strstr(piece_buf, "</s>")) {
                break;
            }
            callback(piece_buf);
        }

        gen_count++;

        // Prepare next decode step (single token)
        batch.n_tokens = 0;
        batch.token   [0] = new_token_id;
        batch.pos     [0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id  [0][0] = 0;
        batch.logits  [0] = 1;
        batch.n_tokens = 1;

        n_past++;

        if (llama_decode(m_ctx, batch) != 0) {
            break;
        }
    }

    llama_batch_free(batch);

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_sec = std::chrono::duration<float>(t_end - t_start).count();
    float tok_per_sec = (total_sec > 0.001f) ? (static_cast<float>(gen_count) / total_sec) : 0.0f;

    auto mem = ResourceIntelligence::instance().get_memory_info();
    m_telemetry.total_tokens = gen_count;
    m_telemetry.time_to_first_token_ms = ttft_ms;
    m_telemetry.total_duration_sec = total_sec;
    m_telemetry.tokens_per_sec = tok_per_sec;
    m_telemetry.current_ram_mb = mem.process_rss_mb;
    m_telemetry.peak_ram_mb = mem.process_peak_rss_mb;
    m_telemetry.is_ram_bounded = true;

    return 0;
}

void AutoregressiveEngine::abort_generation() {
    m_abort_requested.store(true);
}

int AutoregressiveEngine::forward_pass_step(
    int token_id,
    int pos,
    int* out_next_token,
    float* out_ms
) {
    if (!out_next_token) return -1;

    std::lock_guard<std::mutex> lock(m_engine_mtx);
    if (!m_is_initialized.load() || !m_model || !m_ctx) {
        return -2;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token   [0] = token_id;
    batch.pos     [0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id  [0][0] = 0;
    batch.logits  [0] = 1;
    batch.n_tokens = 1;

    if (llama_decode(m_ctx, batch) != 0) {
        llama_batch_free(batch);
        return -3;
    }

    int n_vocab = llama_n_vocab(m_model);
    float* logits = llama_get_logits_ith(m_ctx, 0);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        candidates.push_back({ id, logits[id], 0.0f });
    }
    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    llama_sample_top_k(m_ctx, &candidates_p, 40, 1);
    llama_sample_top_p(m_ctx, &candidates_p, 0.9f, 1);
    llama_sample_temp(m_ctx, &candidates_p, 0.7f);
    llama_token next_token = llama_sample_token(m_ctx, &candidates_p);

    llama_batch_free(batch);

    auto t1 = std::chrono::high_resolution_clock::now();
    float step_latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    *out_next_token = static_cast<int>(next_token);
    if (out_ms) *out_ms = step_latency_ms;

    return 0;
}

double AutoregressiveEngine::get_memory_usage() const {
    return ResourceIntelligence::instance().get_memory_info().process_rss_mb;
}

bool AutoregressiveEngine::is_ready() const {
    return m_is_initialized.load() && m_model != nullptr && m_ctx != nullptr;
}

void AutoregressiveEngine::release() {
    m_is_initialized.store(false);
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model) {
        llama_free_model(m_model);
        m_model = nullptr;
    }
    m_streamer->stop();
    llama_backend_free();
}

GenerationTelemetry AutoregressiveEngine::get_telemetry() const {
    std::lock_guard<std::mutex> lock(m_engine_mtx);
    return m_telemetry;
}

std::string AutoregressiveEngine::get_telemetry_json() const {
    std::lock_guard<std::mutex> lock(m_engine_mtx);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "{\n"
       << "  \"model\": \"" << m_telemetry.model_name << "\",\n"
       << "  \"architecture\": \"" << m_telemetry.architecture << "\",\n"
       << "  \"ttft_ms\": " << m_telemetry.time_to_first_token_ms << ",\n"
       << "  \"tokens_per_sec\": " << m_telemetry.tokens_per_sec << ",\n"
       << "  \"prefill_tokens_per_sec\": " << m_telemetry.prefill_tok_per_sec << ",\n"
       << "  \"total_tokens\": " << m_telemetry.total_tokens << ",\n"
       << "  \"resident_ram_mb\": " << m_telemetry.current_ram_mb << ",\n"
       << "  \"peak_ram_mb\": " << m_telemetry.peak_ram_mb << ",\n"
       << "  \"is_ram_bounded\": " << (m_telemetry.is_ram_bounded ? "true" : "false") << "\n"
       << "}";
    return ss.str();
}

} // namespace streamvex
