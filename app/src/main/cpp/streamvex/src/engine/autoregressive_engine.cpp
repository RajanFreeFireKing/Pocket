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

    if (model_path.empty()) {
        std::cerr << "[AutoregressiveEngine] Error: Empty model path provided.\n";
        return false;
    }

    // Instant Zero-Wait Cache Reuse: keep loaded model in memory across turns (0ms reload)
    if (m_is_initialized.load() && m_model != nullptr && m_model_path == model_path && m_ctx != nullptr) {
        llama_memory_clear(llama_get_memory(m_ctx), true);
        m_cached_tokens.clear();
        m_abort_requested.store(false);
        std::cout << "[AutoregressiveEngine] Instant cache reuse for loaded model: " << model_path << "\n";
        return true;
    }

    release_internal();

    struct stat st;
    if (stat(model_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        std::cerr << "[AutoregressiveEngine] Error: Model file not found on disk: " << model_path << "\n";
        return false;
    }

    m_telemetry.architecture = "GGUF";

    // Initialize llama.cpp backend
    llama_backend_init();

    // Model configuration (Rule 1 & 3: Instant zero-wait mmap loading without mlock)
    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;

    m_model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!m_model) {
        std::cerr << "[AutoregressiveEngine] Error: llama_model_load_from_file failed for: " << model_path << "\n";
        return false;
    }

    // Smart Big-Core Burst Scheduling: 4 decode threads, 6 batch threads across big/mid cores
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 8;
    m_n_threads = std::min(4, static_cast<int>(hw));
    int n_threads_batch = std::min(6, static_cast<int>(hw));

    // Vectorized context (512) and batch (512) with Quantized Q8_0 KV Cache + Flash Attention
    m_n_ctx = (n_ctx > 0 && n_ctx <= 512) ? n_ctx : 512;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = m_n_ctx;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_threads = m_n_threads;
    cparams.n_threads_batch = n_threads_batch;
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    cparams.flash_attn = true;

    m_ctx = llama_init_from_model(m_model, cparams);
    if (!m_ctx) {
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
        cparams.flash_attn = false;
        m_ctx = llama_init_from_model(m_model, cparams);
        if (!m_ctx) {
            std::cerr << "[AutoregressiveEngine] Error: llama_init_from_model failed.\n";
            llama_model_free(m_model);
            m_model = nullptr;
            return false;
        }
    }

    m_model_path = model_path;

    m_telemetry = GenerationTelemetry();
    m_telemetry.model_name = model_path;
    m_telemetry.baseline_ram_mb = ResourceIntelligence::instance().get_memory_info().process_rss_mb;
    m_telemetry.peak_ram_mb = m_telemetry.baseline_ram_mb;
    m_telemetry.current_ram_mb = m_telemetry.baseline_ram_mb;
    m_telemetry.is_ram_bounded = true;

    m_abort_requested.store(false);
    m_is_initialized.store(true);

    const struct llama_vocab * vocab = llama_model_get_vocab(m_model);
    std::cout << "[AutoregressiveEngine] Loaded authentic model: " << model_path
              << " | Vocab: " << (vocab ? llama_vocab_n_tokens(vocab) : 0)
              << " | Context: " << m_n_ctx
              << " | Threads: " << m_n_threads
              << " | Batch Threads: " << n_threads_batch
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

    // Smart Big-Core Burst Scheduling: burst across big/mid cores (0..7) with nice(0)
#if defined(__linux__) || defined(__ANDROID__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int c = 0; c < 8; ++c) {
        CPU_SET(c, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    nice(0);
#endif

    m_abort_requested.store(false);
    auto t_start = std::chrono::high_resolution_clock::now();

    const struct llama_vocab * vocab = llama_model_get_vocab(m_model);
    if (!vocab) {
        callback("[StreamVex Error]: Failed to get model vocabulary.");
        return -4;
    }

    // Dynamic Chat Template Detection: Llama 3.2 Instruct, ChatML, Gemma
    bool is_already_templated = (prompt.find("<|start_header_id|>") != std::string::npos) ||
                                (prompt.find("<|im_start|>") != std::string::npos) ||
                                (prompt.find("<start_of_turn>") != std::string::npos) ||
                                (prompt.find("[INST]") != std::string::npos);

    std::string formatted_prompt;
    if (is_already_templated) {
        formatted_prompt = prompt;
    } else {
        const char* model_chat_tmpl = llama_model_chat_template(m_model, nullptr);
        std::string tmpl_str = model_chat_tmpl ? std::string(model_chat_tmpl) : "";

        // Check if vocabulary has Llama 3 tokens
        llama_token llama3_hdr = -1;
        int n_l3 = llama_tokenize(vocab, "<|start_header_id|>", 19, &llama3_hdr, 1, false, true);
        bool has_llama3 = (n_l3 == 1 && llama3_hdr >= 0) ||
                          (tmpl_str.find("start_header_id") != std::string::npos) ||
                          (tmpl_str.find("eot_id") != std::string::npos);

        // Check if vocabulary has ChatML tokens
        llama_token chatml_hdr = -1;
        int n_cml = llama_tokenize(vocab, "<|im_start|>", 12, &chatml_hdr, 1, false, true);
        bool has_chatml = (n_cml == 1 && chatml_hdr >= 0) ||
                          (tmpl_str.find("im_start") != std::string::npos);

        // Check if vocabulary has Gemma tokens
        llama_token gemma_hdr = -1;
        int n_gemma = llama_tokenize(vocab, "<start_of_turn>", 15, &gemma_hdr, 1, false, true);
        bool has_gemma = (n_gemma == 1 && gemma_hdr >= 0);

        if (has_llama3) {
            // Authentic Llama 3.2 1B Instruct format
            if (prompt.find("User: ") != std::string::npos && prompt.find("Assistant:") != std::string::npos) {
                std::istringstream stream(prompt);
                std::string line;
                std::string current_role = "";
                std::string current_content = "";
                std::ostringstream out;
                out << "<|begin_of_text|>";

                auto flush_turn = [&](const std::string& role, const std::string& content) {
                    if (role.empty() || content.empty()) return;
                    out << "<|start_header_id|>" << role << "<|end_header_id|>\n\n" << content << "<|eot_id|>";
                };

                while (std::getline(stream, line)) {
                    if (line.compare(0, 6, "User: ") == 0) {
                        flush_turn(current_role, current_content);
                        current_role = "user";
                        current_content = line.substr(6);
                    } else if (line.compare(0, 10, "Assistant:") == 0) {
                        flush_turn(current_role, current_content);
                        current_role = "assistant";
                        current_content = line.length() > 11 ? line.substr(11) : "";
                    } else if (!current_role.empty()) {
                        if (!current_content.empty()) current_content += "\n";
                        current_content += line;
                    }
                }
                flush_turn(current_role, current_content);
                out << "<|start_header_id|>assistant<|end_header_id|>\n\n";
                formatted_prompt = out.str();
            } else {
                formatted_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
            }
        } else if (has_gemma) {
            formatted_prompt = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
        } else {
            formatted_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        }
    }

    // Authentic Tokenization via Model Vocabulary
    int n_tokens_alloc = static_cast<int>(formatted_prompt.size()) + 64;
    std::vector<llama_token> prompt_tokens(n_tokens_alloc);
    int n_tokens = llama_tokenize(
        vocab,
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
            vocab,
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

    // KV Cache Prefix Matching & Context Reuse
    size_t common_len = 0;
    while (common_len < m_cached_tokens.size() &&
           common_len < prompt_tokens.size() &&
           m_cached_tokens[common_len] == prompt_tokens[common_len]) {
        common_len++;
    }

    int n_past = 0;
    if (common_len > 0 && prompt_tokens.size() < static_cast<size_t>(m_n_ctx - 16)) {
        if (common_len < m_cached_tokens.size()) {
            llama_memory_seq_rm(llama_get_memory(m_ctx), 0, static_cast<llama_pos>(common_len), -1);
            m_cached_tokens.resize(common_len);
        }
        n_past = static_cast<int>(common_len);
    } else {
        llama_memory_clear(llama_get_memory(m_ctx), true);
        m_cached_tokens.clear();
        n_past = 0;
    }

    float ttft_ms = 0.0f;
    int n_tokens_to_eval = static_cast<int>(prompt_tokens.size()) - n_past;

    // Prefill Phase: Process prompt in a single vectorized batch (n_batch = 512)
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    llama_batch batch = llama_batch_init(512, 0, 1);

    if (n_tokens_to_eval > 0) {
        int eval_start = n_past;
        for (int i = 0; i < n_tokens_to_eval; i += 512) {
            if (m_abort_requested.load()) {
                llama_batch_free(batch);
                return 0;
            }

            int n_eval = std::min(512, n_tokens_to_eval - i);
            batch.n_tokens = 0;
            for (int j = 0; j < n_eval; ++j) {
                int token_idx = eval_start + i + j;
                batch.token   [batch.n_tokens] = prompt_tokens[token_idx];
                batch.pos     [batch.n_tokens] = token_idx;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id  [batch.n_tokens][0] = 0;
                batch.logits  [batch.n_tokens] = (token_idx == static_cast<int>(prompt_tokens.size()) - 1) ? 1 : 0;
                batch.n_tokens++;
            }

            if (llama_decode(m_ctx, batch) != 0) {
                llama_batch_free(batch);
                callback("[StreamVex Error]: Prompt prefill failed in llama_decode.");
                return -5;
            }
            n_past += n_eval;
        }
    }

    m_cached_tokens = prompt_tokens;

    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_sec = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    m_telemetry.prefill_tok_per_sec = prefill_sec > 0.0001 ? static_cast<float>(n_tokens_to_eval / prefill_sec) : 0.0f;

    // Autoregressive Generation Loop with modern llama_sampler_chain
    int n_vocab = llama_vocab_n_tokens(vocab);
    llama_token eos_token = llama_vocab_eos(vocab);
    int gen_count = 0;
    int max_gen = (max_tokens > 0 && max_tokens <= 128) ? max_tokens : 128;

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    if (top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    }
    if (temperature > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    for (int i = 0; i < max_gen; ++i) {
        if (m_abort_requested.load()) {
            break;
        }

        // Lightweight cooperative yield every 16 tokens (zero sleep_for delays)
        if (gen_count > 0 && gen_count % 16 == 0) {
            sched_yield();
        }

        // Cooperative scheduling: yield to voice AI if high priority work is active
        if (ComputeScheduler::instance().should_yield(WorkloadPriority::P3_LLM_INFERENCE)) {
            sched_yield();
        }

        llama_token new_token_id = llama_sampler_sample(smpl, m_ctx, -1);
        llama_sampler_accept(smpl, new_token_id);

        if (i == 0) {
            auto t_first = std::chrono::high_resolution_clock::now();
            ttft_ms = std::chrono::duration<float, std::milli>(t_first - t_start).count();
        }

        // Stop on EOS or EOG
        if (new_token_id == eos_token || llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Decode token to UTF-8 text chunk
        char piece_buf[128] = {0};
        int n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf) - 1, 0, false);
        if (n_chars > 0) {
            piece_buf[n_chars] = '\0';
            std::string piece_str(piece_buf);

            // Comprehensive stop tokens check
            if (piece_str.find("<|eot_id|>") != std::string::npos ||
                piece_str.find("<|endoftext|>") != std::string::npos ||
                piece_str.find("<|end_of_text|>") != std::string::npos ||
                piece_str.find("<|im_end|>") != std::string::npos ||
                piece_str.find("<|im_start|>") != std::string::npos ||
                piece_str.find("<|start_header_id|>") != std::string::npos ||
                piece_str.find("<|end_header_id|>") != std::string::npos ||
                piece_str.find("<end_of_turn>") != std::string::npos ||
                piece_str.find("end_of_turn") != std::string::npos ||
                piece_str.find("</s>") != std::string::npos ||
                piece_str.find("<|turn_end|>") != std::string::npos) {
                break;
            }

            // Strip special control tokens from user-visible response
            static const std::string SPECIAL_PATTERNS[] = {
                "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
                "<|eot_id|>", "<|im_start|>", "<|im_end|>",
                "<start_of_turn>", "<end_of_turn>", "end_of_turn",
                "<|endoftext|>", "<|end_of_text|>"
            };
            for (const auto& pattern : SPECIAL_PATTERNS) {
                size_t pos = 0;
                while ((pos = piece_str.find(pattern, pos)) != std::string::npos) {
                    piece_str.erase(pos, pattern.length());
                }
            }

            if (!piece_str.empty()) {
                callback(piece_str.c_str());
            }
        }

        m_cached_tokens.push_back(new_token_id);
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

        if (n_past >= m_n_ctx || llama_decode(m_ctx, batch) != 0) {
            break;
        }
    }

    llama_sampler_free(smpl);
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

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_token next_token = llama_sampler_sample(smpl, m_ctx, -1);
    llama_sampler_accept(smpl, next_token);
    llama_sampler_free(smpl);

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

void AutoregressiveEngine::reset_context() {
    std::lock_guard<std::mutex> lock(m_engine_mtx);
    if (m_ctx) {
        llama_memory_clear(llama_get_memory(m_ctx), true);
    }
    m_cached_tokens.clear();
}

void AutoregressiveEngine::release() {
    std::lock_guard<std::mutex> lock(m_engine_mtx);
    release_internal();
}

void AutoregressiveEngine::release_internal() {
    m_is_initialized.store(false);
    m_cached_tokens.clear();
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model) {
        llama_model_free(m_model);
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
