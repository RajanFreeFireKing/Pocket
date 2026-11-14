#ifndef DUAL_MODEL_RUNTIME_HPP
#define DUAL_MODEL_RUNTIME_HPP

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>

#include "voice_engine.hpp"
#include "../resource/resource_intelligence.hpp"
#include "../scheduler/compute_scheduler.hpp"

namespace streamvex {

enum class ModelRuntimeStatus {
    IDLE = 0,
    LLM_ACTIVE = 1,
    VOICE_ACTIVE = 2,
    DUAL_ACTIVE = 3,
    THROTTLED = 4,
    ERROR_STATE = 5
};

struct DualModelTelemetry {
    ModelRuntimeStatus status = ModelRuntimeStatus::IDLE;
    double llm_resident_mb = 0.0;
    double voice_resident_mb = 0.0;
    double total_resident_mb = 0.0;
    int llm_tokens_generated = 0;
    float llm_tokens_per_sec = 0.0f;
    int voice_frames_processed = 0;
    int speech_detected_count = 0;
    int preemption_events = 0;
    bool is_throttled = false;
};

/**
 * Multi-Model Runtime Manager orchestrating concurrent Large Language Model
 * and Offline Voice AI workloads on mobile devices.
 */
class DualModelRuntime {
public:
    static DualModelRuntime& instance();

    DualModelRuntime();
    ~DualModelRuntime() = default;

    DualModelRuntime(const DualModelRuntime&) = delete;
    DualModelRuntime& operator=(const DualModelRuntime&) = delete;

    /**
     * Initializes the dual model subsystem.
     */
    bool initialize(const std::string& llm_model_path);
    void shutdown();

    bool is_llm_ready() const { return m_llm_ready.load(); }
    bool is_voice_ready() const { return VoiceEngine::instance().is_ready(); }

    /**
     * Executes Voice Activity Detection on a PCM16 audio frame.
     * Higher priority (P2) - automatically signals background LLM to yield if needed.
     */
    VadResult process_voice_activity(const int16_t* pcm16, size_t num_samples);

    /**
     * Synthesizes text to offline speech waveform.
     */
    bool synthesize_voice(const std::string& text, std::vector<int16_t>& out_pcm16);

    /**
     * Transcribes spoken audio into text.
     */
    std::string transcribe_voice(const int16_t* pcm16, size_t num_samples);

    /**
     * Delivers memory trim signal and reclaims non-essential resources.
     */
    void on_memory_pressure(MemoryPressure pressure);

    DualModelTelemetry get_telemetry() const;
    std::string to_json_telemetry() const;

private:
    mutable std::mutex m_mutex;
    std::atomic<bool> m_llm_ready{false};
    std::atomic<bool> m_voice_active{false};
    std::atomic<bool> m_llm_active{false};
    std::atomic<int> m_preemption_count{0};
    DualModelTelemetry m_telemetry;
};

} // namespace streamvex

#endif // DUAL_MODEL_RUNTIME_HPP
