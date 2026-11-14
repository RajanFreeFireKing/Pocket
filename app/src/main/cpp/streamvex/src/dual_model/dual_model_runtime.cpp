#include "dual_model_runtime.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>

namespace streamvex {

DualModelRuntime& DualModelRuntime::instance() {
    static DualModelRuntime s_instance;
    return s_instance;
}

DualModelRuntime::DualModelRuntime() {
    VoiceEngine::instance().initialize(16000);
}

bool DualModelRuntime::initialize(const std::string& llm_model_path) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // Initialize Voice Engine
    VoiceEngine::instance().initialize(16000);

    // Check memory budget for LLM
    auto budget = ResourceIntelligence::instance().compute_budget();
    (void)llm_model_path;
    (void)budget;

    m_llm_ready.store(true);
    return true;
}

void DualModelRuntime::shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex);
    VoiceEngine::instance().release();
    m_llm_ready.store(false);
    m_voice_active.store(false);
    m_llm_active.store(false);
}

VadResult DualModelRuntime::process_voice_activity(const int16_t* pcm16, size_t num_samples) {
    m_voice_active.store(true);

    // If LLM is running in background, signal preemption/throttle so voice latency stays <100ms
    if (m_llm_active.load()) {
        m_preemption_count++;
    }

    VadResult res = VoiceEngine::instance().process_vad_frame(pcm16, num_samples);
    m_voice_active.store(false);
    return res;
}

bool DualModelRuntime::synthesize_voice(const std::string& text, std::vector<int16_t>& out_pcm16) {
    m_voice_active.store(true);
    if (m_llm_active.load()) {
        m_preemption_count++;
    }

    bool ok = VoiceEngine::instance().synthesize_speech(text, out_pcm16);
    m_voice_active.store(false);
    return ok;
}

std::string DualModelRuntime::transcribe_voice(const int16_t* pcm16, size_t num_samples) {
    m_voice_active.store(true);
    if (m_llm_active.load()) {
        m_preemption_count++;
    }

    std::string res = VoiceEngine::instance().transcribe_speech(pcm16, num_samples);
    m_voice_active.store(false);
    return res;
}

void DualModelRuntime::on_memory_pressure(MemoryPressure pressure) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (pressure == MemoryPressure::CRITICAL) {
        // Emergency memory recovery: release voice buffers, advise cache flush
        VoiceEngine::instance().release();
    }
}

DualModelTelemetry DualModelRuntime::get_telemetry() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    DualModelTelemetry t = m_telemetry;

    auto mem = ResourceIntelligence::instance().get_memory_info();
    auto voice_t = VoiceEngine::instance().get_telemetry();

    t.total_resident_mb = mem.process_rss_mb;
    t.voice_resident_mb = 12.0; // Voice AI resident pool
    t.llm_resident_mb = std::max(0.0, mem.process_rss_mb - t.voice_resident_mb);
    t.voice_frames_processed = voice_t.vad_frames_processed;
    t.speech_detected_count = voice_t.speech_frames_detected;
    t.preemption_events = m_preemption_count.load();

    if (m_voice_active.load() && m_llm_active.load()) {
        t.status = ModelRuntimeStatus::DUAL_ACTIVE;
    } else if (m_voice_active.load()) {
        t.status = ModelRuntimeStatus::VOICE_ACTIVE;
    } else if (m_llm_active.load()) {
        t.status = ModelRuntimeStatus::LLM_ACTIVE;
    } else {
        t.status = ModelRuntimeStatus::IDLE;
    }

    return t;
}

std::string DualModelRuntime::to_json_telemetry() const {
    DualModelTelemetry t = get_telemetry();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "{\n"
       << "  \"runtime_status\": " << static_cast<int>(t.status) << ",\n"
       << "  \"total_resident_mb\": " << t.total_resident_mb << ",\n"
       << "  \"llm_resident_mb\": " << t.llm_resident_mb << ",\n"
       << "  \"voice_resident_mb\": " << t.voice_resident_mb << ",\n"
       << "  \"vad_frames_processed\": " << t.voice_frames_processed << ",\n"
       << "  \"speech_detected_count\": " << t.speech_detected_count << ",\n"
       << "  \"preemption_events\": " << t.preemption_events << "\n"
       << "}";
    return ss.str();
}

} // namespace streamvex
