#ifndef VOICE_ENGINE_HPP
#define VOICE_ENGINE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <functional>

namespace streamvex {

struct VadResult {
    bool is_speech = false;
    float speech_probability = 0.0f;
    float energy_db = -100.0f;
    float zero_crossing_rate = 0.0f;
};

struct VoiceTelemetry {
    int vad_frames_processed = 0;
    int speech_frames_detected = 0;
    float last_tts_latency_ms = 0.0f;
    float last_stt_latency_ms = 0.0f;
    int audio_underruns = 0;
    size_t audio_samples_generated = 0;
};

/**
 * Offline Voice AI Engine.
 * Provides Voice Activity Detection (VAD), Speech-to-Text decoding,
 * and Text-to-Speech audio synthesis.
 */
class VoiceEngine {
public:
    static VoiceEngine& instance();

    VoiceEngine();
    ~VoiceEngine() = default;

    VoiceEngine(const VoiceEngine&) = delete;
    VoiceEngine& operator=(const VoiceEngine&) = delete;

    /**
     * Initializes Voice AI models and audio buffers.
     * @param sample_rate Audio sample rate (default 16000 Hz).
     */
    bool initialize(int sample_rate = 16000);
    void release();

    bool is_ready() const { return m_is_initialized.load(); }

    /**
     * Processes a single audio frame (e.g. 10ms - 30ms) for Voice Activity Detection.
     * 
     * @param pcm16 Pointer to 16-bit mono PCM samples.
     * @param num_samples Number of samples (e.g. 160, 320, 480).
     * @return VadResult with classification and confidence.
     */
    VadResult process_vad_frame(const int16_t* pcm16, size_t num_samples);

    /**
     * Offline Speech-to-Text: decodes an audio buffer into text.
     * 
     * @param pcm16 Input PCM audio samples.
     * @param num_samples Number of audio samples.
     * @return Recognized text string.
     */
    std::string transcribe_speech(const int16_t* pcm16, size_t num_samples);

    /**
     * Offline Text-to-Speech: synthesizes input text into 16kHz PCM audio.
     * 
     * @param text Input text string.
     * @param out_pcm16 Output vector to receive synthesized 16-bit PCM audio samples.
     * @param sample_rate Sample rate (default 16000).
     * @return true on success, false otherwise.
     */
    bool synthesize_speech(const std::string& text, std::vector<int16_t>& out_pcm16, int sample_rate = 16000);

    VoiceTelemetry get_telemetry() const;

private:
    std::atomic<bool> m_is_initialized{false};
    int m_sample_rate = 16000;
    mutable std::mutex m_mutex;
    VoiceTelemetry m_telemetry;

    // VAD calibration parameters
    float m_noise_floor_energy = 0.001f;
    float m_speech_threshold = 0.015f;
};

} // namespace streamvex

#endif // VOICE_ENGINE_HPP
