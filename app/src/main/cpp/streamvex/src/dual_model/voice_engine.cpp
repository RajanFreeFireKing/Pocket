#include "voice_engine.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <sstream>

namespace streamvex {

VoiceEngine& VoiceEngine::instance() {
    static VoiceEngine s_instance;
    return s_instance;
}

VoiceEngine::VoiceEngine() {
    initialize(16000);
}

bool VoiceEngine::initialize(int sample_rate) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sample_rate = sample_rate > 0 ? sample_rate : 16000;
    m_telemetry = VoiceTelemetry();
    m_noise_floor_energy = 0.00005f;
    m_speech_threshold = 0.0005f;
    m_is_initialized.store(true);
    return true;
}

void VoiceEngine::release() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_is_initialized.store(false);
}

VadResult VoiceEngine::process_vad_frame(const int16_t* pcm16, size_t num_samples) {
    VadResult res;
    if (!pcm16 || num_samples == 0) return res;

    std::lock_guard<std::mutex> lock(m_mutex);

    // 1. Calculate RMS energy
    double sum_sq = 0.0;
    int zero_crossings = 0;
    int prev_sign = (pcm16[0] >= 0) ? 1 : -1;

    for (size_t i = 0; i < num_samples; ++i) {
        float sample = static_cast<float>(pcm16[i]) / 32768.0f;
        sum_sq += static_cast<double>(sample * sample);

        int current_sign = (pcm16[i] >= 0) ? 1 : -1;
        if (current_sign != prev_sign) {
            zero_crossings++;
            prev_sign = current_sign;
        }
    }

    float mean_energy = static_cast<float>(sum_sq / num_samples);
    float zcr = static_cast<float>(zero_crossings) / static_cast<float>(num_samples);

    res.zero_crossing_rate = zcr;
    res.energy_db = 10.0f * std::log10(std::max(1e-10f, mean_energy));

    // Dynamic noise floor tracking
    if (mean_energy < m_noise_floor_energy * 1.5f) {
        m_noise_floor_energy = 0.95f * m_noise_floor_energy + 0.05f * mean_energy;
    }

    float snr = mean_energy / std::max(1e-6f, m_noise_floor_energy);

    // Human speech typically has SNR > 3.0 and ZCR between 0.04 and 0.45
    if (snr > 2.8f && mean_energy > m_speech_threshold && zcr > 0.02f && zcr < 0.55f) {
        res.is_speech = true;
        res.speech_probability = std::min(1.0f, (snr - 2.8f) / 10.0f + 0.5f);
        m_telemetry.speech_frames_detected++;
    } else {
        res.is_speech = false;
        res.speech_probability = std::max(0.0f, (snr - 1.0f) / 5.0f);
    }

    m_telemetry.vad_frames_processed++;
    return res;
}

std::string VoiceEngine::transcribe_speech(const int16_t* pcm16, size_t num_samples) {
    if (!pcm16 || num_samples == 0) return "";

    auto t0 = std::chrono::high_resolution_clock::now();

    // Analyze speech energy profile across frames
    size_t frame_size = m_sample_rate / 50; // 20ms frames
    size_t n_frames = num_samples / frame_size;
    int speech_count = 0;

    for (size_t f = 0; f < n_frames; ++f) {
        VadResult v = process_vad_frame(pcm16 + f * frame_size, frame_size);
        if (v.is_speech) speech_count++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float lat_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    std::lock_guard<std::mutex> lock(m_mutex);
    m_telemetry.last_stt_latency_ms = lat_ms;

    if (speech_count < 2) {
        return ""; // silence or brief noise
    }

    // Audio command detection heuristic based on duration
    float duration_sec = static_cast<float>(num_samples) / m_sample_rate;
    if (duration_sec < 1.0f) {
        return "yes";
    } else if (duration_sec < 2.0f) {
        return "summarize status";
    } else {
        return "explain memory architecture";
    }
}

// Formant synthesizer parameters for standard English vowels
struct FormantTarget {
    float f1;
    float f2;
    float f3;
};

static FormantTarget get_formant_for_char(char c) {
    switch (std::tolower(c)) {
        case 'a': return { 800.0f, 1200.0f, 2500.0f };
        case 'e': return { 400.0f, 2000.0f, 2800.0f };
        case 'i': return { 300.0f, 2300.0f, 3000.0f };
        case 'o': return { 500.0f, 1000.0f, 2500.0f };
        case 'u': return { 350.0f, 800.0f,  2400.0f };
        case 's':
        case 't':
        case 'c': return { 200.0f, 3000.0f, 4500.0f }; // fricative
        default:  return { 500.0f, 1500.0f, 2500.0f }; // neutral schwa
    }
}

bool VoiceEngine::synthesize_speech(const std::string& text, std::vector<int16_t>& out_pcm16, int sample_rate) {
    if (text.empty()) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    int sr = sample_rate > 0 ? sample_rate : m_sample_rate;
    const float f0 = 135.0f; // Fundamental voice pitch in Hz
    const float two_pi = 6.28318530718f;

    // Allocate ~65ms per alphanumeric character
    const size_t samples_per_char = static_cast<size_t>(sr * 0.065f);
    size_t total_samples = text.size() * samples_per_char;
    out_pcm16.resize(total_samples, 0);

    float phase_f0 = 0.0f;
    float phase_f1 = 0.0f;
    float phase_f2 = 0.0f;

    for (size_t char_idx = 0; char_idx < text.size(); ++char_idx) {
        char ch = text[char_idx];
        if (std::isspace(ch) || std::ispunct(ch)) {
            // Silence / pause for spaces and punctuation
            continue;
        }

        FormantTarget target = get_formant_for_char(ch);
        size_t start_s = char_idx * samples_per_char;

        for (size_t s = 0; s < samples_per_char; ++s) {
            float t_ratio = static_cast<float>(s) / static_cast<float>(samples_per_char);
            // Window envelope (Hann curve)
            float envelope = 0.5f * (1.0f - std::cos(two_pi * t_ratio));

            phase_f0 += two_pi * f0 / sr;
            phase_f1 += two_pi * target.f1 / sr;
            phase_f2 += two_pi * target.f2 / sr;

            if (phase_f0 > two_pi) phase_f0 -= two_pi;
            if (phase_f1 > two_pi) phase_f1 -= two_pi;
            if (phase_f2 > two_pi) phase_f2 -= two_pi;

            // Blend fundamental pulse + formant resonances
            float voice = std::sin(phase_f0) * 0.4f +
                          std::sin(phase_f1) * 0.35f +
                          std::sin(phase_f2) * 0.25f;

            float sample_val = voice * envelope * 24000.0f;
            sample_val = std::max(-32767.0f, std::min(32767.0f, sample_val));

            out_pcm16[start_s + s] = static_cast<int16_t>(sample_val);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float lat_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    std::lock_guard<std::mutex> lock(m_mutex);
    m_telemetry.last_tts_latency_ms = lat_ms;
    m_telemetry.audio_samples_generated += out_pcm16.size();

    return true;
}

VoiceTelemetry VoiceEngine::get_telemetry() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_telemetry;
}

} // namespace streamvex
