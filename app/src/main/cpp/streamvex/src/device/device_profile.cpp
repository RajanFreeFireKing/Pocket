#include "device_profile.hpp"
#include "../resource/resource_intelligence.hpp"
#include "../scheduler/compute_scheduler.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>

namespace streamvex {

DeviceHardwareProfile DeviceProfileManager::get_current_profile() {
    DeviceHardwareProfile p;
    auto& rim = ResourceIntelligence::instance();
    rim.update_metrics();

    auto mem = rim.get_memory_info();
    auto cpu = rim.get_cpu_info();
    auto& sched = ComputeScheduler::instance();
    auto feat = sched.cpu_features();

    p.total_ram_mb = mem.total_ram_mb;
    p.available_ram_mb = mem.available_ram_mb;
    p.total_cores = cpu.total_cores;
    p.big_cores = cpu.big_cores;
    p.has_dotprod = feat.has_dotprod;
    p.has_fp16 = feat.has_fp16;

    if (p.total_ram_mb < 3800.0) {
        p.tier = DeviceTier::LOW_MEMORY;
        p.tier_name = "Low-Memory Mobile (< 4GB)";
        p.max_recommended_model_size_mb = 800.0; // up to ~1B parameter Q4
        p.max_recommended_context = 1024;
        p.recommended_quantization = "Q4_0";
        p.recommended_prefetch_buffer_mb = 32;
        p.recommended_threads = std::max(1, p.big_cores);
    } else if (p.total_ram_mb < 7800.0) {
        p.tier = DeviceTier::MID_RANGE;
        p.tier_name = "Mid-Range Mobile (4GB - 8GB)";
        p.max_recommended_model_size_mb = 2500.0; // up to ~3B parameter Q4_K_M
        p.max_recommended_context = 2048;
        p.recommended_quantization = "Q4_K_M";
        p.recommended_prefetch_buffer_mb = 64;
        p.recommended_threads = std::max(2, std::min(4, p.big_cores + 1));
    } else {
        p.tier = DeviceTier::FLAGSHIP;
        p.tier_name = "Flagship Mobile (> 8GB)";
        p.max_recommended_model_size_mb = 5200.0; // up to 7B-8B Q4_K_M
        p.max_recommended_context = 4096;
        p.recommended_quantization = "Q4_K_M / Q8_0";
        p.recommended_prefetch_buffer_mb = 128;
        p.recommended_threads = std::max(4, p.big_cores + 2);
    }

    return p;
}

ModelCompatibilityCheck DeviceProfileManager::validate_model_compatibility(
    double model_file_size_mb,
    int requested_context_len
) {
    ModelCompatibilityCheck c;
    auto p = get_current_profile();

    double estimated_kv_mb = (requested_context_len / 1024.0) * 128.0;
    double total_need = model_file_size_mb + estimated_kv_mb;
    double safe_ceiling = p.available_ram_mb - 350.0; // 350MB OS margin

    if (total_need <= safe_ceiling && model_file_size_mb <= p.max_recommended_model_size_mb * 1.25) {
        c.is_compatible = true;
        c.message = "Model is fully compatible with device profile (" + p.tier_name + ").";
        c.suggested_model_size = std::to_string(static_cast<int>(model_file_size_mb)) + " MB";
        c.suggested_context = requested_context_len;
        return c;
    }

    c.is_compatible = false;
    std::ostringstream ss;
    ss << "Requested model (" << static_cast<int>(model_file_size_mb) << " MB + "
       << static_cast<int>(estimated_kv_mb) << " MB KV cache) exceeds safe allocation margin ("
       << static_cast<int>(safe_ceiling) << " MB) for " << p.tier_name << ". "
       << "Loading this model risks Android low-memory-killer (OOM) termination of foreground apps.";
    c.message = ss.str();

    if (p.tier == DeviceTier::LOW_MEMORY) {
        c.suggested_model_size = "SmolLM2-135M (~100 MB) or Qwen2.5-0.5B (~350 MB Q4)";
        c.suggested_context = 1024;
    } else if (p.tier == DeviceTier::MID_RANGE) {
        c.suggested_model_size = "Llama-3.2-1B (~800 MB Q4) or Qwen2.5-1.5B (~1.2 GB Q4)";
        c.suggested_context = 2048;
    } else {
        c.suggested_model_size = "Llama-3.2-3B (~2.1 GB Q4_K_M)";
        c.suggested_context = 4096;
    }

    return c;
}

std::string DeviceProfileManager::to_json_report() {
    auto p = get_current_profile();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "{\n"
       << "  \"tier\": " << static_cast<int>(p.tier) << ",\n"
       << "  \"tier_name\": \"" << p.tier_name << "\",\n"
       << "  \"total_ram_mb\": " << p.total_ram_mb << ",\n"
       << "  \"available_ram_mb\": " << p.available_ram_mb << ",\n"
       << "  \"total_cores\": " << p.total_cores << ",\n"
       << "  \"big_cores\": " << p.big_cores << ",\n"
       << "  \"has_dotprod\": " << (p.has_dotprod ? "true" : "false") << ",\n"
       << "  \"has_fp16\": " << (p.has_fp16 ? "true" : "false") << ",\n"
       << "  \"max_recommended_model_size_mb\": " << p.max_recommended_model_size_mb << ",\n"
       << "  \"max_recommended_context\": " << p.max_recommended_context << ",\n"
       << "  \"recommended_quantization\": \"" << p.recommended_quantization << "\",\n"
       << "  \"recommended_prefetch_buffer_mb\": " << p.recommended_prefetch_buffer_mb << ",\n"
       << "  \"recommended_threads\": " << p.recommended_threads << "\n"
       << "}";
    return ss.str();
}

} // namespace streamvex
