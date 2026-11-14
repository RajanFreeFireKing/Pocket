#ifndef RESOURCE_INTELLIGENCE_HPP
#define RESOURCE_INTELLIGENCE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>

namespace streamvex {

enum class ThermalStatus {
    NORMAL = 0,
    WARM = 1,
    HOT = 2,
    CRITICAL = 3,
    UNKNOWN = 4
};

enum class MemoryPressure {
    NONE = 0,
    MODERATE = 1,
    HIGH = 2,
    CRITICAL = 3
};

enum class BatteryStatus {
    UNKNOWN = 0,
    CHARGING = 1,
    DISCHARGING = 2,
    FULL = 3
};

struct SystemMemoryInfo {
    double total_ram_mb = 0.0;
    double available_ram_mb = 0.0;
    double free_ram_mb = 0.0;
    double buffers_ram_mb = 0.0;
    double cached_ram_mb = 0.0;
    double swap_total_mb = 0.0;
    double swap_free_mb = 0.0;
    double process_rss_mb = 0.0;
    double process_peak_rss_mb = 0.0;
    double process_virt_mb = 0.0;
    MemoryPressure pressure = MemoryPressure::NONE;
};

struct CpuCoreInfo {
    int core_id = 0;
    int cur_freq_khz = 0;
    int max_freq_khz = 0;
    bool is_online = true;
    bool is_big_core = false;
};

struct CpuSystemInfo {
    int total_cores = 0;
    int online_cores = 0;
    int big_cores = 0;
    int little_cores = 0;
    float overall_utilization_pct = 0.0f;
    float process_utilization_pct = 0.0f;
    std::vector<CpuCoreInfo> cores;
};

struct DeviceThermalInfo {
    float temp_celsius = 0.0f;
    ThermalStatus status = ThermalStatus::NORMAL;
    std::string zone_type;
};

struct BatteryInfo {
    int capacity_pct = 100;
    BatteryStatus status = BatteryStatus::UNKNOWN;
    float temp_celsius = 0.0f;
};

struct AdaptiveBudget {
    double max_safe_model_budget_mb = 0.0;
    double max_safe_kv_cache_budget_mb = 0.0;
    size_t layer_prefetch_buffer_size_bytes = 0;
    int recommended_threads = 4;
    double voice_model_memory_budget_mb = 0.0;
    int background_priority = 3; // P3 default
    bool should_throttle = false;
    float throttle_factor = 1.0f; // 1.0 = full speed, <1.0 = slower pacing
};

/**
 * Central Resource Intelligence Manager for StreamVex.
 * Continuously samples device metrics and computes safe, adaptive resource budgets.
 */
class ResourceIntelligence {
public:
    static ResourceIntelligence& instance();

    ResourceIntelligence();
    ~ResourceIntelligence() = default;

    // Prevent copies
    ResourceIntelligence(const ResourceIntelligence&) = delete;
    ResourceIntelligence& operator=(const ResourceIntelligence&) = delete;

    /**
     * Refreshes all device metrics synchronously.
     */
    void update_metrics();

    // Query individual metrics
    SystemMemoryInfo get_memory_info();
    CpuSystemInfo get_cpu_info();
    DeviceThermalInfo get_thermal_info();
    BatteryInfo get_battery_info();

    /**
     * Calculates the safe resource budget based on live hardware state and model size.
     * 
     * @param requested_model_size_mb Optional size of the candidate model in MB.
     * @param is_foreground Whether StreamVex is currently in the interactive foreground.
     * @return Computed AdaptiveBudget.
     */
    AdaptiveBudget compute_budget(double requested_model_size_mb = 0.0, bool is_foreground = true);

    /**
     * Delivers Android OS memory trim notification to the engine.
     * 
     * @param trim_level Level constant from Android ComponentCallbacks2.
     */
    void on_trim_memory(int trim_level);

    /**
     * Sets a safety margin below available RAM (default 350 MB on mobile).
     */
    void set_safety_margin_mb(double margin_mb) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_safety_margin_mb = margin_mb;
    }

    double safety_margin_mb() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_safety_margin_mb;
    }

    /**
     * Checks if a candidate model can safely fit in memory.
     * 
     * @param model_file_size_mb File size of the candidate GGUF model in MB.
     * @param required_kv_mb Estimated KV cache in MB.
     * @param out_reason Populated with rejection rationale if unsafe.
     * @return true if safe to load, false otherwise.
     */
    bool can_safely_fit(double model_file_size_mb, double required_kv_mb, std::string& out_reason);

    /**
     * Measures storage sequential read throughput in MB/s.
     */
    double measure_storage_throughput_mb_s(const std::string& test_file_path);

    /**
     * Exports all telemetry and resource intelligence as a JSON string.
     */
    std::string to_json_telemetry() const;

private:
    void sample_memory();
    void sample_cpu();
    void sample_thermals();
    void sample_battery();

    mutable std::mutex m_mutex;
    SystemMemoryInfo m_mem_info;
    CpuSystemInfo m_cpu_info;
    DeviceThermalInfo m_thermal_info;
    BatteryInfo m_battery_info;
    double m_safety_margin_mb = 350.0; // 350MB reserve for OS & foreground apps

    // CPU load calculation tracking
    uint64_t m_last_total_cpu_time = 0;
    uint64_t m_last_idle_cpu_time = 0;
    uint64_t m_last_process_cpu_time = 0;
    std::chrono::steady_clock::time_point m_last_cpu_sample_time;
};

} // namespace streamvex

#endif // RESOURCE_INTELLIGENCE_HPP
