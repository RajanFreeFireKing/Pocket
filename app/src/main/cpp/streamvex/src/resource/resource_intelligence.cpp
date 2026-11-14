#include "resource_intelligence.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>

namespace streamvex {

ResourceIntelligence& ResourceIntelligence::instance() {
    static ResourceIntelligence s_instance;
    return s_instance;
}

ResourceIntelligence::ResourceIntelligence() {
    m_last_cpu_sample_time = std::chrono::steady_clock::now();
    update_metrics();
}

void ResourceIntelligence::update_metrics() {
    std::lock_guard<std::mutex> lock(m_mutex);
    sample_memory();
    sample_cpu();
    sample_thermals();
    sample_battery();
}

void ResourceIntelligence::sample_memory() {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            std::istringstream iss(line);
            std::string key;
            long value = 0;
            std::string unit;
            if (iss >> key >> value) {
                double mb = static_cast<double>(value) / 1024.0;
                if (key == "MemTotal:") m_mem_info.total_ram_mb = mb;
                else if (key == "MemAvailable:") m_mem_info.available_ram_mb = mb;
                else if (key == "MemFree:") m_mem_info.free_ram_mb = mb;
                else if (key == "Buffers:") m_mem_info.buffers_ram_mb = mb;
                else if (key == "Cached:") m_mem_info.cached_ram_mb = mb;
                else if (key == "SwapTotal:") m_mem_info.swap_total_mb = mb;
                else if (key == "SwapFree:") m_mem_info.swap_free_mb = mb;
            }
        }
    }

    // Process memory from statm
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long total_pages = 0;
        long resident_pages = 0;
        if (statm >> total_pages >> resident_pages) {
            long page_size_bytes = sysconf(_SC_PAGESIZE);
            if (page_size_bytes > 0) {
                m_mem_info.process_rss_mb = static_cast<double>(resident_pages * page_size_bytes) / (1024.0 * 1024.0);
                m_mem_info.process_virt_mb = static_cast<double>(total_pages * page_size_bytes) / (1024.0 * 1024.0);
                if (m_mem_info.process_rss_mb > m_mem_info.process_peak_rss_mb) {
                    m_mem_info.process_peak_rss_mb = m_mem_info.process_rss_mb;
                }
            }
        }
    }

    // Determine memory pressure level
    if (m_mem_info.total_ram_mb > 0.0) {
        double avail_ratio = m_mem_info.available_ram_mb / m_mem_info.total_ram_mb;
        if (avail_ratio < 0.10 || m_mem_info.available_ram_mb < 200.0) {
            m_mem_info.pressure = MemoryPressure::CRITICAL;
        } else if (avail_ratio < 0.20 || m_mem_info.available_ram_mb < 400.0) {
            m_mem_info.pressure = MemoryPressure::HIGH;
        } else if (avail_ratio < 0.30 || m_mem_info.available_ram_mb < 800.0) {
            m_mem_info.pressure = MemoryPressure::MODERATE;
        } else {
            m_mem_info.pressure = MemoryPressure::NONE;
        }
    }
#endif
}

void ResourceIntelligence::sample_cpu() {
#if defined(__linux__) || defined(__ANDROID__)
    // Read overall CPU ticks
    std::ifstream stat("/proc/stat");
    if (stat.is_open()) {
        std::string line;
        if (std::getline(stat, line) && line.compare(0, 3, "cpu") == 0) {
            std::istringstream iss(line);
            std::string cpu_label;
            uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
            iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

            uint64_t total_time = user + nice + system + idle + iowait + irq + softirq + steal;
            uint64_t idle_time = idle + iowait;

            if (m_last_total_cpu_time > 0 && total_time > m_last_total_cpu_time) {
                uint64_t d_total = total_time - m_last_total_cpu_time;
                uint64_t d_idle = idle_time - m_last_idle_cpu_time;
                m_cpu_info.overall_utilization_pct = 100.0f * (1.0f - static_cast<float>(d_idle) / static_cast<float>(d_total));
                m_cpu_info.overall_utilization_pct = std::max(0.0f, std::min(100.0f, m_cpu_info.overall_utilization_pct));
            }
            m_last_total_cpu_time = total_time;
            m_last_idle_cpu_time = idle_time;
        }
    }

    // Process CPU ticks
    std::ifstream pstat("/proc/self/stat");
    if (pstat.is_open()) {
        std::string token;
        // Skip first 13 fields to reach utime and stime
        for (int i = 0; i < 13; ++i) pstat >> token;
        uint64_t utime = 0, stime = 0;
        if (pstat >> utime >> stime) {
            uint64_t p_total = utime + stime;
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - m_last_cpu_sample_time).count();
            if (elapsed_sec > 0.05 && m_last_process_cpu_time > 0) {
                long clk_tck = sysconf(_SC_CLK_TCK);
                if (clk_tck > 0) {
                    double delta_sec = static_cast<double>(p_total - m_last_process_cpu_time) / clk_tck;
                    m_cpu_info.process_utilization_pct = static_cast<float>((delta_sec / elapsed_sec) * 100.0);
                }
            }
            m_last_process_cpu_time = p_total;
            m_last_cpu_sample_time = now;
        }
    }

    // Cores and topology
    long n_configured = sysconf(_SC_NPROCESSORS_CONF);
    long n_online = sysconf(_SC_NPROCESSORS_ONLN);
    m_cpu_info.total_cores = n_configured > 0 ? static_cast<int>(n_configured) : 4;
    m_cpu_info.online_cores = n_online > 0 ? static_cast<int>(n_online) : m_cpu_info.total_cores;

    m_cpu_info.cores.clear();
    int big_count = 0;
    int max_overall_freq = 0;

    for (int i = 0; i < m_cpu_info.total_cores; ++i) {
        CpuCoreInfo core;
        core.core_id = i;
        std::string freq_path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/scaling_cur_freq";
        std::string max_freq_path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq";

        std::ifstream ff(freq_path);
        if (ff >> core.cur_freq_khz) {}

        std::ifstream mf(max_freq_path);
        if (mf >> core.max_freq_khz) {
            if (core.max_freq_khz > max_overall_freq) {
                max_overall_freq = core.max_freq_khz;
            }
        }
        core.is_online = (i < m_cpu_info.online_cores);
        m_cpu_info.cores.push_back(core);
    }

    // Classify big vs little cores based on max frequency
    for (auto& c : m_cpu_info.cores) {
        if (max_overall_freq > 0 && c.max_freq_khz >= static_cast<int>(max_overall_freq * 0.85)) {
            c.is_big_core = true;
            big_count++;
        } else {
            c.is_big_core = false;
        }
    }
    m_cpu_info.big_cores = big_count > 0 ? big_count : m_cpu_info.total_cores / 2;
    m_cpu_info.little_cores = m_cpu_info.total_cores - m_cpu_info.big_cores;
#endif
}

void ResourceIntelligence::sample_thermals() {
#if defined(__linux__) || defined(__ANDROID__)
    float highest_temp = 0.0f;
    std::string highest_zone;

    // Scan /sys/class/thermal/thermal_zone*
    for (int z = 0; z < 32; ++z) {
        std::string tpath = "/sys/class/thermal/thermal_zone" + std::to_string(z) + "/temp";
        std::string typath = "/sys/class/thermal/thermal_zone" + std::to_string(z) + "/type";

        std::ifstream tf(tpath);
        if (!tf.is_open()) break;

        long raw_temp = 0;
        if (tf >> raw_temp) {
            float temp_c = raw_temp > 1000 ? static_cast<float>(raw_temp) / 1000.0f : static_cast<float>(raw_temp);
            if (temp_c > highest_temp && temp_c < 125.0f) { // filter bad sensor spikes >125C
                highest_temp = temp_c;
                std::ifstream tyf(typath);
                std::string ztype;
                if (tyf >> ztype) highest_zone = ztype;
            }
        }
    }

    m_thermal_info.temp_celsius = highest_temp > 0.0f ? highest_temp : 35.0f;
    m_thermal_info.zone_type = highest_zone.empty() ? "cpu-thermal" : highest_zone;

    if (m_thermal_info.temp_celsius >= 80.0f) {
        m_thermal_info.status = ThermalStatus::CRITICAL;
    } else if (m_thermal_info.temp_celsius >= 68.0f) {
        m_thermal_info.status = ThermalStatus::HOT;
    } else if (m_thermal_info.temp_celsius >= 52.0f) {
        m_thermal_info.status = ThermalStatus::WARM;
    } else {
        m_thermal_info.status = ThermalStatus::NORMAL;
    }
#else
    m_thermal_info.temp_celsius = 35.0f;
    m_thermal_info.status = ThermalStatus::NORMAL;
#endif
}

void ResourceIntelligence::sample_battery() {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream cap_f("/sys/class/power_supply/battery/capacity");
    if (cap_f.is_open()) {
        cap_f >> m_battery_info.capacity_pct;
    }

    std::ifstream stat_f("/sys/class/power_supply/battery/status");
    if (stat_f.is_open()) {
        std::string bstat;
        stat_f >> bstat;
        if (bstat == "Charging") m_battery_info.status = BatteryStatus::CHARGING;
        else if (bstat == "Discharging") m_battery_info.status = BatteryStatus::DISCHARGING;
        else if (bstat == "Full") m_battery_info.status = BatteryStatus::FULL;
        else m_battery_info.status = BatteryStatus::UNKNOWN;
    }

    std::ifstream temp_f("/sys/class/power_supply/battery/temp");
    if (temp_f.is_open()) {
        int btemp = 0;
        if (temp_f >> btemp) {
            m_battery_info.temp_celsius = static_cast<float>(btemp) / 10.0f;
        }
    }
#endif
}

SystemMemoryInfo ResourceIntelligence::get_memory_info() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_mem_info;
}

CpuSystemInfo ResourceIntelligence::get_cpu_info() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_cpu_info;
}

DeviceThermalInfo ResourceIntelligence::get_thermal_info() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_thermal_info;
}

BatteryInfo ResourceIntelligence::get_battery_info() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_battery_info;
}

AdaptiveBudget ResourceIntelligence::compute_budget(double requested_model_size_mb, bool is_foreground) {
    std::lock_guard<std::mutex> lock(m_mutex);

    AdaptiveBudget b;
    double avail = m_mem_info.available_ram_mb > 0.0 ? m_mem_info.available_ram_mb : 1024.0;
    double safety = m_safety_margin_mb;

    // Available RAM after reserving safety margin
    double safe_usable = std::max(128.0, avail - safety);

    // Max safe model memory budget
    b.max_safe_model_budget_mb = safe_usable * 0.75;

    // Safe KV cache budget
    b.max_safe_kv_cache_budget_mb = std::max(32.0, safe_usable * 0.15);

    // Voice model memory reservation (VAD + STT/TTS scratch)
    b.voice_model_memory_budget_mb = std::min(80.0, safe_usable * 0.10);

    // Prefetch layer buffer size: 32MB on low ram, 64MB default, 128MB on high ram
    if (m_mem_info.total_ram_mb > 7000.0) {
        b.layer_prefetch_buffer_size_bytes = 128 * 1024 * 1024;
    } else if (m_mem_info.total_ram_mb < 3500.0) {
        b.layer_prefetch_buffer_size_bytes = 32 * 1024 * 1024;
    } else {
        b.layer_prefetch_buffer_size_bytes = 64 * 1024 * 1024;
    }

    // Thread scheduling recommendations
    int total_online = m_cpu_info.online_cores > 0 ? m_cpu_info.online_cores : 4;
    int big_cores = m_cpu_info.big_cores > 0 ? m_cpu_info.big_cores : (total_online / 2);

    if (is_foreground) {
        // Foreground interactive: utilize big cores + up to 2 little cores, capped to total - 1
        b.recommended_threads = std::max(2, std::min(total_online - 1, big_cores + 1));
        b.background_priority = 1;
        b.should_throttle = false;
        b.throttle_factor = 1.0f;
    } else {
        // Background AI workload: conservative, don't hog phone
        b.recommended_threads = std::max(1, std::min(2, total_online / 2));
        b.background_priority = 3;
        b.should_throttle = true;
        b.throttle_factor = 0.65f;
    }

    // Thermal throttling adjustment
    if (m_thermal_info.status == ThermalStatus::CRITICAL) {
        b.recommended_threads = 1;
        b.should_throttle = true;
        b.throttle_factor = 0.35f;
    } else if (m_thermal_info.status == ThermalStatus::HOT) {
        b.recommended_threads = std::max(1, b.recommended_threads / 2);
        b.should_throttle = true;
        b.throttle_factor = std::min(b.throttle_factor, 0.50f);
    }

    // Battery conservation adjustment
    if (m_battery_info.status == BatteryStatus::DISCHARGING && m_battery_info.capacity_pct < 15) {
        b.recommended_threads = std::max(1, b.recommended_threads - 1);
        b.should_throttle = true;
        b.throttle_factor = std::min(b.throttle_factor, 0.60f);
    }

    // Memory pressure adjustment
    if (m_mem_info.pressure == MemoryPressure::CRITICAL) {
        b.max_safe_kv_cache_budget_mb = std::max(16.0, b.max_safe_kv_cache_budget_mb * 0.4);
        b.should_throttle = true;
        b.throttle_factor = 0.30f;
    }

    (void)requested_model_size_mb;
    return b;
}

bool ResourceIntelligence::can_safely_fit(double model_file_size_mb, double required_kv_mb, std::string& out_reason) {
    update_metrics();
    std::lock_guard<std::mutex> lock(m_mutex);

    double avail = m_mem_info.available_ram_mb;
    double safety = m_safety_margin_mb;
    double total_required = model_file_size_mb + required_kv_mb;

    if (avail <= safety) {
        out_reason = "Device memory pressure too high. Available RAM (" + 
                     std::to_string(static_cast<int>(avail)) + " MB) is below safe OS reserve (" +
                     std::to_string(static_cast<int>(safety)) + " MB).";
        return false;
    }

    double max_allowable = avail - safety;
    if (total_required > max_allowable) {
        out_reason = "Requested model requires ~" + std::to_string(static_cast<int>(total_required)) +
                     " MB (weights + KV-cache), but device can only safely allocate " +
                     std::to_string(static_cast<int>(max_allowable)) + " MB while maintaining system stability.";
        return false;
    }

    out_reason = "Safe";
    return true;
}

void ResourceIntelligence::on_trim_memory(int trim_level) {
    std::lock_guard<std::mutex> lock(m_mutex);
    // Standard Android ComponentCallbacks2 trim levels
    // TRIM_MEMORY_RUNNING_CRITICAL = 15, TRIM_MEMORY_COMPLETE = 80
    if (trim_level >= 15) {
        m_mem_info.pressure = MemoryPressure::CRITICAL;
    } else if (trim_level >= 10) {
        m_mem_info.pressure = MemoryPressure::HIGH;
    } else if (trim_level >= 5) {
        m_mem_info.pressure = MemoryPressure::MODERATE;
    }
}

double ResourceIntelligence::measure_storage_throughput_mb_s(const std::string& test_file_path) {
    int fd = open(test_file_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0.0;

    const size_t test_chunk = 4 * 1024 * 1024; // 4MB test read
    std::vector<uint8_t> buf(test_chunk);

    auto t0 = std::chrono::high_resolution_clock::now();
    ssize_t bytes_read = pread(fd, buf.data(), test_chunk, 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    close(fd);

    if (bytes_read <= 0) return 0.0;

    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed_sec <= 0.00001) return 2000.0; // Cached read cap

    double mb = static_cast<double>(bytes_read) / (1024.0 * 1024.0);
    return mb / elapsed_sec;
}

std::string ResourceIntelligence::to_json_telemetry() const {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"memory\": {\n"
       << "    \"total_mb\": " << m_mem_info.total_ram_mb << ",\n"
       << "    \"available_mb\": " << m_mem_info.available_ram_mb << ",\n"
       << "    \"process_rss_mb\": " << m_mem_info.process_rss_mb << ",\n"
       << "    \"process_peak_rss_mb\": " << m_mem_info.process_peak_rss_mb << ",\n"
       << "    \"pressure\": " << static_cast<int>(m_mem_info.pressure) << "\n"
       << "  },\n"
       << "  \"cpu\": {\n"
       << "    \"total_cores\": " << m_cpu_info.total_cores << ",\n"
       << "    \"online_cores\": " << m_cpu_info.online_cores << ",\n"
       << "    \"big_cores\": " << m_cpu_info.big_cores << ",\n"
       << "    \"little_cores\": " << m_cpu_info.little_cores << ",\n"
       << "    \"overall_utilization_pct\": " << m_cpu_info.overall_utilization_pct << ",\n"
       << "    \"process_utilization_pct\": " << m_cpu_info.process_utilization_pct << "\n"
       << "  },\n"
       << "  \"thermal\": {\n"
       << "    \"celsius\": " << m_thermal_info.temp_celsius << ",\n"
       << "    \"status\": " << static_cast<int>(m_thermal_info.status) << ",\n"
       << "    \"zone\": \"" << m_thermal_info.zone_type << "\"\n"
       << "  },\n"
       << "  \"battery\": {\n"
       << "    \"capacity_pct\": " << m_battery_info.capacity_pct << ",\n"
       << "    \"status\": " << static_cast<int>(m_battery_info.status) << ",\n"
       << "    \"temp_celsius\": " << m_battery_info.temp_celsius << "\n"
       << "  }\n"
       << "}";
    return ss.str();
}

} // namespace streamvex
