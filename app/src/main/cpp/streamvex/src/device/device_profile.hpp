#ifndef DEVICE_PROFILE_HPP
#define DEVICE_PROFILE_HPP

#include <string>
#include <vector>

namespace streamvex {

enum class DeviceTier {
    LOW_MEMORY = 0,    // < 4 GB RAM
    MID_RANGE = 1,     // 4 GB - 8 GB RAM
    FLAGSHIP = 2       // > 8 GB RAM
};

struct DeviceHardwareProfile {
    DeviceTier tier = DeviceTier::MID_RANGE;
    std::string tier_name;
    double total_ram_mb = 0.0;
    double available_ram_mb = 0.0;
    int total_cores = 0;
    int big_cores = 0;
    bool has_dotprod = false;
    bool has_fp16 = false;

    // Recommendations
    double max_recommended_model_size_mb = 0.0;
    int max_recommended_context = 2048;
    std::string recommended_quantization;
    size_t recommended_prefetch_buffer_mb = 64;
    int recommended_threads = 4;
};

struct ModelCompatibilityCheck {
    bool is_compatible = false;
    std::string message;
    std::string suggested_model_size;
    int suggested_context = 2048;
};

class DeviceProfileManager {
public:
    static DeviceHardwareProfile get_current_profile();

    static ModelCompatibilityCheck validate_model_compatibility(
        double model_file_size_mb,
        int requested_context_len = 2048
    );

    static std::string to_json_report();
};

} // namespace streamvex

#endif // DEVICE_PROFILE_HPP
