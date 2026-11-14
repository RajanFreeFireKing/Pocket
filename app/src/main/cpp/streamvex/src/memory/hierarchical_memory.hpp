#ifndef HIERARCHICAL_MEMORY_HPP
#define HIERARCHICAL_MEMORY_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

namespace streamvex {

/**
 * Bounded Slab Allocator for reusable scratch and activation buffers.
 * Eliminates heap allocation latency and memory fragmentation during inference.
 */
class MemorySlabPool {
public:
    explicit MemorySlabPool(size_t total_capacity_bytes = 64 * 1024 * 1024, size_t alignment = 4096);
    ~MemorySlabPool();

    MemorySlabPool(const MemorySlabPool&) = delete;
    MemorySlabPool& operator=(const MemorySlabPool&) = delete;

    void* allocate(size_t bytes);
    void reset();

    size_t capacity() const { return m_capacity; }
    size_t used() const { return m_offset.load(); }
    size_t remaining() const { return m_capacity > used() ? m_capacity - used() : 0; }

private:
    uint8_t* m_memory = nullptr;
    size_t m_capacity = 0;
    size_t m_alignment = 4096;
    std::atomic<size_t> m_offset{0};
};

/**
 * Memory-Mapped Model Access Wrapper.
 * Provides zero-copy direct access to weights with POSIX page eviction advice.
 */
class MmapModelRegion {
public:
    MmapModelRegion();
    ~MmapModelRegion();

    bool open_and_map(const std::string& path, bool sequential_advice = true);
    void close();

    const uint8_t* data() const { return m_addr; }
    size_t size() const { return m_size; }
    bool is_mapped() const { return m_addr != nullptr; }

    void advise_willneed(size_t offset, size_t length);
    void advise_dontneed(size_t offset, size_t length);

private:
    int m_fd = -1;
    uint8_t* m_addr = nullptr;
    size_t m_size = 0;
};

/**
 * Multi-buffer configuration for layer streaming benchmark and execution.
 */
enum class BufferMode {
    DOUBLE_BUFFER = 2,
    TRIPLE_BUFFER = 3
};

struct BufferBenchmarkResult {
    BufferMode mode;
    double throughput_mb_s = 0.0;
    double avg_prefetch_latency_ms = 0.0;
    double peak_ram_mb = 0.0;
    double duration_sec = 0.0;
};

/**
 * Hierarchical Memory Manager orchestrating mmap regions,
 * activation slab pools, streaming prefetch buffers, and memory pressure recovery.
 */
class HierarchicalMemoryManager {
public:
    static HierarchicalMemoryManager& instance();

    HierarchicalMemoryManager();
    ~HierarchicalMemoryManager() = default;

    HierarchicalMemoryManager(const HierarchicalMemoryManager&) = delete;
    HierarchicalMemoryManager& operator=(const HierarchicalMemoryManager&) = delete;

    /**
     * Initializes memory pools and buffers.
     */
    bool initialize(size_t scratch_pool_mb = 64);

    /**
     * Acquires temporary scratchpad memory for activations without malloc.
     */
    void* acquire_scratch(size_t bytes);

    /**
     * Resets scratch memory offset at the start of a new token step.
     */
    void reset_scratch();

    /**
     * Maps model weights using zero-copy mmap.
     */
    bool map_model_file(const std::string& path);
    void unmap_model();

    const MmapModelRegion& mapped_model() const { return m_mmap_region; }

    /**
     * Benchmarks Double-Buffering vs Triple-Buffering on the current mobile hardware.
     */
    BufferBenchmarkResult benchmark_buffering(
        const std::string& file_path,
        BufferMode mode,
        size_t chunk_bytes,
        int total_layers
    );

    /**
     * Memory pressure recovery: evicts non-critical buffers, releases slab capacity,
     * and advises OS kernel to discard clean mmap pages.
     */
    void handle_memory_pressure_recovery();

    /**
     * Queries current resident memory (RSS) from /proc/self/statm.
     */
    static double get_process_rss_mb();

private:
    std::mutex m_mutex;
    std::unique_ptr<MemorySlabPool> m_scratch_pool;
    MmapModelRegion m_mmap_region;
    bool m_initialized = false;
};

} // namespace streamvex

#endif // HIERARCHICAL_MEMORY_HPP
