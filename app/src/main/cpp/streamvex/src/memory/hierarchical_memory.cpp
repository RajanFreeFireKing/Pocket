#include "hierarchical_memory.hpp"

#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace streamvex {

// ============================================================================
// MemorySlabPool Implementation
// ============================================================================

MemorySlabPool::MemorySlabPool(size_t total_capacity_bytes, size_t alignment)
    : m_capacity(total_capacity_bytes), m_alignment(alignment) {
    if (m_capacity < 4096) m_capacity = 4096;
    if (m_alignment < 16) m_alignment = 16;

    int res = posix_memalign(reinterpret_cast<void**>(&m_memory), m_alignment, m_capacity);
    if (res != 0 || !m_memory) {
        m_memory = static_cast<uint8_t*>(malloc(m_capacity));
    }
}

MemorySlabPool::~MemorySlabPool() {
    if (m_memory) {
        free(m_memory);
        m_memory = nullptr;
    }
}

void* MemorySlabPool::allocate(size_t bytes) {
    if (!m_memory || bytes == 0) return nullptr;

    // Align bytes to m_alignment
    size_t aligned_bytes = (bytes + m_alignment - 1) & ~(m_alignment - 1);
    size_t old_offset = m_offset.fetch_add(aligned_bytes);

    if (old_offset + aligned_bytes > m_capacity) {
        // Pool exhausted
        return nullptr;
    }

    return m_memory + old_offset;
}

void MemorySlabPool::reset() {
    m_offset.store(0);
}

// ============================================================================
// MmapModelRegion Implementation
// ============================================================================

MmapModelRegion::MmapModelRegion() = default;

MmapModelRegion::~MmapModelRegion() {
    close();
}

bool MmapModelRegion::open_and_map(const std::string& path, bool sequential_advice) {
    close();

    m_fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (m_fd < 0) return false;

    struct stat st;
    if (fstat(m_fd, &st) != 0 || st.st_size <= 0) {
        ::close(m_fd);
        m_fd = -1;
        return false;
    }

    m_size = static_cast<size_t>(st.st_size);
    void* addr = mmap(nullptr, m_size, PROT_READ, MAP_SHARED, m_fd, 0);
    if (addr == MAP_FAILED) {
        ::close(m_fd);
        m_fd = -1;
        m_size = 0;
        return false;
    }

    m_addr = static_cast<uint8_t*>(addr);

    if (sequential_advice) {
#if defined(MADV_SEQUENTIAL)
        madvise(m_addr, m_size, MADV_SEQUENTIAL);
#endif
    }

    return true;
}

void MmapModelRegion::close() {
    if (m_addr && m_size > 0) {
        munmap(m_addr, m_size);
        m_addr = nullptr;
        m_size = 0;
    }
    if (m_fd >= 0) {
        ::close(m_fd);
        m_fd = -1;
    }
}

void MmapModelRegion::advise_willneed(size_t offset, size_t length) {
    if (!m_addr || offset + length > m_size) return;
#if defined(MADV_WILLNEED)
    size_t page_mask = ~(sysconf(_SC_PAGESIZE) - 1);
    size_t aligned_offset = offset & page_mask;
    size_t aligned_len = (offset + length - aligned_offset + sysconf(_SC_PAGESIZE) - 1) & page_mask;
    madvise(m_addr + aligned_offset, aligned_len, MADV_WILLNEED);
#endif
}

void MmapModelRegion::advise_dontneed(size_t offset, size_t length) {
    if (!m_addr || offset + length > m_size) return;
#if defined(MADV_DONTNEED)
    size_t page_mask = ~(sysconf(_SC_PAGESIZE) - 1);
    size_t aligned_offset = offset & page_mask;
    size_t aligned_len = (offset + length - aligned_offset + sysconf(_SC_PAGESIZE) - 1) & page_mask;
    madvise(m_addr + aligned_offset, aligned_len, MADV_DONTNEED);
#endif
}

// ============================================================================
// HierarchicalMemoryManager Implementation
// ============================================================================

HierarchicalMemoryManager& HierarchicalMemoryManager::instance() {
    static HierarchicalMemoryManager s_instance;
    return s_instance;
}

HierarchicalMemoryManager::HierarchicalMemoryManager() {
    initialize(64);
}

bool HierarchicalMemoryManager::initialize(size_t scratch_pool_mb) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_scratch_pool = std::make_unique<MemorySlabPool>(scratch_pool_mb * 1024 * 1024);
    m_initialized = true;
    return true;
}

void* HierarchicalMemoryManager::acquire_scratch(size_t bytes) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_scratch_pool) return nullptr;
    return m_scratch_pool->allocate(bytes);
}

void HierarchicalMemoryManager::reset_scratch() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_scratch_pool) {
        m_scratch_pool->reset();
    }
}

bool HierarchicalMemoryManager::map_model_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_mmap_region.open_and_map(path, true);
}

void HierarchicalMemoryManager::unmap_model() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_mmap_region.close();
}

double HierarchicalMemoryManager::get_process_rss_mb() {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long total_pages = 0;
        long resident_pages = 0;
        if (statm >> total_pages >> resident_pages) {
            long page_size_bytes = sysconf(_SC_PAGESIZE);
            if (page_size_bytes > 0) {
                return static_cast<double>(resident_pages * page_size_bytes) / (1024.0 * 1024.0);
            }
        }
    }
#endif
    return 0.0;
}

void HierarchicalMemoryManager::handle_memory_pressure_recovery() {
    std::lock_guard<std::mutex> lock(m_mutex);
    // Reset scratch pool
    if (m_scratch_pool) {
        m_scratch_pool->reset();
    }

    // Advise OS that mapped pages can be reclaimed immediately
    if (m_mmap_region.is_mapped()) {
        m_mmap_region.advise_dontneed(0, m_mmap_region.size());
    }
}

BufferBenchmarkResult HierarchicalMemoryManager::benchmark_buffering(
    const std::string& file_path,
    BufferMode mode,
    size_t chunk_bytes,
    int total_layers
) {
    BufferBenchmarkResult result;
    result.mode = mode;

    int n_buffers = (mode == BufferMode::TRIPLE_BUFFER) ? 3 : 2;
    std::vector<uint8_t*> buffers(n_buffers, nullptr);

    for (int i = 0; i < n_buffers; ++i) {
        int r = posix_memalign(reinterpret_cast<void**>(&buffers[i]), 4096, chunk_bytes);
        if (r != 0 || !buffers[i]) {
            buffers[i] = static_cast<uint8_t*>(malloc(chunk_bytes));
        }
    }

    int fd = open(file_path.c_str(), O_RDONLY | O_CLOEXEC);
    std::vector<double> latencies;
    latencies.reserve(total_layers);

    double base_ram = get_process_rss_mb();
    double peak_ram = base_ram;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int layer = 0; layer < total_layers; ++layer) {
        int buf_idx = layer % n_buffers;
        uint8_t* target = buffers[buf_idx];

        auto l_start = std::chrono::high_resolution_clock::now();
        if (fd >= 0) {
            off_t offset = static_cast<off_t>(layer) * chunk_bytes;
            ssize_t n = pread(fd, target, chunk_bytes, offset);
            (void)n;
        } else {
            // Memory write simulation
            for (size_t k = 0; k < chunk_bytes; k += 4096) {
                target[k] = static_cast<uint8_t>((layer + k) & 0xFF);
            }
        }
        auto l_end = std::chrono::high_resolution_clock::now();
        double lat_ms = std::chrono::duration<double, std::milli>(l_end - l_start).count();
        latencies.push_back(lat_ms);

        // Touch buffer
        volatile uint8_t dummy = 0;
        for (size_t k = 0; k < chunk_bytes; k += 65536) {
            dummy ^= target[k];
        }
        (void)dummy;

        double cur_ram = get_process_rss_mb();
        if (cur_ram > peak_ram) peak_ram = cur_ram;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    result.peak_ram_mb = peak_ram;

    double total_mb = (static_cast<double>(total_layers) * static_cast<double>(chunk_bytes)) / (1024.0 * 1024.0);
    result.throughput_mb_s = result.duration_sec > 0.001 ? (total_mb / result.duration_sec) : 0.0;

    if (!latencies.empty()) {
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        result.avg_prefetch_latency_ms = sum / latencies.size();
    }

    if (fd >= 0) ::close(fd);

    for (int i = 0; i < n_buffers; ++i) {
        if (buffers[i]) free(buffers[i]);
    }

    return result;
}

} // namespace streamvex
