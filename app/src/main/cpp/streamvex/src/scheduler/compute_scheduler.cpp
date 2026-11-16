#include "compute_scheduler.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#endif

// AArch64 HWCAP defines if not in older headers
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1 << 1)
#endif
#ifndef HWCAP_FPHP
#define HWCAP_FPHP (1 << 9)
#endif
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1 << 10)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif

namespace streamvex {

ComputeScheduler& ComputeScheduler::instance() {
    static ComputeScheduler s_instance;
    return s_instance;
}

ComputeScheduler::ComputeScheduler() {
    m_features = detect_cpu_features();
    start(2);
}

ComputeScheduler::~ComputeScheduler() {
    stop();
}

CpuFeatures ComputeScheduler::detect_cpu_features() {
    CpuFeatures f;
    f.architecture = "ARM64";

#if defined(__aarch64__)
    f.has_neon = true; // Mandatory on AArch64

#if defined(__linux__) || defined(__ANDROID__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_ASIMDDP) {
        f.has_dotprod = true;
    }
    if (hwcap & HWCAP_FPHP) {
        f.has_fp16 = true;
    }
    if (hwcap & HWCAP_SVE) {
        f.has_sve = true;
    }

    // Secondary verification via /proc/cpuinfo
    if (!f.has_dotprod) {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("Features") != std::string::npos || line.find("flags") != std::string::npos) {
                if (line.find("asimddp") != std::string::npos || line.find("dotprod") != std::string::npos) {
                    f.has_dotprod = true;
                }
                if (line.find("fphp") != std::string::npos || line.find("fp16") != std::string::npos) {
                    f.has_fp16 = true;
                }
                break;
            }
        }
    }
#endif

#else
    f.architecture = "Generic-CPU";
    f.has_neon = false;
    f.has_dotprod = false;
    f.has_fp16 = false;
#endif

    return f;
}

void ComputeScheduler::start(int num_workers) {
    stop();

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 8;
    int safe_max = std::max(2, std::min(4, static_cast<int>(hw)));
    if (num_workers <= 0 || num_workers > 4) num_workers = safe_max;
    num_workers = std::min(4, num_workers);

    m_worker_cap.store(num_workers);
    m_stopping.store(false);

    m_workers.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        m_workers.emplace_back(&ComputeScheduler::worker_loop, this, i);
    }
    m_active_workers.store(num_workers);
}

void ComputeScheduler::stop() {
    {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        if (m_stopping) return;
        m_stopping.store(true);
    }
    m_cv.notify_all();

    for (auto& w : m_workers) {
        if (w.joinable()) w.join();
    }
    m_workers.clear();

    std::unique_lock<std::mutex> lock(m_queue_mutex);
    while (!m_task_queue.empty()) {
        m_task_queue.pop();
    }
    m_active_workers.store(0);
}

void ComputeScheduler::set_worker_cap(int max_active_workers) {
    if (max_active_workers > 0) {
        m_worker_cap.store(max_active_workers);
    }
}

bool ComputeScheduler::should_yield(WorkloadPriority current_priority) {
    int highest = m_highest_active_priority.load();
    return highest < static_cast<int>(current_priority);
}

bool ComputeScheduler::set_thread_affinity(std::thread::native_handle_type handle, const std::vector<int>& cpu_cores) {
#if defined(__linux__) || defined(__ANDROID__)
    if (cpu_cores.empty()) return false;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core : cpu_cores) {
        if (core >= 0 && core < CPU_SETSIZE) {
            CPU_SET(core, &cpuset);
        }
    }

    int rc = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    (void)handle;
    return rc == 0;
#else
    (void)handle;
    (void)cpu_cores;
    return false;
#endif
}

void ComputeScheduler::worker_loop(int worker_id) {
#if defined(__linux__) || defined(__ANDROID__)
    // Allow compute workers to burst across big and mid cores (cores 0..7)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int c = 0; c < 8; ++c) {
        CPU_SET(c, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    // Set thread priority to nice level 0 (SCHED_OTHER normal priority)
    nice(0);
#endif

    while (true) {
        PrioritizedTask task;
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_cv.wait(lock, [this, worker_id]() {
                return m_stopping.load() || (!m_task_queue.empty() && worker_id < m_worker_cap.load());
            });

            if (m_stopping.load() && m_task_queue.empty()) {
                break;
            }

            if (worker_id >= m_worker_cap.load()) {
                // If throttled out, wait again
                continue;
            }

            if (m_task_queue.empty()) {
                continue;
            }

            task = std::move(const_cast<PrioritizedTask&>(m_task_queue.top()));
            m_task_queue.pop();

            // Track highest active priority
            int current_p = static_cast<int>(task.priority);
            int current_high = m_highest_active_priority.load();
            if (current_p < current_high) {
                m_highest_active_priority.store(current_p);
            }
        }

        // Execute prioritized work
        if (task.work) {
            task.work();
        }

        // Reset highest active priority after execution
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            if (m_task_queue.empty()) {
                m_highest_active_priority.store(static_cast<int>(WorkloadPriority::P4_BACKGROUND_PREFETCH));
            } else {
                m_highest_active_priority.store(static_cast<int>(m_task_queue.top().priority));
            }
        }
    }
}

} // namespace streamvex
