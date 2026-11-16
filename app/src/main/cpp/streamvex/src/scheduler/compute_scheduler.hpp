#ifndef COMPUTE_SCHEDULER_HPP
#define COMPUTE_SCHEDULER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>

namespace streamvex {

enum class WorkloadPriority : int {
    P0_SYSTEM_STABILITY = 0,
    P1_FOREGROUND_INTERACTIVE = 1,
    P2_VOICE_INTERACTION = 2,
    P3_LLM_INFERENCE = 3,
    P4_BACKGROUND_PREFETCH = 4
};

struct CpuFeatures {
    bool has_neon = false;
    bool has_dotprod = false;
    bool has_fp16 = false;
    bool has_sve = false;
    std::string architecture;
};

struct PrioritizedTask {
    WorkloadPriority priority;
    std::function<void()> work;
    uint64_t sequence_id = 0;

    // Highest priority (lowest numeric value) executes first
    bool operator>(const PrioritizedTask& other) const {
        if (priority != other.priority) {
            return static_cast<int>(priority) > static_cast<int>(other.priority);
        }
        return sequence_id > other.sequence_id;
    }
};

/**
 * Priority Compute Scheduler and Hardware Abstraction Layer.
 * Manages worker threads, CPU affinity, NEON/DotProd runtime dispatch,
 * and preemption between Voice AI and background LLM inference.
 */
class ComputeScheduler {
public:
    static ComputeScheduler& instance();

    ComputeScheduler();
    ~ComputeScheduler();

    ComputeScheduler(const ComputeScheduler&) = delete;
    ComputeScheduler& operator=(const ComputeScheduler&) = delete;

    /**
     * Detects CPU architecture features at runtime.
     */
    CpuFeatures detect_cpu_features();
    const CpuFeatures& cpu_features() const { return m_features; }

    /**
     * Initializes the thread pool with the specified number of compute workers.
     */
    void start(int num_workers = 2);
    void stop();

    /**
     * Enqueues a task with a designated priority level.
     */
    template <typename F, typename... Args>
    auto enqueue(WorkloadPriority priority, F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            if (m_stopping) {
                throw std::runtime_error("ComputeScheduler is stopped");
            }

            PrioritizedTask ptask;
            ptask.priority = priority;
            ptask.sequence_id = ++m_task_counter;
            ptask.work = [task]() { (*task)(); };

            m_task_queue.push(std::move(ptask));
        }
        m_cv.notify_one();
        return res;
    }

    /**
     * Adjusts active compute thread allocation based on live thermal/battery throttling.
     */
    void set_worker_cap(int max_active_workers);

    /**
     * Yields execution if higher priority workload (e.g. P2 Voice) is active.
     */
    bool should_yield(WorkloadPriority current_priority);

    /**
     * Sets thread affinity to performance (big) or efficiency (little) cores.
     */
    static bool set_thread_affinity(std::thread::native_handle_type handle, const std::vector<int>& cpu_cores);

    int active_workers() const { return m_active_workers.load(); }

private:
    void worker_loop(int worker_id);

    CpuFeatures m_features;
    std::vector<std::thread> m_workers;
    std::priority_queue<PrioritizedTask, std::vector<PrioritizedTask>, std::greater<PrioritizedTask>> m_task_queue;

    std::mutex m_queue_mutex;
    std::condition_variable m_cv;
    std::atomic<bool> m_stopping{false};
    std::atomic<uint64_t> m_task_counter{0};
    std::atomic<int> m_worker_cap{4};
    std::atomic<int> m_active_workers{0};
    std::atomic<int> m_highest_active_priority{static_cast<int>(WorkloadPriority::P4_BACKGROUND_PREFETCH)};
};

} // namespace streamvex

#endif // COMPUTE_SCHEDULER_HPP
