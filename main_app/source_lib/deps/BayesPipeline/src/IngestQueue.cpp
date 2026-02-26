#include "IngestQueue.h"

#include <utility>

namespace BayesPipeline {

void IngestQueue::Push(FeatureData event) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        queue_.push_back(std::move(event));
    }
    cv_.notify_one();
}

bool IngestQueue::Pop(FeatureData& out_event) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() {
        return closed_ || !queue_.empty();
    });

    if (queue_.empty()) {
        return false;
    }

    out_event = std::move(queue_.front());
    queue_.pop_front();
    return true;
}

void IngestQueue::Close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    cv_.notify_all();
}

} // namespace BayesPipeline
