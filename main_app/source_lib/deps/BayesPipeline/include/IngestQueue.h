#pragma once

#include "DataSink.h"

#include <condition_variable>
#include <deque>
#include <mutex>

namespace BayesPipeline {

class IngestQueue {
public:
    void Push(FeatureData event);
    bool Pop(FeatureData& out_event);
    void Close();

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::deque<FeatureData> queue_;
    bool                    closed_ = false;
};

} // namespace BayesPipeline
