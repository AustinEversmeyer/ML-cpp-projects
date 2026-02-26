#include "MessageSimulator.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

// ---------------------------------------------------------------------------
MessageSimulator::MessageSimulator(TestMessageProcessor1& proc1,
                                   TestMessageProcessor2& proc2,
                                   unsigned seed)
    : proc1_(proc1)
    , proc2_(proc2)
    , rng_(seed)
    , rcs_dist_(0.0, 1.0)   // re-parameterised per call in GenerateSynthetic
    , len_dist_(0.0, 1.0)
    , jitter_dist_(-1.0, 1.0)
{}

// ---------------------------------------------------------------------------
void MessageSimulator::Enqueue(SimMessage msg) {
    queue_.push_back(std::move(msg));
}

void MessageSimulator::Clear() {
    queue_.clear();
}

// ---------------------------------------------------------------------------
// GenerateSynthetic
//
// Proc1 fires at proc1_time_step intervals for num_steps steps.
// Proc2 fires at proc2_time_step intervals for the same total duration.
// All messages are sorted by timestamp before being appended to the queue.
// ---------------------------------------------------------------------------
void MessageSimulator::GenerateSynthetic(int num_ids, int num_steps,
                                         double proc1_time_step,
                                         double proc2_time_step,
                                         SyntheticParams params) {
    rcs_dist_    = std::normal_distribution<double>(params.rcs_mean,  params.rcs_stddev);
    len_dist_    = std::normal_distribution<double>(params.len_mean,  params.len_stddev);
    jitter_dist_ = std::uniform_real_distribution<double>(-params.time_jitter,
                                                           params.time_jitter);

    std::vector<SimMessage> new_messages;

    // Proc1: num_steps messages per ID at proc1_time_step intervals.
    for (int step = 0; step < num_steps; ++step) {
        const double t = step * proc1_time_step;
        for (int id = 0; id < num_ids; ++id) {
            new_messages.push_back(Proc1Message{id, t, rcs_dist_(rng_)});
        }
    }

    // Proc2: independent schedule over the same duration.
    const double duration    = (num_steps - 1) * proc1_time_step;
    const int    proc2_steps = static_cast<int>(duration / proc2_time_step) + 1;
    for (int step = 0; step < proc2_steps; ++step) {
        const double t = step * proc2_time_step;
        for (int id = 0; id < num_ids; ++id) {
            new_messages.push_back(Proc2Message{id, t + jitter_dist_(rng_), len_dist_(rng_)});
        }
    }

    // Sort by timestamp so dispatch order reflects realistic arrival order.
    std::sort(new_messages.begin(), new_messages.end(),
              [](const SimMessage& a, const SimMessage& b) {
                  auto time_of = [](const SimMessage& m) {
                      return std::visit([](const auto& msg) { return msg.time; }, m);
                  };
                  return time_of(a) < time_of(b);
              });

    for (SimMessage& msg : new_messages) {
        queue_.push_back(std::move(msg));
    }
}

// ---------------------------------------------------------------------------
void MessageSimulator::Run() {
    for (const SimMessage& msg : queue_) {
        Dispatch(msg);
    }
}

void MessageSimulator::RunRealTime(std::chrono::milliseconds msg_interval) {
    for (const SimMessage& msg : queue_) {
        Dispatch(msg);
        std::this_thread::sleep_for(msg_interval);
    }
}

// ---------------------------------------------------------------------------
void MessageSimulator::Dispatch(const SimMessage& msg) {
    std::visit([this](const auto& m) {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, Proc1Message>) {
            proc1_.ProcessMessage(m);
        } else if constexpr (std::is_same_v<T, Proc2Message>) {
            proc2_.ProcessMessage(m);
        }
    }, msg);
}
