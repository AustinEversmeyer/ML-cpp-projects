#include "MessageSimulator.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

constexpr double kNsPerSecond = 1000000000.0;

double ToSeconds(int64_t time_ns) {
    return static_cast<double>(time_ns) / kNsPerSecond;
}

std::string Trim(const std::string& input) {
    std::size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
        ++start;
    }
    std::size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(start, end - start);
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    std::istringstream stream(line);
    while (std::getline(stream, current, ',')) {
        fields.push_back(Trim(current));
    }
    return fields;
}

double TimeOf(const SimMessage& msg) {
    return std::visit([](const auto& m) { return m.time; }, msg);
}

}  // namespace

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

void MessageSimulator::EnqueueMany(const std::vector<SimMessage>& messages) {
    queue_.insert(queue_.end(), messages.begin(), messages.end());
}

void MessageSimulator::Clear() {
    queue_.clear();
}

void MessageSimulator::SortByTimestamp() {
    std::stable_sort(queue_.begin(), queue_.end(),
                     [](const SimMessage& lhs, const SimMessage& rhs) {
                         return TimeOf(lhs) < TimeOf(rhs);
                     });
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

    std::stable_sort(new_messages.begin(), new_messages.end(),
                     [](const SimMessage& lhs, const SimMessage& rhs) {
                         return TimeOf(lhs) < TimeOf(rhs);
                     });

    for (SimMessage& msg : new_messages) {
        queue_.push_back(std::move(msg));
    }
}

void MessageSimulator::LoadScenarioFromCsv(const std::filesystem::path& csv_path,
                                           bool clear_first,
                                           bool sort_by_timestamp) {
    std::ifstream input(csv_path);
    if (!input) {
        throw std::runtime_error("Failed to open scenario CSV: " + csv_path.string());
    }

    if (clear_first) {
        Clear();
    }

    std::string header_line;
    bool found_header = false;
    while (std::getline(input, header_line)) {
        const std::string trimmed = Trim(header_line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        header_line = trimmed;
        found_header = true;
        break;
    }
    if (!found_header) {
        throw std::runtime_error("Scenario CSV is empty or missing header: " + csv_path.string());
    }
    const std::vector<std::string> header = SplitCsvLine(header_line);
    auto find_index = [&header](const std::string& name) -> std::size_t {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == name) {
                return i;
            }
        }
        throw std::runtime_error("Scenario CSV missing required column: " + name);
    };

    const std::size_t id_idx = find_index("id");
    const std::size_t source_idx = find_index("source");
    const std::size_t time_ns_idx = find_index("time_ns");
    const std::size_t value_idx = find_index("value");
    std::size_t truth_idx = header.size();
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i] == "truth_label") {
            truth_idx = i;
            break;
        }
    }

    std::string line;
    std::size_t line_no = 1;
    while (std::getline(input, line)) {
        ++line_no;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        const std::vector<std::string> fields = SplitCsvLine(trimmed);
        if (fields.size() < header.size()) {
            throw std::runtime_error("Scenario CSV line " + std::to_string(line_no) +
                                     " has too few fields");
        }

        const int id = std::stoi(fields[id_idx]);
        const std::string source = fields[source_idx];
        const int64_t time_ns = std::stoll(fields[time_ns_idx]);
        const double value = std::stod(fields[value_idx]);
        std::optional<std::string> truth_label = std::nullopt;
        if (truth_idx < fields.size() && !fields[truth_idx].empty()) {
            truth_label = fields[truth_idx];
        }

        if (source == "rcs") {
            Enqueue(Proc1Message{id, ToSeconds(time_ns), value, truth_label});
        } else if (source == "length") {
            Enqueue(Proc2Message{id, ToSeconds(time_ns), value, truth_label});
        } else {
            throw std::runtime_error("Scenario CSV line " + std::to_string(line_no) +
                                     " has unknown source '" + source + "'");
        }
    }

    if (sort_by_timestamp) {
        SortByTimestamp();
    }
}

// ---------------------------------------------------------------------------
void MessageSimulator::Run() {
    for (const SimMessage& msg : queue_) {
        Dispatch(msg);
    }
}

void MessageSimulator::RunStepwise(
    const std::function<void(std::size_t, const SimMessage&)>& after_dispatch) {
    for (std::size_t i = 0; i < queue_.size(); ++i) {
        const SimMessage& msg = queue_[i];
        Dispatch(msg);
        after_dispatch(i, msg);
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
