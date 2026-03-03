#include "BayesRuntimeManager.h"

#include <stdexcept>
#include <utility>

namespace BayesPipeline {

BayesRuntimeManager::BayesRuntimeManager(std::filesystem::path model_config_path,
                                         std::filesystem::path output_file,
                                         size_t max_records,
                                         int64_t time_tolerance_ns,
                                         ClassificationTrigger trigger,
                                         bool   allow_partial)
    : manager_(std::move(model_config_path), max_records, time_tolerance_ns, trigger, allow_partial)
    , output_file_(std::move(output_file))
{}

BayesRuntimeManager::BayesRuntimeManager(std::filesystem::path model_config_path,
                                         std::filesystem::path output_file,
                                         size_t max_records,
                                         int64_t time_tolerance_ns,
                                         EvaluationPolicy evaluation_policy,
                                         PartialPolicy partial_policy,
                                         int64_t partial_grace_window_ns)
    : manager_(std::move(model_config_path),
               max_records,
               time_tolerance_ns,
               evaluation_policy,
               partial_policy,
               partial_grace_window_ns)
    , output_file_(std::move(output_file))
{}

BayesRuntimeManager::~BayesRuntimeManager() {
    Stop();
}

void BayesRuntimeManager::Start() {
    if (started_) {
        return;
    }
    written_rows_.clear();

    worker_ = std::thread(&BayesRuntimeManager::Run, this);
    started_ = true;
}

void BayesRuntimeManager::Stop() {
    if (!started_) {
        return;
    }

    queue_.Close();
    if (worker_.joinable()) {
        worker_.join();
    }
    if (!written_rows_.empty()) {
        naive_bayes::pipeline::WritePredictionsCsv(output_file_, written_rows_, /*use_row_index=*/false);
    }
    started_ = false;
}

void BayesRuntimeManager::PublishFeature(const FeatureData& data) {
    queue_.Push(data);
}

const std::vector<ClassificationResult>& BayesRuntimeManager::GetLatestResults() const {
    return manager_.GetLatestResults();
}

const std::filesystem::path& BayesRuntimeManager::GetOutputFile() const {
    return output_file_;
}

void BayesRuntimeManager::Run() {
    FeatureData event;
    while (queue_.Pop(event)) {
        manager_.RecordFeatureSample(event);

        if (manager_.ClassifyIfReady()) {
            for (const ClassificationResult& r : manager_.GetLatestResults()) {
                naive_bayes::pipeline::BatchPredictionRow row;
                row.time_ns = r.time_ns;
                row.id = r.id;
                if (r.truth_label.has_value() && !r.truth_label.value().empty()) {
                    row.truth_label = r.truth_label.value();
                } else {
                    row.truth_label = "nan";
                }
                row.classification_state = r.classification_state;
                row.feature_inputs = r.feature_inputs;
                row.predicted_class = r.predicted_class;
                row.predicted_prob = r.predicted_prob;
                row.probabilities = r.posteriors;
                row.predicted_group = r.predicted_group;
                row.predicted_group_prob = r.predicted_group_prob;
                row.group_probabilities = r.group_posteriors;
                written_rows_.push_back(std::move(row));
            }
        }
    }
}

} // namespace BayesPipeline
