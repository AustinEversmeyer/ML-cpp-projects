#pragma once

#include "DataSink.h"

#include "naive_bayes/naive_bayes.h"   // from BayesClassifier/include/

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace BayesPipeline {

class BayesRuntimeManager;

// ---------------------------------------------------------------------------
// ClassificationResult
// ---------------------------------------------------------------------------
struct ClassificationResult {
    int    id;
    int64_t time;
    std::optional<std::string> truth_label;
    std::string classification_state = "full";
    std::vector<std::pair<std::string, double>> feature_inputs;
    std::string predicted_class;
    double      predicted_prob;
    std::vector<std::pair<std::string, double>> posteriors;
    std::string predicted_group;
    double      predicted_group_prob = 0.0;
    std::vector<std::pair<std::string, double>> group_posteriors;
    bool        is_partial = false; // true when NaN features were used in this classification
};

// ---------------------------------------------------------------------------
// BayesClassifierManager
//
class BayesClassifierManager {
public:
    static constexpr int64_t kDefaultPartialGraceWindow = 200000000; // 200ms in nanoseconds (scale to your time unit)

    BayesClassifierManager(std::filesystem::path model_config_path,
                           size_t max_records          = FeatureAlignmentStore::kDefaultMaxRecords,
                           int64_t time_tolerance      = FeatureAlignmentStore::kDefaultTimeTolerance,
                           ClassificationTrigger trigger = ClassificationTrigger::kAllFeaturesUpdated,
                           bool   allow_partial        = false);

    BayesClassifierManager(std::filesystem::path model_config_path,
                           size_t max_records,
                           int64_t time_tolerance,
                           EvaluationPolicy evaluation_policy,
                           PartialPolicy partial_policy,
                           int64_t partial_grace_window = kDefaultPartialGraceWindow);

    const std::vector<ClassificationResult>& GetLatestResults() const;

    bool ClassifyIfReady();

private:
    friend class BayesRuntimeManager;

    void RecordFeatureSample(const FeatureData& data);

    std::vector<ClassificationResult> Classify();

    std::unique_ptr<FeatureAlignmentStore>    alignment_store_;
    std::unique_ptr<naive_bayes::NaiveBayes>  bayesClassifier_;
    EvaluationPolicy      evaluation_policy_;
    PartialPolicy         partial_policy_;
    int64_t               partial_grace_window_;
    std::string           last_event_feature_;
    int64_t               last_event_time_ = 0;

    struct EmissionState {
        bool emitted_partial = false;
        bool emitted_full = false;
    };
    std::map<std::pair<int, int64_t>, EmissionState> emission_state_by_key_;

    std::vector<ClassificationResult> latestResults_;
};

} // namespace BayesPipeline
