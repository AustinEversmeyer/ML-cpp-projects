#pragma once

#include "DataSink.h"

#include "naive_bayes/naive_bayes.h"   // from BayesClassifier/include/

#include <filesystem>
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
    double time;
    std::optional<std::string> truth_label;
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
    BayesClassifierManager(std::filesystem::path model_config_path,
                           size_t max_records         = FeatureAlignmentStore::kDefaultMaxRecords,
                           double time_tolerance      = FeatureAlignmentStore::kDefaultTimeTolerance,
                           ClassificationTrigger trigger = ClassificationTrigger::kAllFeaturesUpdated,
                           bool   allow_partial       = false);

    const std::vector<ClassificationResult>& GetLatestResults() const;

    bool ClassifyIfReady();

private:
    friend class BayesRuntimeManager;

    void RecordFeatureSample(const FeatureData& data);

    std::vector<ClassificationResult> Classify();

    std::unique_ptr<FeatureAlignmentStore>    alignment_store_;
    std::unique_ptr<naive_bayes::NaiveBayes>  bayesClassifier_;

    ClassificationTrigger trigger_;
    bool                  allow_partial_;

    std::vector<ClassificationResult> latestResults_;
};

} // namespace BayesPipeline
