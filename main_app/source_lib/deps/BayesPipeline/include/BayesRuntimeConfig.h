#pragma once

#include "BayesClassifierManager.h"
#include "DataSink.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace BayesPipeline {

struct BayesRuntimeConfig {
    std::filesystem::path output_file = "bayes_classifier_output.csv";
    size_t max_records = FeatureAlignmentStore::kDefaultMaxRecords;
    int64_t time_tolerance = FeatureAlignmentStore::kDefaultTimeTolerance;
    EvaluationPolicy evaluation_policy = EvaluationPolicy::kHybridDeadline;
    PartialPolicy partial_policy = PartialPolicy::kAllowAfterDeadline;
    int64_t partial_grace_window = BayesClassifierManager::kDefaultPartialGraceWindow;
};

[[nodiscard]] BayesRuntimeConfig LoadBayesRuntimeConfig(const std::filesystem::path& file_path);

}  // namespace BayesPipeline
