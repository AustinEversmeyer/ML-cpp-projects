#pragma once

#include "BayesClassifierManager.h"
#include "IngestQueue.h"
#include "PipelinePublishers.h"
#include "pipeline/pipeline_config.h"

#include <filesystem>
#include <thread>

namespace BayesPipeline {

class BayesRuntimeManager : public IFeaturePublisher {
public:
    explicit BayesRuntimeManager(
        std::filesystem::path model_config_path,
        std::filesystem::path output_file    = "bayes_classifier_output.csv",
        size_t max_records                   = FeatureAlignmentStore::kDefaultMaxRecords,
        double time_tolerance                = FeatureAlignmentStore::kDefaultTimeTolerance,
        ClassificationTrigger trigger        = ClassificationTrigger::kAllFeaturesUpdated,
        bool   allow_partial                 = false);
    ~BayesRuntimeManager();

    void Start();
    void Stop();

    void PublishFeature(const FeatureData& data) override;

    const std::vector<ClassificationResult>& GetLatestResults() const;
    const std::filesystem::path& GetOutputFile() const;

private:
    void Run();

    IngestQueue             queue_;
    BayesClassifierManager  manager_;
    std::filesystem::path   output_file_;
    std::thread             worker_;
    bool                    started_ = false;
    std::vector<naive_bayes::pipeline::BatchPredictionRow> written_rows_;
};

} // namespace BayesPipeline
