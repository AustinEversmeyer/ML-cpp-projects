#pragma once

#include "BayesClassifierManager.h"
#include "BayesRuntimeConfig.h"
#include "IngestQueue.h"
#include "PipelinePublishers.h"
#include "pipeline/pipeline_config.h"

#include <cstdint>
#include <filesystem>
#include <thread>

namespace BayesPipeline {

class BayesRuntimeManager : public IFeaturePublisher {
public:
    explicit BayesRuntimeManager(std::filesystem::path runtime_config_path,
                                 std::filesystem::path model_config_path);

    explicit BayesRuntimeManager(
        std::filesystem::path model_config_path,
        std::filesystem::path output_file,
        size_t max_records,
        int64_t time_tolerance_ns,
        EvaluationPolicy evaluation_policy,
        PartialPolicy partial_policy,
        int64_t partial_grace_window_ns = BayesClassifierManager::kDefaultPartialGraceWindowNs);
    ~BayesRuntimeManager();

    void Start();
    void Stop();

    void PublishFeature(const FeatureData& data) override;

    const std::vector<ClassificationResult>& GetLatestResults() const;
    const std::filesystem::path& GetOutputFile() const;

private:
    explicit BayesRuntimeManager(std::filesystem::path model_config_path,
                                 BayesRuntimeConfig runtime_config);

    void Run();

    IngestQueue             queue_;
    BayesClassifierManager  manager_;
    std::filesystem::path   output_file_;
    std::thread             worker_;
    bool                    started_ = false;
    std::vector<naive_bayes::pipeline::BatchPredictionRow> written_rows_;
};

} // namespace BayesPipeline
