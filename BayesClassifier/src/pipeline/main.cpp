#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

#include "pipeline/pipeline_helpers.h"
#include "pipeline/pipeline_config.h"
#include "naive_bayes/naive_bayes.h"
#include "test/tester.h"

namespace {

static naive_bayes::pipeline::InferenceConfig LoadInferenceConfigOrExit(int argc, char** argv) {
  std::filesystem::path config_path = naive_bayes::pipeline::ResolveConfigPath(argc, argv);

  try {
    return naive_bayes::pipeline::LoadInferenceConfig(config_path);
  } catch (const std::exception& ex) {
    std::cerr << "Config Error: " << ex.what() << "\n";
    std::exit(1);
  }
}

static naive_bayes::NaiveBayes LoadClassifierOrExit(naive_bayes::pipeline::InferenceConfig& config) {
  try {
    return naive_bayes::pipeline::LoadModel(config);
  } catch (const std::exception& ex) {
    std::cerr << "Model Error: " << ex.what() << "\n";
    std::exit(1);
  }
}

static void RunBatchPipeline(const naive_bayes::NaiveBayes& clf,
                             const naive_bayes::pipeline::InferenceConfig& config) {
  const std::size_t feature_dim = clf.FeatureDim();
  if (feature_dim == 0U) {
    std::cerr << "Classifier has no features configured\n";
    std::exit(1);
  }

  // Guard: If using text, feature count must match
  if (config.input_path.has_value() &&
      config.input_format == naive_bayes::pipeline::InputFormat::kText &&
      config.layout.FeatureCount() != feature_dim) {
    std::cerr << "Config feature count (" << config.layout.FeatureCount()
              << ") does not match classifier feature count (" << feature_dim << ")\n";
    std::exit(1);
  }

  if (!config.input_path.has_value()) {
    std::cout << "No batch input file configured; skipping CSV generation.\n";
    return;
  }

  std::vector<naive_bayes::pipeline::Observation> observations;
  try {
    observations = naive_bayes::pipeline::LoadObservations(*config.input_path, config.layout,
                                                           feature_dim, config.input_format);
  } catch (const std::exception& ex) {
    std::cerr << "Data Loading Error: " << ex.what() << "\n";
    std::exit(1);
  }

  std::vector<naive_bayes::pipeline::BatchPredictionRow> prediction_rows =
      naive_bayes::pipeline::RunInference(clf, observations);

  try {
    naive_bayes::pipeline::WritePredictionsCsv(config.output_path, prediction_rows, config.output_use_index);
  } catch (const std::exception& ex) {
    std::cerr << "Output Error: " << ex.what() << "\n";
    std::exit(1);
  }

  std::cout << "Processed " << prediction_rows.size()
            << (config.output_use_index ? " rows (indexed output)." : " timesteps.")
            << " Results written to " << config.output_path << "\n";
}

static void RunSinglePrediction(const naive_bayes::NaiveBayes& clf,
                                const naive_bayes::pipeline::InferenceConfig& config) {
  if (!config.single_prediction.has_value()) {
    return;
  }

  const naive_bayes::pipeline::SinglePredictionConfig& single = *config.single_prediction;

  if (single.features.size() != clf.FeatureDim()) {
    std::cerr << "Single prediction feature count (" << single.features.size()
              << ") does not match classifier feature count (" << clf.FeatureDim() << ")\n";
    std::exit(1);
  }

  naive_bayes::pipeline::PrintSinglePrediction(clf, single);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--test") == 0) {
    return naive_bayes::test::RunTestSuite();
  }

  naive_bayes::pipeline::InferenceConfig inference_config = LoadInferenceConfigOrExit(argc, argv);
  naive_bayes::NaiveBayes classifier = LoadClassifierOrExit(inference_config);
  RunBatchPipeline(classifier, inference_config);
  RunSinglePrediction(classifier, inference_config);

  return 0;
}
