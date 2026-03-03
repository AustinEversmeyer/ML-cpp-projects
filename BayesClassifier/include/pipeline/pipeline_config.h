#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace naive_bayes {

namespace pipeline {

enum class InputFormat {
  kText,
  kJson,
};

struct LayoutConfig {
  std::string timestep_field;
  std::string truth_field;
  std::vector<std::string> feature_fields;
  char delimiter = ' ';

  std::size_t FeatureCount() const {
    return feature_fields.size();
  }
};

struct Observation {
  double timestep{};
  std::string truth_label;
  std::vector<double> features;
};

struct BatchPredictionRow {
  int64_t time_ns{};
  std::optional<int> id;
  std::string truth_label;
  std::string classification_state = "full";
  std::vector<std::pair<std::string, double>> feature_inputs;
  std::string predicted_class;
  double predicted_prob{};
  std::vector<std::pair<std::string, double>> probabilities;
  std::string predicted_group;
  double predicted_group_prob{};
  std::vector<std::pair<std::string, double>> group_probabilities;
};

struct SinglePrediction {
  std::string predicted_class;
  double predicted_prob{};
  std::vector<std::pair<std::string, double>> probabilities;
  std::string predicted_group;
  double predicted_group_prob{};
  std::vector<std::pair<std::string, double>> group_probabilities;
};

struct SinglePredictionConfig {
  std::vector<double> features;
  std::optional<double> timestep;
  std::optional<std::string> truth_label;
};

struct InferenceConfig {
  std::optional<std::filesystem::path> input_path;
  std::filesystem::path output_path;
  LayoutConfig layout;
  std::optional<SinglePredictionConfig> single_prediction;
  bool output_use_index = false;
  InputFormat input_format = InputFormat::kText;
  std::optional<std::filesystem::path> model_config;
};

[[nodiscard]] std::filesystem::path ResolveConfigPath(int argc, char** argv);

[[nodiscard]] InferenceConfig LoadInferenceConfig(const std::filesystem::path& file_path);

std::vector<Observation> LoadObservations(const std::filesystem::path& file_path,
                                          const LayoutConfig& layout,
                                          std::size_t feature_dim,
                                          InputFormat format);

SinglePredictionConfig LoadSinglePredictionFromJson(const std::filesystem::path& file_path);

void WritePredictionsCsv(const std::filesystem::path& file_path,
                         const std::vector<BatchPredictionRow>& rows,
                         bool use_row_index);

}  // namespace pipeline

}  // namespace naive_bayes
