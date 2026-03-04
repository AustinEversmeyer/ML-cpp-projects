#include "pipeline/pipeline_config.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace naive_bayes::pipeline {

void WritePredictionsCsv(const std::filesystem::path& file_path,
                         const std::vector<BatchPredictionRow>& rows,
                         bool use_row_index) {
  (void)use_row_index;
  if (rows.empty()) {
    throw std::runtime_error("No prediction rows supplied for CSV output");
  }

  if (!file_path.parent_path().empty()) {
    std::error_code ec;
    std::filesystem::create_directories(file_path.parent_path(), ec);
    if (ec) {
      throw std::runtime_error(
          "Failed to create output directory: " + file_path.parent_path().string() +
          " (" + ec.message() + ")");
    }
  }

  std::ofstream output_stream(file_path);
  if (!output_stream) {
    throw std::runtime_error("Failed to open output file: " + file_path.string());
  }
  output_stream << std::fixed << std::setprecision(6);

  const std::vector<std::pair<std::string, double>>& canonical_feature_inputs =
      rows.front().feature_inputs;
  std::vector<std::string> canonical_feature_names;
  canonical_feature_names.reserve(canonical_feature_inputs.size());
  for (const auto& entry : canonical_feature_inputs) {
    if (std::find(canonical_feature_names.begin(), canonical_feature_names.end(), entry.first) !=
        canonical_feature_names.end()) {
      throw std::runtime_error("Duplicate feature name in output row schema: " + entry.first);
    }
    canonical_feature_names.push_back(entry.first);
  }

  output_stream << "time,id,truth_label,classification_state";
  for (const std::string& feature_name : canonical_feature_names) {
    output_stream << ",feature_" << feature_name;
  }
  output_stream << ",predicted_class,predicted_prob";
  for (const auto& entry : rows.front().probabilities) {
    output_stream << ",prob_" << entry.first;
  }
  if (!rows.front().group_probabilities.empty()) {
    output_stream << ",predicted_group,predicted_group_prob";
    for (const auto& entry : rows.front().group_probabilities) {
      output_stream << ",group_prob_" << entry.first;
    }
  }
  output_stream << '\n';

  for (std::size_t i = 0; i < rows.size(); ++i) {
    const BatchPredictionRow& row = rows[i];
    if (row.feature_inputs.size() != canonical_feature_names.size()) {
      throw std::runtime_error(
          "Feature input column count mismatch at row " + std::to_string(i) +
          ": expected " + std::to_string(canonical_feature_names.size()) + ", got " +
          std::to_string(row.feature_inputs.size()));
    }

    output_stream << row.time << ',';
    if (row.id.has_value()) {
      output_stream << *row.id;
    }
    output_stream << ',' << row.truth_label << ',' << row.classification_state;
    for (const std::string& feature_name : canonical_feature_names) {
      std::vector<std::pair<std::string, double>>::const_iterator feature_it =
          std::find_if(row.feature_inputs.begin(), row.feature_inputs.end(),
                       [&feature_name](const std::pair<std::string, double>& entry) {
                         return entry.first == feature_name;
                       });
      output_stream << ',';
      if (feature_it == row.feature_inputs.end()) {
        output_stream << "nan";
      } else {
        output_stream << feature_it->second;
      }
    }
    output_stream << ',' << row.predicted_class << ',' << row.predicted_prob;
    for (const auto& entry : row.probabilities) {
      output_stream << ',' << entry.second;
    }
    if (!row.group_probabilities.empty()) {
      output_stream << ',' << row.predicted_group << ',' << row.predicted_group_prob;
      for (const auto& entry : row.group_probabilities) {
        output_stream << ',' << entry.second;
      }
    }
    output_stream << '\n';
  }
}

}  // namespace naive_bayes::pipeline
