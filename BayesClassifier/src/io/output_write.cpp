#include "pipeline/pipeline_config.h"

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

  output_stream << (use_row_index ? "index" : "time_step") << ",truth_label,predicted_class,predicted_prob";
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
    double leading_value = use_row_index ? static_cast<double>(i) : row.timestep;
    output_stream << leading_value << ',' << row.truth_label << ','
                  << row.predicted_class << ',' << row.predicted_prob;
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
