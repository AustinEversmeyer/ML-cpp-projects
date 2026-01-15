#include "pipeline/pipeline_config.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "io/json.h"

namespace naive_bayes::pipeline {
namespace {

bool IsWhitespace(char ch) {
  return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
}

std::string Trim(std::string_view view) {
  std::size_t start = 0;
  while (start < view.size() && IsWhitespace(view[start])) {
    ++start;
  }
  std::size_t end = view.size();
  while (end > start && IsWhitespace(view[end - 1])) {
    --end;
  }
  return std::string(view.substr(start, end - start));
}

std::string RemoveComment(const std::string& line) {
  std::size_t comment_pos = line.find('#');
  if (comment_pos == std::string::npos) {
    return line;
  }
  return line.substr(0, comment_pos);
}

std::vector<std::string> Tokenize(const std::string& line, char delimiter) {
  std::vector<std::string> tokens;
  if (delimiter == ' ' || delimiter == '\t') {
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    return tokens;
  }

  std::string current;
  std::istringstream stream(line);
  while (std::getline(stream, current, delimiter)) {
    tokens.push_back(Trim(current));
  }
  return tokens;
}

std::vector<std::string> LoadHeader(std::ifstream& stream,
                                    std::size_t& line_number,
                                    char delimiter) {
  std::string line;
  while (std::getline(stream, line)) {
    ++line_number;
    std::string trimmed = Trim(RemoveComment(line));
    if (!trimmed.empty()) {
      return Tokenize(trimmed, delimiter);
    }
  }
  throw std::runtime_error("Input file ended before header was read");
}

void EnsureColumnCount(std::size_t required_columns,
                       std::size_t /*line_number*/,
                       std::vector<std::string>& tokens) {
  if (tokens.size() < required_columns) {
    tokens.resize(required_columns, "");
  }
}

bool IsMissingToken(const std::string& token) {
  return token.empty();
}

double ParseOptionalDouble(const std::string& token,
                           std::size_t line_number,
                           const std::string& field_name,
                           double missing_value = std::numeric_limits<double>::quiet_NaN()) {
  if (IsMissingToken(token)) {
    return missing_value;
  }
  try {
    return std::stod(token);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid " + field_name + " value on line " + std::to_string(line_number));
  }
}

Observation ParseObservationFromTokens(const std::vector<std::string>& tokens,
                                       const std::vector<std::size_t>& feature_indices,
                                       std::optional<std::size_t> timestep_index,
                                       std::size_t truth_index,
                                       std::size_t line_number,
                                       double fallback_timestep) {
  Observation observation;
  if (timestep_index.has_value()) {
    double parsed = ParseOptionalDouble(tokens[*timestep_index], line_number, "timestep");
    observation.timestep = IsMissingToken(tokens[*timestep_index]) ? fallback_timestep : parsed;
  } else {
    observation.timestep = fallback_timestep;
  }
  observation.truth_label = tokens[truth_index];

  observation.features.reserve(feature_indices.size());
  for (std::size_t index : feature_indices) {
    observation.features.push_back(ParseOptionalDouble(tokens[index], line_number, "feature"));
  }

  return observation;
}

Observation ParseObservationFromJson(const naive_bayes::io::Json& node,
                                     const LayoutConfig& layout,
                                     std::size_t feature_dim,
                                     std::size_t event_index,
                                     double fallback_timestep) {
  if (!node.is_object()) {
    throw std::runtime_error("Event " + std::to_string(event_index) + " is not a JSON object");
  }

  Observation observation;
  
  std::string truth_key = layout.truth_field.empty() ? "truth" : layout.truth_field;

  std::optional<double> timestep_value;
  if (!layout.timestep_field.empty()) {
    std::string ts_key = layout.timestep_field;
    if (node.contains(ts_key) && node.at(ts_key).is_number()) {
      timestep_value = node.at(ts_key).get<double>();
    } else if (node.contains(ts_key)) {
      throw std::runtime_error("Event " + std::to_string(event_index) +
                               " timestep field is present but not numeric: " + ts_key);
    }
  } else {
    // If no timestep field is specified, accept any numeric "timestep" key or fall back to order.
    const std::string default_key = "timestep";
    if (node.contains(default_key) && node.at(default_key).is_number()) {
      timestep_value = node.at(default_key).get<double>();
    }
  }
  observation.timestep = timestep_value.value_or(fallback_timestep);
  if (!node.contains(truth_key) || !node.at(truth_key).is_string()) {
    throw std::runtime_error("Event " + std::to_string(event_index) + " missing string truth field: " + truth_key);
  }
  observation.truth_label = node.at(truth_key).get<std::string>();

  if (layout.feature_fields.empty()) {
    throw std::runtime_error("layout.feature_fields must be provided for JSON input");
  }
  if (feature_dim != 0 && layout.feature_fields.size() != feature_dim) {
    throw std::runtime_error("Layout feature count does not match classifier feature count");
  }

  observation.features.reserve(layout.feature_fields.size());
  for (const auto& key : layout.feature_fields) {
    if (!node.contains(key) || node.at(key).is_null()) {
      observation.features.push_back(std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    const naive_bayes::io::Json& val = node.at(key);
    if (!val.is_number()) {
      throw std::runtime_error("Event " + std::to_string(event_index) + " feature " + key + " is not numeric");
    }
    observation.features.push_back(val.get<double>());
  }
  
  return observation;
}

std::vector<Observation> LoadObservationsTextInternal(const std::filesystem::path& file_path,
                                                      const LayoutConfig& layout,
                                                      std::size_t feature_dim) {
  if (layout.FeatureCount() != feature_dim) {
    throw std::runtime_error("Layout feature count does not match classifier feature count");
  }

  std::ifstream input_stream(file_path);
  if (!input_stream) {
    throw std::runtime_error("Failed to open input file: " + file_path.string());
  }

  std::size_t line_number = 0;
  std::vector<std::size_t> feature_indices;
  std::optional<std::size_t> timestep_index;
  std::size_t truth_index = 0;

  std::vector<std::string> header_tokens = LoadHeader(input_stream, line_number, layout.delimiter);
  std::unordered_map<std::string, std::size_t> header_index;
  header_index.reserve(header_tokens.size());
  for (std::size_t i = 0; i < header_tokens.size(); ++i) {
    header_index.emplace(header_tokens[i], i);
  }

  if (!layout.timestep_field.empty()) {
    auto timestep_it = header_index.find(layout.timestep_field);
    if (timestep_it == header_index.end()) {
      throw std::runtime_error("Timestep column not found: " + layout.timestep_field);
    }
    timestep_index = timestep_it->second;
  }
  auto truth_it = header_index.find(layout.truth_field);
  if (truth_it == header_index.end()) {
    throw std::runtime_error("Truth column not found: " + layout.truth_field);
  }
  feature_indices.reserve(layout.feature_fields.size());
  for (const std::string& feature_name : layout.feature_fields) {
    auto feature_it = header_index.find(feature_name);
    if (feature_it == header_index.end()) {
      throw std::runtime_error("Feature column not found in header: " + feature_name);
    }
    feature_indices.push_back(feature_it->second);
  }
  truth_index = truth_it->second;

  std::vector<Observation> observations;
  std::size_t timestep_counter = 0;
  std::string line;
  while (std::getline(input_stream, line)) {
    ++line_number;
    std::string content = Trim(RemoveComment(line));
    if (content.empty()) {
      continue;
    }

    std::vector<std::string> tokens = Tokenize(content, layout.delimiter);
    std::size_t required_columns = feature_indices.empty()
                                       ? truth_index
                                       : std::max(truth_index, *std::max_element(feature_indices.begin(),
                                                                                  feature_indices.end()));
    if (timestep_index.has_value()) {
      required_columns = std::max(required_columns, *timestep_index);
    }
    ++required_columns;
    EnsureColumnCount(required_columns, line_number, tokens);

    double fallback_timestep = static_cast<double>(timestep_counter++);
    observations.push_back(ParseObservationFromTokens(tokens, feature_indices, timestep_index, truth_index,
                                                      line_number, fallback_timestep));
  }

  if (observations.empty()) {
    throw std::runtime_error("No observations found in input file: " + file_path.string());
  }

  return observations;
}

std::vector<Observation> LoadObservationsJsonInternal(const std::filesystem::path& file_path,
                                                      const LayoutConfig& layout,
                                                      std::size_t feature_dim) {
  std::ifstream input_stream(file_path);
  if (!input_stream) {
    throw std::runtime_error("Failed to open input file: " + file_path.string());
  }

  naive_bayes::io::Json root = naive_bayes::io::Json::parse(input_stream);

  std::vector<Observation> observations;
  if (!root.is_array()) {
    throw std::runtime_error("JSON input root must be an array of events");
  }

  const auto& array = root.as_array();
  observations.reserve(array.size());
  for (std::size_t i = 0; i < array.size(); ++i) {
    double fallback_timestep = static_cast<double>(i);
    observations.push_back(ParseObservationFromJson(array[i], layout, feature_dim, i, fallback_timestep));
  }

  if (observations.empty()) {
    throw std::runtime_error("No observations found in input file: " + file_path.string());
  }

  return observations;
}

}  // namespace

std::vector<Observation> LoadObservations(const std::filesystem::path& file_path,
                                          const LayoutConfig& layout,
                                          std::size_t feature_dim,
                                          InputFormat format) {
  if (format == InputFormat::kJson) {
    return LoadObservationsJsonInternal(file_path, layout, feature_dim);
  }
  return LoadObservationsTextInternal(file_path, layout, feature_dim);
}

}  // namespace naive_bayes::pipeline
