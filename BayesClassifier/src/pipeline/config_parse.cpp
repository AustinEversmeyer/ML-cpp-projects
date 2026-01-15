#include "pipeline/pipeline_config.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "io/json.h"

namespace naive_bayes::pipeline {
namespace {

using Json = naive_bayes::io::Json;

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

std::string Trim(const std::string& text) {
  return Trim(std::string_view(text));
}

std::string ReadFileToString(const std::filesystem::path& file_path) {
  std::ifstream stream(file_path);
  if (!stream) {
    throw std::runtime_error("Failed to open config file: " + file_path.string());
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

char ParseDelimiter(const std::string& token) {
  if (token == "TAB") {
    return '\t';
  }
  if (token == "SPACE") {
    return ' ';
  }
  if (token.size() == 1U) {
    return token[0];
  }
  throw std::runtime_error("Invalid delimiter specification: " + token);
}

std::filesystem::path ResolvePath(const std::filesystem::path& base_dir,
                                  const std::string& raw) {
  std::filesystem::path path_value(raw);
  if (path_value.is_relative()) {
    path_value = base_dir / path_value;
  }
  path_value = std::filesystem::absolute(path_value);
  return path_value.lexically_normal();
}

InputFormat ParseInputFormatToken(const std::string& format_value) {
  if (format_value == "json") {
    return InputFormat::kJson;
  }
  if (format_value == "text") {
    return InputFormat::kText;
  }
  throw std::runtime_error("Unknown input_format: " + format_value);
}

void EnsureLayoutConsistency(const InferenceConfig& config) {
  if (!config.input_path.has_value()) {
    return;
  }
  if (config.input_format == InputFormat::kText) {
    if (config.layout.truth_field.empty() || config.layout.feature_fields.empty()) {
      throw std::runtime_error(
          "layout must specify truth_field and feature_fields for text input");
    }
  }
}

void ApplyLayoutFromJson(LayoutConfig& layout, const Json& layout_json) {
  if (!layout_json.is_object()) {
    throw std::runtime_error("layout must be an object");
  }
  if (layout_json.contains("timestep_field") && layout_json.at("timestep_field").is_string()) {
    layout.timestep_field = layout_json.at("timestep_field").get<std::string>();
  }
  if (layout_json.contains("truth_field") && layout_json.at("truth_field").is_string()) {
    layout.truth_field = layout_json.at("truth_field").get<std::string>();
  }
  if (layout_json.contains("feature_fields") && layout_json.at("feature_fields").is_array()) {
    const auto& arr = layout_json.at("feature_fields").as_array();
    layout.feature_fields.clear();
    layout.feature_fields.reserve(arr.size());
    for (const auto& value : arr) {
      if (!value.is_string()) {
        throw std::runtime_error("feature_fields array must contain strings");
      }
      layout.feature_fields.push_back(value.get<std::string>());
    }
  }
  if (layout_json.contains("delimiter") && layout_json.at("delimiter").is_string()) {
    layout.delimiter = ParseDelimiter(layout_json.at("delimiter").get<std::string>());
  }
}

SinglePredictionConfig ParseSingleFeaturesFromJson(const Json& single_json) {
  SinglePredictionConfig single;
  if (!single_json.is_array()) {
    throw std::runtime_error("single_features must be an array of numbers");
  }

  const auto& arr = single_json.as_array();
  single.features.reserve(arr.size());
  for (const auto& value : arr) {
    if (!value.is_number()) {
      throw std::runtime_error("single_features contains non-numeric value");
    }
    single.features.push_back(value.get<double>());
  }
  if (single.features.empty()) {
    throw std::runtime_error("single_features must contain at least one value");
  }
  return single;
}

SinglePredictionConfig ParseSingleFromJsonRoot(const Json& root,
                                               const std::filesystem::path& config_dir,
                                               bool& has_single) {
  SinglePredictionConfig single;
  has_single = false;

  if (root.contains("single_features")) {
    single = ParseSingleFeaturesFromJson(root.at("single_features"));
    has_single = true;
  }

  if (root.contains("single_features_json") && root.at("single_features_json").is_string()) {
    single =
        LoadSinglePredictionFromJson(ResolvePath(config_dir, root.at("single_features_json").get<std::string>()));
    has_single = true;
  }

  if (root.contains("single_truth") && root.at("single_truth").is_string()) {
    if (!has_single) {
      throw std::runtime_error("single_truth specified without single_features");
    }
    single.truth_label = root.at("single_truth").get<std::string>();
  }

  if (root.contains("single_timestep") && root.at("single_timestep").is_number()) {
    if (!has_single) {
      throw std::runtime_error("single_timestep specified without single_features");
    }
    single.timestep = root.at("single_timestep").get<double>();
  }

  return single;
}

InferenceConfig ParseConfigFromJson(const Json& root, const std::filesystem::path& config_dir) {
  InferenceConfig config;

  if (root.contains("input_format") && root.at("input_format").is_string()) {
    config.input_format = ParseInputFormatToken(root.at("input_format").get<std::string>());
  }

  if (root.contains("input_file") && root.at("input_file").is_string()) {
    config.input_path = ResolvePath(config_dir, root.at("input_file").get<std::string>());
  }

  if (root.contains("output_file") && root.at("output_file").is_string()) {
    config.output_path = ResolvePath(config_dir, root.at("output_file").get<std::string>());
  } else {
    std::filesystem::path default_output = config_dir / ".." / "output" / "predictions.csv";
    default_output = std::filesystem::absolute(default_output);
    config.output_path = default_output.lexically_normal();
  }

  if (root.contains("model_config") && root.at("model_config").is_string()) {
    config.model_config = ResolvePath(config_dir, root.at("model_config").get<std::string>());
  }

  if (root.contains("output_use_index") && root.at("output_use_index").is_boolean()) {
    config.output_use_index = root.at("output_use_index").get<bool>();
  }

  bool has_single = false;
  SinglePredictionConfig single = ParseSingleFromJsonRoot(root, config_dir, has_single);

  if (root.contains("layout") && root.at("layout").is_object()) {
    ApplyLayoutFromJson(config.layout, root.at("layout"));
  }

  if (!config.input_path.has_value() && !has_single) {
    throw std::runtime_error("Config missing input_file or single_features");
  }

  if (has_single) {
    config.single_prediction = single;
  }

  EnsureLayoutConsistency(config);
  return config;
}

}  // namespace

[[nodiscard]] InferenceConfig LoadInferenceConfig(const std::filesystem::path& file_path) {
  std::string content = ReadFileToString(file_path);
  std::string trimmed = Trim(content);
  if (trimmed.empty()) {
    throw std::runtime_error("Config file is empty: " + file_path.string());
  }

  std::filesystem::path config_dir = file_path.parent_path();
  std::istringstream iss(content);
  Json root = Json::parse(iss);
  if (!root.is_object()) {
    throw std::runtime_error("JSON config must be an object");
  }
  InferenceConfig config = ParseConfigFromJson(root, config_dir);

  if (!config.model_config.has_value()) {
    std::filesystem::path default_model = config_dir / "model.configuration.example.json";
    if (std::filesystem::exists(default_model)) {
      config.model_config = default_model.lexically_normal();
    } else {
      throw std::runtime_error(
          "Model configuration path not provided and default 'model.configuration.example.json' not found");
    }
  }

  return config;
}

SinglePredictionConfig LoadSinglePredictionFromJson(const std::filesystem::path& file_path) {
  std::ifstream stream(file_path);
  if (!stream) {
    throw std::runtime_error("Failed to open single prediction JSON: " + file_path.string());
  }

  naive_bayes::io::Json root = naive_bayes::io::Json::parse(stream);
  if (root.is_array()) {
    return ParseSingleFeaturesFromJson(root);
  }
  if (!root.is_object()) {
    throw std::runtime_error("Single prediction JSON must be an array or object");
  }
  if (!root.contains("features")) {
    throw std::runtime_error("Single prediction JSON missing 'features'");
  }

  SinglePredictionConfig config = ParseSingleFeaturesFromJson(root.at("features"));

  if (root.contains("timestep") && root.at("timestep").is_number()) {
    config.timestep = root.at("timestep").get<double>();
  }
  if (root.contains("truth") && root.at("truth").is_string()) {
    config.truth_label = root.at("truth").get<std::string>();
  }

  return config;
}

}  // namespace naive_bayes::pipeline
