#include "io/model_loader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "io/json.h"
#include "naive_bayes/factory.h"
#include "naive_bayes/naive_bayes.h"

namespace naive_bayes::io {
namespace {

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

ProbabilitySpace ParseLogMode(const naive_bayes::io::Json& root) {
  const char* keys[] = {"computation_mode", "log_mode"};
  for (const char* key : keys) {
    if (root.contains(key) && root.at(key).is_string()) {
      std::string mode = ToLower(root.at(key).get<std::string>());
      if (mode == "log") {
        return ProbabilitySpace::kLog;
      }
      if (mode == "linear") {
        return ProbabilitySpace::kLinear;
      }
      throw std::runtime_error("Unsupported computation_mode value in model configuration: " + mode);
    }
  }
  return ProbabilitySpace::kLog;
}

DistributionType ParseDistributionType(const std::string& type_string) {
  std::string lower = ToLower(type_string);
  if (lower == "gaussian") {
    return DistributionType::kGaussian;
  }
  if (lower == "rayleigh") {
    return DistributionType::kRayleigh;
  }
  throw std::runtime_error("Unsupported distribution type in model configuration: " + type_string);
}

std::vector<double> ParseParams(const naive_bayes::io::Json& node,
                                const std::string& feature_name,
                                DistributionType dist_type) {
  if (!node.is_object()) {
    throw std::runtime_error("Parameters for feature '" + feature_name + "' must be an object");
  }
  std::vector<double> params;
  switch (dist_type) {
    case DistributionType::kGaussian: {
      if (!node.contains("mean") || !node.at("mean").is_number()) {
        throw std::runtime_error("Feature '" + feature_name + "' missing numeric 'mean'");
      }
      if (!node.contains("sigma") || !node.at("sigma").is_number()) {
        throw std::runtime_error("Feature '" + feature_name + "' missing numeric 'sigma'");
      }
      params.push_back(node.at("mean").get<double>());
      params.push_back(node.at("sigma").get<double>());
      break;
    }
    case DistributionType::kRayleigh: {
      if (!node.contains("sigma") || !node.at("sigma").is_number()) {
        throw std::runtime_error("Feature '" + feature_name + "' missing numeric 'sigma'");
      }
      params.push_back(node.at("sigma").get<double>());
      break;
    }
    default:
      throw std::runtime_error("Unsupported distribution for feature '" + feature_name + "'");
  }
  return params;
}

void ValidateParams(DistributionType type,
                    const std::vector<double>& params,
                    const std::string& context) {
  switch (type) {
    case DistributionType::kGaussian:
      if (params.size() != 2) {
        throw std::runtime_error("Gaussian distribution for " + context + " requires [mean, sigma]");
      }
      break;
    case DistributionType::kRayleigh:
      if (params.size() != 1) {
        throw std::runtime_error("Rayleigh distribution for " + context + " requires [sigma]");
      }
      break;
    default:
      throw std::runtime_error("Unknown distribution type for " + context);
  }
}

std::unique_ptr<FeatureDistribution> BuildFeatureModel(const naive_bayes::io::Json& feature_node,
                                                       std::size_t feature_index) {
  if (!feature_node.is_object()) {
    throw std::runtime_error("Feature index " + std::to_string(feature_index) + " is not an object");
  }
  if (!feature_node.contains("name") || !feature_node.at("name").is_string()) {
    throw std::runtime_error("Feature index " + std::to_string(feature_index) + " missing 'name'");
  }
  const std::string feature_name = feature_node.at("name").get<std::string>();

  if (!feature_node.contains("type") || !feature_node.at("type").is_string()) {
    throw std::runtime_error("Feature '" + feature_name + "' missing 'type'");
  }
  std::string type_string = feature_node.at("type").get<std::string>();
  DistributionType dist_type = ParseDistributionType(type_string);

  if (!feature_node.contains("params")) {
    throw std::runtime_error("Feature '" + feature_name + "' missing 'params'");
  }
  std::vector<double> params = ParseParams(feature_node.at("params"), feature_name, dist_type);
  ValidateParams(dist_type, params, "feature '" + feature_name + "'");

  return CreateDistribution(dist_type, params);
}

ClassDefinition BuildClassDefinition(const naive_bayes::io::Json& class_node,
                                     std::size_t class_index,
                                     const std::vector<std::string>* expected_feature_names) {
  if (!class_node.is_object()) {
    throw std::runtime_error("Class index " + std::to_string(class_index) + " is not an object");
  }

  if (!class_node.contains("name") || !class_node.at("name").is_string()) {
    throw std::runtime_error("Class index " + std::to_string(class_index) + " missing 'name'");
  }
  if (!class_node.contains("prior") || !class_node.at("prior").is_number()) {
    throw std::runtime_error("Class index " + std::to_string(class_index) + " missing numeric 'prior'");
  }
  if (!class_node.contains("features")) {
    throw std::runtime_error("Class index " + std::to_string(class_index) + " missing 'features'");
  }

  const naive_bayes::io::Json& features = class_node.at("features");
  if (!features.is_array()) {
    throw std::runtime_error("'features' for class index " + std::to_string(class_index) + " must be an array");
  }
  if (features.size() == 0) {
    throw std::runtime_error("Class index " + std::to_string(class_index) + " must declare at least one feature");
  }

  ClassDefinition model;
  model.name = class_node.at("name").get<std::string>();
  model.prior = class_node.at("prior").get<double>();

  // First class defines global feature order.
  if (expected_feature_names == nullptr) {
    std::unordered_set<std::string> seen;
    model.feature_models.reserve(features.size());
    model.feature_names.reserve(features.size());
    for (std::size_t i = 0; i < features.size(); ++i) {
      if (!features.at(i).is_object() || !features.at(i).contains("name")) {
        throw std::runtime_error("Feature index " + std::to_string(i) + " missing 'name'");
      }
      std::string feature_name = features.at(i).at("name").get<std::string>();
      if (!seen.insert(feature_name).second) {
        throw std::runtime_error("Duplicate feature '" + feature_name + "' in class index " +
                                 std::to_string(class_index));
      }
      model.feature_names.push_back(feature_name);
      model.feature_models.push_back(BuildFeatureModel(features.at(i), i));
    }
    return model;
  }

  // Subsequent classes align to the provided global feature set.
  model.feature_names = *expected_feature_names;
  model.feature_models.resize(expected_feature_names->size());

  std::unordered_map<std::string, std::size_t> feature_to_index;
  feature_to_index.reserve(expected_feature_names->size());
  for (std::size_t i = 0; i < expected_feature_names->size(); ++i) {
    feature_to_index.emplace((*expected_feature_names)[i], i);
  }

  std::unordered_set<std::string> seen;
  for (std::size_t i = 0; i < features.size(); ++i) {
    if (!features.at(i).is_object() || !features.at(i).contains("name")) {
      throw std::runtime_error("Feature index " + std::to_string(i) + " missing 'name'");
    }
    std::string feature_name = features.at(i).at("name").get<std::string>();
    if (!seen.insert(feature_name).second) {
      throw std::runtime_error("Duplicate feature '" + feature_name + "' in class index " +
                               std::to_string(class_index));
    }
    auto it = feature_to_index.find(feature_name);
    if (it == feature_to_index.end()) {
      throw std::runtime_error("Feature '" + feature_name + "' in class index " +
                               std::to_string(class_index) +
                               " not declared in the global feature list");
    }
    model.feature_models[it->second] = BuildFeatureModel(features.at(i), i);
  }

  return model;
}

std::vector<ClassGroup> ParseClassGroups(const naive_bayes::io::Json& root) {
  const char* keys[] = {"class_groups", "groups"};
  const char* found_key = nullptr;
  for (const char* key : keys) {
    if (root.contains(key)) {
      if (found_key != nullptr) {
        throw std::runtime_error("Model configuration contains multiple group keys");
      }
      found_key = key;
    }
  }
  if (found_key == nullptr) {
    return {};
  }
  const naive_bayes::io::Json& group_node = root.at(found_key);
  if (!group_node.is_array()) {
    throw std::runtime_error("Model configuration '" + std::string(found_key) + "' must be an array");
  }
  std::vector<ClassGroup> groups;
  groups.reserve(group_node.size());
  for (std::size_t i = 0; i < group_node.size(); ++i) {
    const auto& item = group_node.at(i);
    if (!item.is_object()) {
      throw std::runtime_error("Group index " + std::to_string(i) + " is not an object");
    }
    if (!item.contains("name") || !item.at("name").is_string()) {
      throw std::runtime_error("Group index " + std::to_string(i) + " missing 'name'");
    }
    if (!item.contains("classes") || !item.at("classes").is_array()) {
      throw std::runtime_error("Group index " + std::to_string(i) + " missing 'classes' array");
    }
    ClassGroup group;
    group.name = item.at("name").get<std::string>();
    const auto& class_nodes = item.at("classes").as_array();
    if (class_nodes.empty()) {
      throw std::runtime_error("Group '" + group.name + "' must declare at least one class");
    }
    group.class_names.reserve(class_nodes.size());
    for (std::size_t j = 0; j < class_nodes.size(); ++j) {
      if (!class_nodes[j].is_string()) {
        throw std::runtime_error("Group '" + group.name + "' class index " + std::to_string(j) +
                                 " must be a string");
      }
      group.class_names.push_back(class_nodes[j].get<std::string>());
    }
    groups.push_back(std::move(group));
  }
  return groups;
}

}  // namespace

NaiveBayes LoadModelConfiguration(const std::filesystem::path& file_path) {
  std::ifstream stream(file_path);
  if (!stream) {
    throw std::runtime_error("Failed to open model configuration: " + file_path.string());
  }

  naive_bayes::io::Json root = naive_bayes::io::Json::parse(stream);
  if (!root.is_object()) {
    throw std::runtime_error("Model configuration must be a JSON object");
  }

  ProbabilitySpace mode = ParseLogMode(root);
  NaiveBayes model(mode);

  if (!root.contains("classes") || !root.at("classes").is_array()) {
    throw std::runtime_error("Model configuration must contain a 'classes' array");
  }
  const std::vector<naive_bayes::io::Json>& classes = root.at("classes").as_array();
  if (classes.empty()) {
    throw std::runtime_error("Model configuration 'classes' array is empty");
  }

  // First pass: build global feature union in first-seen order.
  std::vector<std::string> global_feature_names;
  std::unordered_set<std::string> feature_seen;
  for (std::size_t i = 0; i < classes.size(); ++i) {
    const auto& features = classes.at(i).at("features");
    if (!features.is_array()) {
      throw std::runtime_error("'features' for class index " + std::to_string(i) + " must be an array");
    }
    for (std::size_t j = 0; j < features.size(); ++j) {
      if (!features.at(j).is_object() || !features.at(j).contains("name")) {
        throw std::runtime_error("Feature index " + std::to_string(j) + " missing 'name' in class index " +
                                 std::to_string(i));
      }
      std::string fname = features.at(j).at("name").get<std::string>();
      if (feature_seen.insert(fname).second) {
        global_feature_names.push_back(fname);
      }
    }
  }

  if (global_feature_names.empty()) {
    throw std::runtime_error("No features declared across any class");
  }

  // Second pass: construct each class aligned to the global list.
  for (std::size_t i = 0; i < classes.size(); ++i) {
    ClassDefinition class_model = BuildClassDefinition(classes.at(i), i, &global_feature_names);
    model.AddClassDefinition(std::move(class_model));
  }

  std::vector<ClassGroup> groups = ParseClassGroups(root);
  if (!groups.empty()) {
    model.SetClassGroups(std::move(groups));
  }

  return model;
}

}  // namespace naive_bayes::io
