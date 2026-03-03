#include "BayesRuntimeConfig.h"

#include "io/json.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace BayesPipeline {

namespace {

std::filesystem::path ResolvePath(const std::filesystem::path& base_dir,
                                  const std::filesystem::path& path_value) {
    if (path_value.is_absolute()) {
        return path_value.lexically_normal();
    }
    return (base_dir / path_value).lexically_normal();
}

double ReadNumberField(const naive_bayes::io::Json& root,
                       const char* field_name,
                       bool required,
                       double default_value) {
    if (!root.contains(field_name)) {
        if (required) {
            throw std::runtime_error(std::string("Runtime config missing required field: ") + field_name);
        }
        return default_value;
    }
    const naive_bayes::io::Json& field = root.at(field_name);
    if (!field.is_number()) {
        throw std::runtime_error(std::string("Runtime config field must be numeric: ") + field_name);
    }
    return field.get<double>();
}

std::string ReadStringField(const naive_bayes::io::Json& root,
                            const char* field_name,
                            bool required,
                            const std::string& default_value = "") {
    if (!root.contains(field_name)) {
        if (required) {
            throw std::runtime_error(std::string("Runtime config missing required field: ") + field_name);
        }
        return default_value;
    }
    const naive_bayes::io::Json& field = root.at(field_name);
    if (!field.is_string()) {
        throw std::runtime_error(std::string("Runtime config field must be a string: ") + field_name);
    }
    return field.get<std::string>();
}

size_t ParseSizeField(const naive_bayes::io::Json& root,
                      const char* field_name,
                      size_t default_value) {
    const double value = ReadNumberField(root, field_name, /*required=*/false,
                                         static_cast<double>(default_value));
    if (!std::isfinite(value) || value <= 0.0 || std::floor(value) != value) {
        throw std::runtime_error(std::string("Runtime config field must be a positive integer: ") +
                                 field_name);
    }
    return static_cast<size_t>(value);
}

int64_t ParseNonNegativeInt64Field(const naive_bayes::io::Json& root,
                                   const char* field_name,
                                   int64_t default_value) {
    const double value = ReadNumberField(root, field_name, /*required=*/false,
                                         static_cast<double>(default_value));
    if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value) {
        throw std::runtime_error(std::string("Runtime config field must be a non-negative integer: ") +
                                 field_name);
    }
    return static_cast<int64_t>(value);
}

EvaluationPolicy ParseEvaluationPolicy(const std::string& value) {
    if (value == "immediate_any_arrival") {
        return EvaluationPolicy::kImmediateAnyArrival;
    }
    if (value == "primary_only") {
        return EvaluationPolicy::kPrimaryOnly;
    }
    if (value == "hybrid_deadline") {
        return EvaluationPolicy::kHybridDeadline;
    }
    throw std::runtime_error(
        "Invalid evaluation_policy '" + value +
        "'. Expected one of: immediate_any_arrival, primary_only, hybrid_deadline");
}

PartialPolicy ParsePartialPolicy(const std::string& value) {
    if (value == "disallow") {
        return PartialPolicy::kDisallow;
    }
    if (value == "allow_after_deadline") {
        return PartialPolicy::kAllowAfterDeadline;
    }
    if (value == "always_allow") {
        return PartialPolicy::kAlwaysAllow;
    }
    throw std::runtime_error(
        "Invalid partial_policy '" + value +
        "'. Expected one of: disallow, allow_after_deadline, always_allow");
}

}  // namespace

BayesRuntimeConfig LoadBayesRuntimeConfig(const std::filesystem::path& file_path) {
    std::ifstream input(file_path);
    if (!input) {
        throw std::runtime_error("Failed to open runtime config file: " + file_path.string());
    }

    naive_bayes::io::Json root = naive_bayes::io::Json::parse(input);
    if (!root.is_object()) {
        throw std::runtime_error("Runtime config must be a JSON object");
    }

    const std::filesystem::path config_dir = file_path.parent_path();
    BayesRuntimeConfig config;

    if (root.contains("output_file")) {
        const std::string output_file_value = ReadStringField(root, "output_file", /*required=*/false);
        if (output_file_value.empty()) {
            throw std::runtime_error("Runtime config field 'output_file' cannot be empty");
        }
        config.output_file = ResolvePath(config_dir, output_file_value);
    }

    config.max_records = ParseSizeField(root, "max_records", config.max_records);
    config.time_tolerance_ns =
        ParseNonNegativeInt64Field(root, "time_tolerance_ns", config.time_tolerance_ns);
    config.partial_grace_window_ns =
        ParseNonNegativeInt64Field(root, "partial_grace_window_ns", config.partial_grace_window_ns);

    if (root.contains("evaluation_policy")) {
        config.evaluation_policy = ParseEvaluationPolicy(
            ReadStringField(root, "evaluation_policy", /*required=*/false));
    }
    if (root.contains("partial_policy")) {
        config.partial_policy = ParsePartialPolicy(
            ReadStringField(root, "partial_policy", /*required=*/false));
    }

    return config;
}

}  // namespace BayesPipeline
