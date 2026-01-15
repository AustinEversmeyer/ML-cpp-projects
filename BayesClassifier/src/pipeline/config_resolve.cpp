#include "pipeline/pipeline_config.h"

#include <filesystem>
#include <vector>

namespace naive_bayes::pipeline {

[[nodiscard]] std::filesystem::path ResolveConfigPath(int argc, char** argv) {
  std::filesystem::path executable_path =
      std::filesystem::weakly_canonical(std::filesystem::path(argv[0]));
  std::filesystem::path executable_dir = executable_path.parent_path();
  std::filesystem::path parent_dir = executable_dir.parent_path();

  std::vector<std::filesystem::path> config_candidates;
  if (argc > 1) {
    std::filesystem::path arg_path(argv[1]);
    config_candidates.push_back(arg_path);
    if (!arg_path.is_absolute()) {
      config_candidates.push_back(executable_dir / arg_path);
      if (!parent_dir.empty()) {
        config_candidates.push_back(parent_dir / arg_path);
      }
    }
  } else {
    config_candidates.push_back(executable_dir / "config" / "pipeline.json");
    if (!parent_dir.empty()) {
      config_candidates.push_back(parent_dir / "config" / "pipeline.json");
    }
  }

  for (const std::filesystem::path& candidate : config_candidates) {
    if (std::filesystem::exists(candidate)) {
      return std::filesystem::weakly_canonical(candidate);
    }
  }

  std::filesystem::path fallback = config_candidates.front();
  if (!fallback.is_absolute()) {
    fallback = std::filesystem::absolute(executable_dir / fallback);
  }
  return fallback.lexically_normal();
}

}  // namespace naive_bayes::pipeline
