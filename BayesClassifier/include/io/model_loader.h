#pragma once

#include <filesystem>

namespace naive_bayes {
class NaiveBayes;
}

namespace naive_bayes::io {

[[nodiscard]] NaiveBayes LoadModelConfiguration(const std::filesystem::path& file_path);

}  // namespace naive_bayes::io

