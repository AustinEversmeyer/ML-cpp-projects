#include <memory>
#include <stdexcept>
#include <vector>
#include "naive_bayes/factory.h"
#include "naive_bayes/gaussian.h"
#include "naive_bayes/rayleigh.h"

namespace naive_bayes {

std::unique_ptr<FeatureDistribution> CreateDistribution(
    DistributionType type, const std::vector<double>& params) {
  switch (type) {
    case DistributionType::kGaussian:
      if (params.size() != 2) {
        throw std::invalid_argument("Gaussian needs [mean, sigma]");
      }
      return std::unique_ptr<FeatureDistribution>(new Gaussian(params[0], params[1]));
    case DistributionType::kRayleigh:
      if (params.size() != 1) {
        throw std::invalid_argument("Rayleigh needs [sigma]");
      }
      return std::unique_ptr<FeatureDistribution>(new Rayleigh(params[0]));
    default:
      throw std::invalid_argument("Unknown distribution type");
  }
}

}  // namespace naive_bayes
