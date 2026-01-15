#pragma once
#include <memory>
#include <vector>
#include "naive_bayes/types.h"
#include "naive_bayes/distribution.h"

namespace naive_bayes {

std::unique_ptr<FeatureDistribution> CreateDistribution(DistributionType type,
                                                        const std::vector<double>& params);

}  // namespace naive_bayes
