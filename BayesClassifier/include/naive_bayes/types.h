#pragma once
#include <cstddef>

namespace naive_bayes {

enum class ProbabilitySpace { kLinear, kLog };
enum class DistributionType { kGaussian, kRayleigh };

inline constexpr double kTwoPi = 6.283185307179586476925286766559;

}  // namespace naive_bayes
