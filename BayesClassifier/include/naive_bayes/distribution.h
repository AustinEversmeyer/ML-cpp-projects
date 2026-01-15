#pragma once
#include <cmath>

namespace naive_bayes {

class FeatureDistribution {
 public:
  virtual ~FeatureDistribution() {}
  virtual double LogPdf(double x) const = 0;
  double Pdf(double x) const { return std::exp(LogPdf(x)); }
};

}  // namespace naive_bayes
