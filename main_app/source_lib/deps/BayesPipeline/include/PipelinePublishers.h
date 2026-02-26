#pragma once

#include "DataSink.h"

namespace BayesPipeline {

class IFeaturePublisher {
public:
    virtual ~IFeaturePublisher() = default;
    virtual void PublishFeature(const FeatureData& data) = 0;
};

} // namespace BayesPipeline
