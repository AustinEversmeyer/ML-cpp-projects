#pragma once

#include "PipelinePublishers.h"

#include <optional>
#include <string>

// ---------------------------------------------------------------------------
// TestMessageProcessor2
//
// Receives raw peer messages that contain:  id, time, length
// ---------------------------------------------------------------------------

struct Proc2Message {
    int    id;
    double time;
    double length;
    std::optional<std::string> truth_label = std::nullopt;
};

class TestMessageProcessor2 {
public:
    explicit TestMessageProcessor2(BayesPipeline::IFeaturePublisher& publisher);

    void ProcessMessage(const Proc2Message& msg);

private:
    BayesPipeline::IFeaturePublisher& publisher_;
};
