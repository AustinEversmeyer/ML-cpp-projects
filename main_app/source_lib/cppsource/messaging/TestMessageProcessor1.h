#pragma once

#include "PipelinePublishers.h"

#include <optional>
#include <string>

// ---------------------------------------------------------------------------
// TestMessageProcessor1
//
// Receives raw peer messages that contain:  id, time, RCS
// Validates / parses them and forwards sink-owned structured data to proc1 publisher.
// ---------------------------------------------------------------------------

struct Proc1Message {
    int    id;
    double time;
    double rcs;
    std::optional<std::string> truth_label = std::nullopt;
};

class TestMessageProcessor1 {
public:
    explicit TestMessageProcessor1(BayesPipeline::IFeaturePublisher& publisher);

    // Called by BSidePeerDataSync when a raw message arrives for this
    // processor. Performs any parsing/validation then publishes sink payloads.
    void ProcessMessage(const Proc1Message& msg);

private:
    BayesPipeline::IFeaturePublisher& publisher_;
};
