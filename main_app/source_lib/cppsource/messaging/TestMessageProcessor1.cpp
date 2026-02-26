#include "TestMessageProcessor1.h"

TestMessageProcessor1::TestMessageProcessor1(BayesPipeline::IFeaturePublisher& publisher)
    : publisher_(publisher)
{}

void TestMessageProcessor1::ProcessMessage(const Proc1Message& msg) {
    // TODO: add any validation/parsing/filtering here
    // e.g. discard messages with rcs <= 0, check id range, etc.
    const BayesPipeline::FeatureData data{msg.id, msg.time, "rcs", msg.rcs};
    publisher_.PublishFeature(data);
}
