#include "TestMessageProcessor2.h"

TestMessageProcessor2::TestMessageProcessor2(BayesPipeline::IFeaturePublisher& publisher)
    : publisher_(publisher)
{}

void TestMessageProcessor2::ProcessMessage(const Proc2Message& msg) {
    // TODO: add validation/filtering here
    const BayesPipeline::FeatureData data{msg.id, msg.time, "length", msg.length};
    publisher_.PublishFeature(data);
}
