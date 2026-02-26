#pragma once

#include "messaging/TestMessageProcessor1.h"
#include "messaging/TestMessageProcessor2.h"
#include "BayesRuntimeManager.h"

#include <memory>

class MessageSimulator;

// ---------------------------------------------------------------------------
// MainApp — top-level application class for source_lib.
//
// Owns the full stack:
//   proc1_ / proc2_               — message processors (messaging layer)
//   myBayesClassifierManager      — classification pipeline (BayesPipeline dep)
//   mySimulator                   — simulated message feed
//
// Wiring: processors ingest raw messages, map them to sink structs, and publish
// into deps-owned BayesPipeline::BayesRuntimeManager.
// ---------------------------------------------------------------------------
class MainApp {
public:
    MainApp();
    ~MainApp();

    void Run();

private:
    // Processors publish parsed payloads into BayesPipeline::BayesRuntimeManager.
    std::unique_ptr<TestMessageProcessor1> proc1_;
    std::unique_ptr<TestMessageProcessor2> proc2_;

    std::unique_ptr<BayesPipeline::BayesRuntimeManager> myBayesRuntimeManager;
    std::unique_ptr<MessageSimulator>       mySimulator;
};
