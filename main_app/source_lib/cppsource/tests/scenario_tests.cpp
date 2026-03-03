#include "BayesRuntimeManager.h"
#include "messaging/MessageSimulator.h"
#include "messaging/TestMessageProcessor1.h"
#include "messaging/TestMessageProcessor2.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>

namespace {

struct CsvRow {
    int64_t time_ns = 0;
    int id = -1;
    std::string classification_state;
};

std::vector<std::string> SplitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::string part;
    std::istringstream stream(line);
    while (std::getline(stream, part, ',')) {
        fields.push_back(part);
    }
    return fields;
}

std::vector<CsvRow> ReadClassificationRows(const std::filesystem::path& csv_path) {
    std::ifstream input(csv_path);
    if (!input) {
        throw std::runtime_error("Unable to open output CSV: " + csv_path.string());
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        return {};
    }
    const std::vector<std::string> header = SplitCsv(header_line);
    std::map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < header.size(); ++i) {
        index[header[i]] = i;
    }

    for (const char* required : {"time_ns", "id", "classification_state"}) {
        if (index.find(required) == index.end()) {
            throw std::runtime_error(std::string("Missing required CSV column: ") + required);
        }
    }

    std::vector<CsvRow> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = SplitCsv(line);
        CsvRow row;
        row.time_ns = std::stoll(fields[index["time_ns"]]);
        row.id = std::stoi(fields[index["id"]]);
        row.classification_state = fields[index["classification_state"]];
        rows.push_back(std::move(row));
    }
    return rows;
}

void Assert(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path UniqueTmpFile(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}

std::filesystem::path OutputPath(const std::string& name) {
    const char* out_dir = std::getenv("BAYES_SCENARIO_OUTPUT_DIR");
    if (out_dir != nullptr && out_dir[0] != '\0') {
        const std::filesystem::path dir(out_dir);
        std::filesystem::create_directories(dir);
        return dir / name;
    }
    return UniqueTmpFile(name);
}

void RunScenarioAndStop(MessageSimulator& simulator,
                        BayesPipeline::BayesRuntimeManager& runtime) {
    runtime.Start();
    simulator.Run();
    runtime.Stop();
}

void TestFullImmediate() {
    std::cout << "[ScenarioTest] full classification emits immediately when all features align\n";
    const std::filesystem::path output = OutputPath("bayes_scenario_full_immediate.csv");

    BayesPipeline::BayesRuntimeManager runtime(
        "deps/BayesPipeline/config/model/implementation.model.json",
        output,
        BayesPipeline::FeatureAlignmentStore::kDefaultMaxRecords,
        BayesPipeline::FeatureAlignmentStore::kDefaultTimeToleranceNs,
        BayesPipeline::EvaluationPolicy::kHybridDeadline,
        BayesPipeline::PartialPolicy::kAllowAfterDeadline,
        200000000LL);
    TestMessageProcessor1 proc1(runtime);
    TestMessageProcessor2 proc2(runtime);
    MessageSimulator simulator(proc1, proc2, 7);
    simulator.Clear();
    simulator.Enqueue(Proc1Message{0, 0.0, 5.0});
    simulator.Enqueue(Proc2Message{0, 0.0, 3.0});

    RunScenarioAndStop(simulator, runtime);
    std::cout << "  output: " << output << "\n";
    const std::vector<CsvRow> rows = ReadClassificationRows(output);

    bool found_full = false;
    bool found_partial = false;
    for (const CsvRow& row : rows) {
        if (row.id == 0 && row.time_ns == 0 && row.classification_state == "full") {
            found_full = true;
        }
        if (row.id == 0 && row.time_ns == 0 && row.classification_state == "partial") {
            found_partial = true;
        }
    }
    Assert(found_full, "Expected full row for id=0,time_ns=0");
    Assert(!found_partial, "Did not expect partial row when full alignment is available");
}

void TestGracePeriodPartial() {
    std::cout << "[ScenarioTest] grace window emits partial after deadline crossing event\n";
    const std::filesystem::path output = OutputPath("bayes_scenario_grace_partial.csv");

    BayesPipeline::BayesRuntimeManager runtime(
        "deps/BayesPipeline/config/model/implementation.model.json",
        output,
        BayesPipeline::FeatureAlignmentStore::kDefaultMaxRecords,
        BayesPipeline::FeatureAlignmentStore::kDefaultTimeToleranceNs,
        BayesPipeline::EvaluationPolicy::kHybridDeadline,
        BayesPipeline::PartialPolicy::kAllowAfterDeadline,
        200000000LL);
    TestMessageProcessor1 proc1(runtime);
    TestMessageProcessor2 proc2(runtime);
    MessageSimulator simulator(proc1, proc2, 11);
    simulator.Clear();

    // Anchor for id=0 at t=0. A later event for another id crosses grace boundary.
    simulator.Enqueue(Proc1Message{0, 0.0, 5.2});
    simulator.Enqueue(Proc1Message{1, 0.3, 4.9});
    simulator.Enqueue(Proc1Message{1, 0.4, 5.1});  // repeated checks should not duplicate id=0 partial

    RunScenarioAndStop(simulator, runtime);
    std::cout << "  output: " << output << "\n";
    const std::vector<CsvRow> rows = ReadClassificationRows(output);

    int partial_count_for_key = 0;
    for (const CsvRow& row : rows) {
        if (row.id == 0 && row.time_ns == 0 && row.classification_state == "partial") {
            ++partial_count_for_key;
        }
    }
    Assert(partial_count_for_key == 1,
           "Expected exactly one partial emission for (id=0,time_ns=0)");
}

void TestScenarioCsvLoaderAndStepwise() {
    std::cout << "[ScenarioTest] CSV scenario loader and stepwise dispatch callbacks\n";
    const std::filesystem::path output = OutputPath("bayes_scenario_csv_loader.csv");
    const std::filesystem::path scenario_csv = UniqueTmpFile("bayes_scenario_input.csv");

    {
        std::ofstream out(scenario_csv);
        out << "seq,id,source,time_ns,value,truth_label\n";
        out << "1,0,rcs,0,5.0,TargetMedium\n";
        out << "2,0,length,0,3.0,TargetMedium\n";
    }

    BayesPipeline::BayesRuntimeManager runtime(
        "deps/BayesPipeline/config/model/implementation.model.json",
        output,
        BayesPipeline::FeatureAlignmentStore::kDefaultMaxRecords,
        BayesPipeline::FeatureAlignmentStore::kDefaultTimeToleranceNs,
        BayesPipeline::EvaluationPolicy::kHybridDeadline,
        BayesPipeline::PartialPolicy::kAllowAfterDeadline,
        200000000LL);
    TestMessageProcessor1 proc1(runtime);
    TestMessageProcessor2 proc2(runtime);
    MessageSimulator simulator(proc1, proc2, 13);
    simulator.LoadScenarioFromCsv(scenario_csv, /*clear_first=*/true, /*sort_by_timestamp=*/false);

    std::size_t dispatch_count = 0;
    runtime.Start();
    simulator.RunStepwise([&dispatch_count](std::size_t, const SimMessage&) {
        ++dispatch_count;
    });
    runtime.Stop();
    std::cout << "  output: " << output << "\n";

    Assert(dispatch_count == 2, "Expected exactly 2 stepwise dispatch callbacks");

    const std::vector<CsvRow> rows = ReadClassificationRows(output);
    bool found_full = false;
    for (const CsvRow& row : rows) {
        if (row.id == 0 && row.time_ns == 0 && row.classification_state == "full") {
            found_full = true;
            break;
        }
    }
    Assert(found_full, "Expected full row from CSV-driven scenario");
}

void TestCheckedInSampleScenario() {
    std::cout << "[ScenarioTest] checked-in sample scenario loads and executes\n";
    const std::filesystem::path output = OutputPath("bayes_scenario_checked_in_sample.csv");
    const std::filesystem::path scenario_csv = "cppsource/tests/testdata/sample_timing_scenario.csv";
    Assert(std::filesystem::exists(scenario_csv), "Missing checked-in sample scenario CSV");

    BayesPipeline::BayesRuntimeManager runtime(
        "deps/BayesPipeline/config/model/implementation.model.json",
        output,
        BayesPipeline::FeatureAlignmentStore::kDefaultMaxRecords,
        BayesPipeline::FeatureAlignmentStore::kDefaultTimeToleranceNs,
        BayesPipeline::EvaluationPolicy::kHybridDeadline,
        BayesPipeline::PartialPolicy::kAllowAfterDeadline,
        200000000LL);
    TestMessageProcessor1 proc1(runtime);
    TestMessageProcessor2 proc2(runtime);
    MessageSimulator simulator(proc1, proc2, 17);
    simulator.LoadScenarioFromCsv(scenario_csv, /*clear_first=*/true, /*sort_by_timestamp=*/false);

    std::size_t dispatch_count = 0;
    runtime.Start();
    simulator.RunStepwise([&dispatch_count](std::size_t, const SimMessage&) {
        ++dispatch_count;
    });
    runtime.Stop();
    std::cout << "  output: " << output << "\n";

    Assert(dispatch_count == 10, "Expected 10 rows dispatched from sample scenario");
    const std::vector<CsvRow> rows = ReadClassificationRows(output);
    Assert(!rows.empty(), "Expected output rows from checked-in sample scenario");
}

}  // namespace

int main() {
    int failures = 0;
    auto run = [&failures](const char* name, void (*fn)()) {
        try {
            fn();
        } catch (const std::exception& ex) {
            ++failures;
            std::cerr << "FAILED: " << name << " -> " << ex.what() << "\n";
        }
    };

    run("TestFullImmediate", &TestFullImmediate);
    run("TestGracePeriodPartial", &TestGracePeriodPartial);
    run("TestScenarioCsvLoaderAndStepwise", &TestScenarioCsvLoaderAndStepwise);
    run("TestCheckedInSampleScenario", &TestCheckedInSampleScenario);

    if (failures == 0) {
        std::cout << "All scenario tests passed.\n";
        return 0;
    }
    std::cerr << failures << " scenario test(s) failed.\n";
    return 1;
}
