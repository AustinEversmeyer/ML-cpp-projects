# Scenario Test Data

This folder contains deterministic input streams for `MessageSimulator::LoadScenarioFromCsv(...)`.

## File format

CSV header:

`seq,id,source,time,value,truth_label`

- `seq`: ordering key in the file (kept for readability; dispatch order is file order unless sorted in code).
- `id`: track/object id used by Bayes runtime.
- `source`: `rcs` or `length`.
- `time`: event timestamp as an opaque integer tick value (unit is caller-defined; current samples use nanoseconds).
- `value`: numeric feature value.
- `truth_label`: optional truth class label.

## Included sample

- `sample_timing_scenario.csv`
  - includes immediate full alignments
  - includes delayed second-feature arrivals
  - includes out-of-order feature-source arrival patterns
  - suitable for repeatable grace-window behavior checks

## Usage

In tests or harness code:

```cpp
simulator.LoadScenarioFromCsv(
    "cppsource/tests/testdata/sample_timing_scenario.csv",
    /*clear_first=*/true,
    /*sort_by_timestamp=*/false);
```

Set `sort_by_timestamp=true` if you want strict event-time replay independent of row order.
