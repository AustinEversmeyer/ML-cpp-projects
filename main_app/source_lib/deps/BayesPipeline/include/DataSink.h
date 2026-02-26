#pragma once

#include <deque>
#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace BayesPipeline {

enum class ClassificationTrigger {
    kAllFeaturesUpdated,    // every feature source must emit once since last classification
    kPrimaryFeatureUpdated, // trigger when the anchor (first model) feature updates
    kAnyFeatureUpdated,     // trigger on any feature arrival
};

struct FeatureEntry {
    double time;
    double value;
};

struct FeatureData {
    int id;
    double time;
    std::string feature_name;
    double value;
    std::optional<std::string> truth_label;
};

struct JoinedFeatureVector {
    int    id;
    double time;
    std::map<std::string, double> feature_values; // missing features stored as NaN when allow_partial
    std::optional<std::string> truth_label;
    bool is_partial = false; // true if any feature was missing / out of tolerance
};

class FeatureAlignmentStore {
public:
    static constexpr size_t kDefaultMaxRecords = 10;
    static constexpr double kDefaultTimeTolerance = 1.0; // seconds

    FeatureAlignmentStore(std::vector<std::string> model_feature_order,
                          size_t max_records = kDefaultMaxRecords,
                          double time_tolerance = kDefaultTimeTolerance);

    void RecordFeatureSample(const FeatureData& data);

    std::vector<JoinedFeatureVector> BuildJoinedFeatureVectors(bool allow_partial = false) const;

    bool ShouldClassify(ClassificationTrigger trigger) const;

    void ResetUpdatedFeatures();

private:
    std::vector<std::string> model_feature_order_;
    size_t max_records_;
    double time_tolerance_;

    std::map<int, std::map<std::string, std::deque<FeatureEntry>>> samples_by_id_and_feature_;
    std::map<int, std::string> truth_label_by_id_;
    std::set<std::string> features_updated_since_last_classification_;

    template <typename T>
    void Trim(std::deque<T>& buf) const {
        while (buf.size() > max_records_) {
            buf.pop_front();
        }
    }
};

} // namespace BayesPipeline
