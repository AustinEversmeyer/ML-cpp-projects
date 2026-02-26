#include "DataSink.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace BayesPipeline {

FeatureAlignmentStore::FeatureAlignmentStore(std::vector<std::string> model_feature_order,
                                             size_t max_records,
                                             double time_tolerance)
    : model_feature_order_(std::move(model_feature_order))
    , max_records_(max_records)
    , time_tolerance_(time_tolerance)
{
    if (model_feature_order_.empty()) {
        throw std::invalid_argument("FeatureAlignmentStore requires at least one feature name");
    }
}

void FeatureAlignmentStore::RecordFeatureSample(const FeatureData& data) {
    std::map<std::string, std::deque<FeatureEntry>>& id_features =
        samples_by_id_and_feature_[data.id];
    std::deque<FeatureEntry>& buf = id_features[data.feature_name];
    buf.push_back({data.time, data.value});
    Trim(buf);
    if (data.truth_label.has_value()) {
        truth_label_by_id_[data.id] = data.truth_label.value();
    }
    features_updated_since_last_classification_.insert(data.feature_name);
}

std::vector<JoinedFeatureVector> FeatureAlignmentStore::BuildJoinedFeatureVectors(bool allow_partial) const {
    std::vector<JoinedFeatureVector> results;

    const std::string& anchor_feature = model_feature_order_.front();

    for (std::map<int, std::map<std::string, std::deque<FeatureEntry>>>::const_iterator id_it =
             samples_by_id_and_feature_.begin();
         id_it != samples_by_id_and_feature_.end();
         ++id_it) {
        const int id = id_it->first;
        const std::map<std::string, std::deque<FeatureEntry>>& feature_map = id_it->second;

        std::map<std::string, std::deque<FeatureEntry>>::const_iterator anchor_it =
            feature_map.find(anchor_feature);
        if (anchor_it == feature_map.end() || anchor_it->second.empty()) {
            continue;
        }

        const FeatureEntry& anchor_entry = anchor_it->second.back();
        std::map<std::string, double> matched_values;
        bool all_features_matched = true;
        bool any_feature_missing  = false;

        for (std::vector<std::string>::const_iterator req_it = model_feature_order_.begin();
             req_it != model_feature_order_.end();
             ++req_it) {
            const std::string& feature_name = *req_it;
            std::map<std::string, std::deque<FeatureEntry>>::const_iterator feat_it =
                feature_map.find(feature_name);
            if (feat_it == feature_map.end() || feat_it->second.empty()) {
                if (allow_partial) {
                    matched_values[feature_name] = std::numeric_limits<double>::quiet_NaN();
                    any_feature_missing = true;
                    continue;
                }
                all_features_matched = false;
                break;
            }

            const std::deque<FeatureEntry>& entries = feat_it->second;
            double best_delta = std::numeric_limits<double>::max();
            const FeatureEntry* best_entry = nullptr;

            for (std::deque<FeatureEntry>::const_iterator entry_it = entries.begin();
                 entry_it != entries.end();
                 ++entry_it) {
                const FeatureEntry& e = *entry_it;
                const double delta = std::fabs(e.time - anchor_entry.time);
                if (delta < best_delta) {
                    best_delta = delta;
                    best_entry = &e;
                }
            }

            if (best_entry == nullptr || best_delta > time_tolerance_) {
                if (allow_partial) {
                    matched_values[feature_name] = std::numeric_limits<double>::quiet_NaN();
                    any_feature_missing = true;
                    continue;
                }
                all_features_matched = false;
                break;
            }

            matched_values[feature_name] = best_entry->value;
        }

        if (all_features_matched || allow_partial) {
            std::optional<std::string> truth_label;
            std::map<int, std::string>::const_iterator truth_label_it =
                truth_label_by_id_.find(id);
            if (truth_label_it != truth_label_by_id_.end()) {
                truth_label = truth_label_it->second;
            }

            results.push_back({
                id,
                anchor_entry.time,
                std::move(matched_values),
                std::move(truth_label),
                any_feature_missing
            });
        }
    }

    return results;
}

bool FeatureAlignmentStore::ShouldClassify(ClassificationTrigger trigger) const {
    switch (trigger) {
        case ClassificationTrigger::kAllFeaturesUpdated:
            for (std::vector<std::string>::const_iterator it = model_feature_order_.begin();
                 it != model_feature_order_.end();
                 ++it) {
                if (features_updated_since_last_classification_.find(*it) ==
                    features_updated_since_last_classification_.end()) {
                    return false;
                }
            }
            return true;
        case ClassificationTrigger::kPrimaryFeatureUpdated:
            return features_updated_since_last_classification_.count(
                       model_feature_order_.front()) > 0;
        case ClassificationTrigger::kAnyFeatureUpdated:
            return !features_updated_since_last_classification_.empty();
    }
    return false;
}

void FeatureAlignmentStore::ResetUpdatedFeatures() {
    features_updated_since_last_classification_.clear();
}

} // namespace BayesPipeline
