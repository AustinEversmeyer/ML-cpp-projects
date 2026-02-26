#include "BayesClassifierManager.h"

#include "io/model_loader.h"   // naive_bayes::io::LoadModelConfiguration()

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace BayesPipeline {

BayesClassifierManager::BayesClassifierManager(std::filesystem::path model_config_path,
                                               size_t max_records,
                                               double time_tolerance,
                                               ClassificationTrigger trigger,
                                               bool   allow_partial)
    : trigger_(trigger)
    , allow_partial_(allow_partial)
{
    naive_bayes::NaiveBayes loaded = naive_bayes::io::LoadModelConfiguration(model_config_path);
    const std::vector<std::string> feature_names = loaded.FeatureNames();
    bayesClassifier_ = std::make_unique<naive_bayes::NaiveBayes>(std::move(loaded));

    alignment_store_ = std::make_unique<FeatureAlignmentStore>(feature_names, max_records, time_tolerance);
}

void BayesClassifierManager::RecordFeatureSample(const FeatureData& data) {
    alignment_store_->RecordFeatureSample(data);
}

const std::vector<ClassificationResult>& BayesClassifierManager::GetLatestResults() const {
    return latestResults_;
}

bool BayesClassifierManager::ClassifyIfReady() {
    if (!alignment_store_->ShouldClassify(trigger_)) {
        return false;
    }
    latestResults_ = Classify();
    alignment_store_->ResetUpdatedFeatures();
    return true;
}

std::vector<ClassificationResult> BayesClassifierManager::Classify() {
    std::vector<ClassificationResult> results;
    const std::vector<std::string>& feature_names = bayesClassifier_->FeatureNames();

    for (const JoinedFeatureVector& rec : alignment_store_->BuildJoinedFeatureVectors(allow_partial_)) {
        std::vector<double> features;
        features.reserve(feature_names.size());
        for (std::vector<std::string>::const_iterator name_it = feature_names.begin();
             name_it != feature_names.end();
             ++name_it) {
            const std::string& feature_name = *name_it;
            std::map<std::string, double>::const_iterator value_it =
                rec.feature_values.find(feature_name);
            if (value_it == rec.feature_values.end()) {
                if (allow_partial_) {
                    features.push_back(std::numeric_limits<double>::quiet_NaN());
                    continue;
                }
                throw std::runtime_error(
                    "Missing aligned feature '" + feature_name + "' for id " +
                    std::to_string(rec.id));
            }
            features.push_back(value_it->second);
        }

        const std::pair<std::string, double> prediction =
            bayesClassifier_->PredictClass(features);

        std::vector<std::pair<std::string, double>> posteriors =
            bayesClassifier_->PredictPosteriors(features);

        ClassificationResult result;
        result.id            = rec.id;
        result.time          = rec.time;
        result.truth_label   = rec.truth_label;
        result.predicted_class = std::move(prediction.first);
        result.predicted_prob  = prediction.second;
        result.posteriors      = std::move(posteriors);
        result.is_partial      = rec.is_partial;

        if (bayesClassifier_->HasClassGroups()) {
            result.group_posteriors = bayesClassifier_->PredictGroupedPosteriors(features);
            const auto best = std::max_element(
                result.group_posteriors.begin(), result.group_posteriors.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            if (best != result.group_posteriors.end()) {
                result.predicted_group      = best->first;
                result.predicted_group_prob = best->second;
            }
        }

        results.push_back(std::move(result));
    }

    return results;
}

} // namespace BayesPipeline
