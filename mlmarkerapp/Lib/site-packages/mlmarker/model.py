from mlmarker.constants import (
    BINARY_MODEL_PATH,
    BINARY_FEATURES_PATH,
    MULTI_CLASS_MODEL_PATH,
    MULTI_CLASS_FEATURES_PATH,
    UPDATED_MULTI_CLASS_MODEL_PATH,
    UPDATED_MULTI_CLASS_FEATURES_PATH,
)
from mlmarker.utils import validate_sample, load_model_and_features
from mlmarker.explainability import Explainability


class MLMarker:
    def __init__(self, sample_df, binary=True, dev=False, explainer=None, penalty_factor=0):
        self.model_path = BINARY_MODEL_PATH if binary else MULTI_CLASS_MODEL_PATH
        self.features_path = BINARY_FEATURES_PATH if binary else MULTI_CLASS_FEATURES_PATH
        if dev:
            self.model_path = UPDATED_MULTI_CLASS_MODEL_PATH
            self.features_path = UPDATED_MULTI_CLASS_FEATURES_PATH
        self.model, self.features = load_model_and_features(
            self.model_path, self.features_path
        )
        self.sample = validate_sample(sample_df, self.features)
        self.penalty_factor = penalty_factor
        self.explainability = Explainability(self.model, self.features, self.sample, self.penalty_factor, explainer=explainer)


    def get_model_features(self):
        """
        Returns the features expected by the model.

        Returns:
        - list: Features used for predictions.
        """
        return self.features

    def get_model_classes(self):
        """
        Returns the classes predicted by the model.

        Returns:
        - list: Classes the model can predict.
        """
        return self.model.classes_

    def predict_top_tissues(self, n_preds=5):
        probabilities = self.model.predict_proba(self.sample).flatten()
        classes = self.model.classes_
        result = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[
            :n_preds
        ]
        return [(pred_tissue, round(prob, 4)) for pred_tissue, prob in result]

    def calculate_shap(self):
        shap_values = self.explainability.calculate_shap(self.sample)
        return shap_values

    def shap_force_plot(self, n_preds=5, tissue_name=None):
        shap_values = self.calculate_shap()
        predictions = self.predict_top_tissues(n_preds)
        tissue_name = tissue_name
        for tissue, _ in predictions:
            tissue_idx = list(self.model.classes_).index(tissue)
        self.explainability.visualize_shap_force_plot(shap_values, self.sample, tissue_idx, n_preds, tissue_name)

    def radar_chart(self, n_preds=100):
        penalty_factor = self.penalty_factor
        shap_values = self.calculate_shap()
        shap_df = self.explainability.shap_values_df(shap_values, self.sample, n_preds)
        adjusted_shap = self.explainability.adjust_absent_shap_values(
            shap_df, self.sample, self.explainability.zero_sample(), penalty_factor
        )
        self.explainability.visualize_radar_chart(adjusted_shap)

    def shap_waterfall_plot(self, n_preds=5, tissue_name=None):
            shap_values = self.calculate_shap()
            predictions = self.predict_top_tissues(n_preds)
            tissue_name = tissue_name
            for tissue, _ in predictions:
                tissue_idx = list(self.model.classes_).index(tissue)
            self.explainability.visualize_shap_waterfall(shap_values, self.sample, tissue_idx, n_preds, tissue_name)