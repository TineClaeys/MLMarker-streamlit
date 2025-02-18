import pandas as pd
import joblib
import logging

# Set up logging in the MLMarker class
logger = logging.getLogger(__name__)

def validate_sample(sample_df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    Validate and transform the input sample for compatibility with the model.
    Logs added, removed, and remaining features.
    """
    matched_features = [f for f in model_features if f in sample_df.columns]
    added_features = [f for f in model_features if f not in sample_df.columns]
    removed_features = [f for f in sample_df.columns if f not in model_features]

    logger.debug(
        f"Features added: {len(added_features)}, removed: {len(removed_features)}, "
        f"remaining: {len(matched_features)}"
    )

    added_features_df = pd.DataFrame(
        0, index=sample_df.index, columns=added_features
    )
    validated_sample = pd.concat([sample_df[matched_features], added_features_df], axis=1)
    validated_sample = validated_sample[model_features]  # Ensure correct column order

    # Drop duplicate columns
    validated_sample = validated_sample.loc[:, ~validated_sample.columns.duplicated()]
    logger.info(f"Validated sample with {len(validated_sample.columns)} features.")
    
    return validated_sample

def load_model_and_features(model_path: str, features_path: str):
    """
    Load the model and features list.
    """
    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        features = f.read().strip().split(",\n")
    return model, features
