import json
import joblib
import os

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")

def save_model(model, path: str):
    try:
        joblib.dump(model, path)
        print(f"Model saved at {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {str(e)}")