import os
import sys
import traceback
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import load_config, save_model

def train_model(config_path: str):
    try:
        # Load config
        config = load_config(config_path)
        
        # Load dataset
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

        # Initialize and train model
        model = LogisticRegression(
            penalty=config["penalty"],
            C=config["C"],
            solver=config["solver"],
            max_iter=config["max_iter"],
            multi_class='auto'
        )
        model.fit(X_train, y_train)
        print("Model trained successfully")

        # Save model
        save_model(model, config["model_output_path"])

    except Exception as e:
        print(f"Error in training pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    train_model("config/config.json")