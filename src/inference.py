import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
import os
import sys

def run_inference(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load the trained model
        model = joblib.load(model_path)

        # Load digit data
        digits = load_digits()
        X, y = digits.data, digits.target

        # Predict
        predictions = model.predict(X)

        # Report
        report = classification_report(y, predictions)
        print("Inference Complete\n")
        print(report)
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    model_path = "model_train.pkl"
    run_inference(model_path)
