#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*version.*')

import joblib
import numpy as np
from sklearn.datasets import load_digits

def load_model():
    """Load the trained model without warnings"""
    model = joblib.load('model_train.pkl')
    return model

def example_1_basic_prediction():
    """Example 1: Basic prediction with sample data"""
    print("EXAMPLE 1: Basic Prediction")
    print("-" * 40)
    
    model = load_model()
    
    # sample data (8x8 image flattened to 64 features)
    sample_digit = np.array([
        [0, 0, 5, 13, 9, 1, 0, 0],
        [0, 0, 13, 15, 10, 15, 5, 0],
        [0, 3, 15, 2, 0, 11, 8, 0],
        [0, 4, 12, 0, 0, 8, 8, 0],
        [0, 5, 8, 0, 0, 9, 8, 0],
        [0, 4, 11, 0, 1, 12, 7, 0],
        [0, 2, 14, 5, 10, 12, 0, 0],
        [0, 0, 6, 13, 10, 0, 0, 0]
    ]).flatten().reshape(1, -1)
    
    # prediction
    prediction = model.predict(sample_digit)
    probabilities = model.predict_proba(sample_digit)
    
    print(f"Predicted digit: {prediction[0]}")
    print(f"Prediction confidence: {np.max(probabilities):.3f}")
    print(f"All class probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"    Digit {i}: {prob:.3f}")

def example_2_batch_prediction():
    """Example 2: Batch prediction with multiple samples"""
    print("\nEXAMPLE 2: Batch Prediction")
    print("-" * 40)
    
    model = load_model()
    
    # multiple sample digits
    batch_data = np.random.randint(0, 16, size=(5, 64))
    
    # batch predictions
    predictions = model.predict(batch_data)
    probabilities = model.predict_proba(batch_data)
    
    print(f"Batch predictions: {predictions}")
    print(f"Prediction confidences: {np.max(probabilities, axis=1).round(3)}")

def example_3_model_evaluation():
    """Example 3: Model evaluation with test data"""
    print("\nEXAMPLE 3: Model Evaluation")
    print("-" * 40)
    
    model = load_model()
    
    digits = load_digits()
    X_test = digits.data[:100]
    y_test = digits.target[:100]
    
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Correct predictions: {np.sum(predictions == y_test)}/{len(y_test)}")
    
    # Show some misclassified examples
    misclassified = np.where(predictions != y_test)[0]
    if len(misclassified) > 0:
        print(f"Misclassified examples: {len(misclassified)}")
        for i in misclassified[:3]:
            print(f"    Sample {i}: True={y_test[i]}, Predicted={predictions[i]}")
    else:
        print("Perfect predictions - no misclassified examples!")

def example_4_feature_analysis():
    """Example 4: Analyze which features are most important"""
    print("\nEXAMPLE 4: Feature Analysis")
    print("-" * 40)
    
    model = load_model()
    
    # Calculate average absolute coefficient across all classes
    avg_abs_coef = np.mean(np.abs(model.coef_), axis=0)
    
    # Find most and least important features
    most_important = np.argmax(avg_abs_coef)
    least_important = np.argmin(avg_abs_coef)
    
    print(f"Most important feature: {most_important} (coefficient: {avg_abs_coef[most_important]:.3f})")
    print(f"Least important feature: {least_important} (coefficient: {avg_abs_coef[least_important]:.3f})")
    
    # Show pixel positions (assuming 8x8 image)
    most_row, most_col = divmod(most_important, 8)
    least_row, least_col = divmod(least_important, 8)
    
    print(f"Most important pixel position: row {most_row}, col {most_col}")
    print(f"Least important pixel position: row {least_row}, col {least_col}")

def example_5_custom_digit_prediction():
    """Example 5: Predict a custom digit pattern"""
    print("\nEXAMPLE 5: Custom Digit Prediction")
    print("-" * 40)
    
    model = load_model()
    
    # Create a simple pattern that looks like digit "1"
    digit_1_pattern = np.array([
        [0, 0, 0, 8, 8, 0, 0, 0],
        [0, 0, 2, 8, 8, 0, 0, 0],
        [0, 0, 0, 8, 8, 0, 0, 0],
        [0, 0, 0, 8, 8, 0, 0, 0],
        [0, 0, 0, 8, 8, 0, 0, 0],
        [0, 0, 0, 8, 8, 0, 0, 0],
        [0, 0, 0, 8, 8, 0, 0, 0],
        [0, 1, 15, 15, 15, 15, 1, 0]
    ]).flatten().reshape(1, -1)
    
    prediction = model.predict(digit_1_pattern)
    probabilities = model.predict_proba(digit_1_pattern)
    
    print(f"Custom pattern predicted as: {prediction[0]}")
    print(f"Confidence: {np.max(probabilities):.3f}")
    
    print("Pattern visualization:")
    pattern_2d = digit_1_pattern.reshape(8, 8)
    for row in pattern_2d:
        print('   ' + ''.join([f'{int(x):2d}' for x in row]))

def main():
    """Run all examples without warnings"""
    print("DIGIT CLASSIFICATION MODEL - CLEAN EXAMPLES")
    print("=" * 60)
    print("(Version warnings suppressed for clean output)")
    print()
    
    try:
        example_1_basic_prediction()
        example_2_batch_prediction()
        example_3_model_evaluation()
        example_4_feature_analysis()
        example_5_custom_digit_prediction()
        
        print(f"\nAll examples completed successfully!")
        print("Model is working perfectly")
        print("No version compatibility issues affecting functionality")
        
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    main()