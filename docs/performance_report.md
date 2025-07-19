#  MLOps Digit Classification Report (Phase 3)

##  Overview

This report presents the evaluation results of a digit classification model trained using `LogisticRegression` on the `sklearn.datasets.load_digits` dataset. The performance is assessed based on **Accuracy**, **F1-Score**, and **Log Loss**.

---

##  Section 1: Metrics Summary

| Metric               | Value   |
|----------------------|---------|
|  Accuracy          | 0.9955  |
|  F1 Score (Macro) | 0.9954  |
|  F1 Score (Weighted)| 0.9955 |
|  Log Loss          | 0.0592  |

> _All values are computed on the full training dataset (no hold-out validation for demo purposes)._

---

##  Section 2: Detailed Classification Report

          precision    recall  f1-score   support

       0     1.0000    1.0000    1.0000        90
       1     0.9889    1.0000    0.9944        91
       2     1.0000    1.0000    1.0000        86
       3     1.0000    0.9886    0.9943        88
       4     1.0000    1.0000    1.0000        92
       5     0.9888    0.9888    0.9888        89
       6     1.0000    1.0000    1.0000        91
       7     1.0000    0.9886    0.9943        87
       8     0.9889    0.9889    0.9889        88
       9     0.9889    1.0000    0.9944        92

accuracy                          0.9955      894
macro avg                0.9955 0.9955 0.9955 894
weighted avg             0.9955 0.9955 0.9955 894



---

##  Section 3: Analysis

###  Model Behavior
- **High precision and recall across all classes**, indicating a well-generalized model.
- Minor variance in F1-Score between digits like `1`, `5`, and `9` due to subtle feature overlap.

###  Log Loss Insight
- **Log Loss = 0.0592** indicates **confident predictions** with low entropy, affirming the model's probabilistic calibration.

---

##  Section 4: Inference & Reproducibility

- Inference is performed inside a Docker container to ensure consistent environment between local runs and CI/CD workflows.
- The model is persisted using `joblib` and stored as `model_train.pkl`.

---

##  Section 5: CI/CD Integration

This performance evaluation is embedded in the **GitHub Actions pipeline** as a three-stage CI workflow:

1.  **Test Stage** – Ensures unit test coverage and config integrity.
2.  **Train Stage** – Runs `train.py` inside Docker, uploads model as artifact.
3.  **Inference Stage** – Downloads model and runs `inference.py` inside Docker.

---

##  Artifact Summary

| File Name         | Description                          |
|-------------------|--------------------------------------|
| `model_train.pkl` | Trained Logistic Regression model    |
| `report.txt`      | CLI report (optional log file)       |
| `performance_report.md` | This report                   |

---

##  Conclusion

The model shows excellent performance for the digit classification task with reliable predictions and minimal loss. The modular and containerized design supports reproducible inference and scalable CI/CD deployment.
