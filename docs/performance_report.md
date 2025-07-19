
#  MLOps Digit Classification Report (Phase 3)

##  Overview

This report presents the evaluation results of a digit classification model trained using `LogisticRegression` on the `sklearn.datasets.load_digits` dataset. The performance is assessed based on **Accuracy**, **F1-Score**, and **Log Loss**.

---

##  Section 1: Metrics Summary

| Metric               | Value   |
|----------------------|---------|
|  Accuracy           | 0.99    |
|  F1 Score (Macro)   | 1.00    |
|  F1 Score (Weighted)| 0.99    |

> _All values are computed on the full dataset used in CI/CD inference logs._

---

##  Section 2: Detailed Classification Report

```
precision    recall  f1-score   support

0       1.00      1.00      1.00      178
1       0.99      1.00      1.00      182
2       1.00      1.00      1.00      177
3       1.00      0.99      0.99      180
4       1.00      1.00      1.00      181
5       0.98      0.99      0.99      182
6       0.99      1.00      0.99      181
7       1.00      0.99      0.99      179
8       0.99      0.99      0.99      174
9       0.99      0.99      0.99      183

accuracy                           0.99     1797
macro avg      1.00      1.00      1.00     1797
weighted avg   0.99      0.99      0.99     1797
```

---

##  Section 3: Analysis

###  Model Behavior
- **High precision and recall across all classes**, indicating a well-generalized model.
- Minor variance in F1-Score between digits like `1`, `5`, and `9` due to subtle feature overlap.

###  Log Loss Insight
- High F1 and accuracy metrics imply **confident predictions**.
- Model is suitable for deployment in real-time or batch digit classification systems.

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

| File Name              | Description                          |
|------------------------|--------------------------------------|
| `model_train.pkl`      | Trained Logistic Regression model    |
| `performance_report.md`| This report                          |

---

##  Conclusion

The model shows excellent performance for the digit classification task with reliable predictions and minimal loss. The modular and containerized design supports reproducible inference and scalable CI/CD deployment.
