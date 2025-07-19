import os
import json
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.utils import load_config, save_model

CONFIG_PATH = "config/config.json"

@pytest.fixture
def config():
    return load_config(CONFIG_PATH)

# --------------- (a) Config File Tests ---------------

def test_config_file_exists():
    assert os.path.exists(CONFIG_PATH), "config.json not found"

def test_required_hyperparameters(config):
    assert "C" in config
    assert "solver" in config
    assert "max_iter" in config

def test_hyperparameter_types(config):
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

# --------------- (b) Model Creation Tests ---------------

def train_model_for_test(config):
    digits = load_digits()
    X_train, _, y_train, _ = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

    model = LogisticRegression(
        penalty=config.get("penalty", "l2"),
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"],
        multi_class="multinomial"
    )
    model.fit(X_train, y_train)
    return model

def test_model_instance(config):
    model = train_model_for_test(config)
    assert isinstance(model, LogisticRegression)

def test_model_fitted(config):
    model = train_model_for_test(config)
    assert hasattr(model, "coef_")
    assert hasattr(model, "classes_")

# --------------- (c) Accuracy Tests ---------------

def test_model_accuracy(config):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

    model = LogisticRegression(
        penalty=config.get("penalty", "l2"),
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"],
        multi_class="multinomial"
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")
    assert accuracy > 0.85
