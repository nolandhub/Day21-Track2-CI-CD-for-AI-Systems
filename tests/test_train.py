import os
import json
import numpy as np
import pandas as pd
from src.train import train

FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]

def _make_temp_data(tmp_path):
    """Tạo dataset giả lập để test."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, n)
    
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    
    train_path = os.path.join(tmp_path, "train.csv")
    eval_path = os.path.join(tmp_path, "eval.csv")
    
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    
    return train_path, eval_path

def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về accuracy (float)."""
    train_path, eval_path = _make_temp_data(tmp_path)
    acc = train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_metrics_file_created(tmp_path):
    """Kiểm tra file metrics.json có được tạo không."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json", "r") as f:
        metrics = json.load(f)
        assert "accuracy" in metrics
        assert "f1_score" in metrics

def test_model_file_created(tmp_path):
    """Kiểm tra file model.pkl có được tạo không."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("models/model.pkl")
