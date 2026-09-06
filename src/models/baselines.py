import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str = 'balanced',
    max_iter: int = 1000,
    random_state: int = 42
) -> Tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    n_unique = len(np.unique(y_train))
    kwargs = {
        'class_weight': class_weight,
        'max_iter': max_iter,
        'solver': 'lbfgs',
        'random_state': random_state,
        'n_jobs': -1,
    }
    if n_unique > 2:
        model = LogisticRegression(**kwargs)
    else:
        model = LogisticRegression(**kwargs)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 15,
    class_weight: str = 'balanced',
    random_state: int = 42
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)
    return model
