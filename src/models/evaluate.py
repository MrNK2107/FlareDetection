import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    brier_score_loss
)
from typing import Dict, Optional


def compute_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < 2:
        return 0.0
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0])
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(tpr - fpr)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    scaler=None,
) -> Dict:
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    y_binary = (y_test > 0).astype(int)
    # Binarize the model's class predictions the same way as the labels:
    # any positive flare class (1..4) counts as a detection. Do NOT derive the
    # binary prediction from the raw integer argmax label (e.g. class 'B' = 1
    # would look like a negative under (pred > 0) confusion-matrix arithmetic).
    y_pred_binary = (np.asarray(y_pred) > 0).astype(int)
    has_positive = y_binary.sum() > 0
    cm = confusion_matrix(y_binary, y_pred_binary, labels=[0, 1])
    if has_positive and cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tss = tpr - fpr
        detection_rate = tpr
        false_alarm_rate = fpr
    else:
        # No positive test windows: detection undefined -> report 0 honestly.
        tn, fp, _, _ = cm.ravel()
        tss = 0.0
        detection_rate = 0.0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    if y_proba.shape[1] >= 2 and has_positive:
        # P(flare) = sum of positive-class probabilities (classes 1..4);
        # for binary models this equals y_proba[:, 1]
        flare_proba = y_proba[:, 1:].sum(axis=1) if y_proba.shape[1] > 2 else y_proba[:, 1]
        flare_proba = np.clip(flare_proba, 0.0, 1.0)
        brier = brier_score_loss(y_binary, flare_proba)
        try:
            auc = roc_auc_score(y_binary, flare_proba)
        except Exception:
            auc = 0.5
    else:
        brier = float(np.mean((y_proba[:, 0] - y_binary) ** 2))
        auc = 0.5
    class_labels = ['None', 'B', 'C', 'M', 'X']
    present_classes = sorted(set(y_test))
    target_names = [class_labels[i] for i in present_classes if i < len(class_labels)]
    try:
        cr = classification_report(
            y_test, y_pred,
            labels=present_classes,
            target_names=target_names,
            zero_division=0,
        )
    except Exception:
        cr = ""
    if hasattr(model, 'feature_importances_'):
        fi = {f"feature_{i}": float(v) for i, v in enumerate(model.feature_importances_)}
    else:
        fi = None
    result = {
        'model_name': model_name,
        'tss': float(tss),
        'brier_score': float(brier),
        'auc_roc': float(auc),
        'false_alarm_rate': float(false_alarm_rate),
        'detection_rate': float(detection_rate),
        'confusion_matrix': cm.tolist() if cm is not None else None,
        'classification_report': cr,
        'feature_importance': fi,
    }
    return result


def save_evaluation_results(results: Dict, path: str = "models/evaluation_results.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, val in results.items():
        if isinstance(val, Dict):
            v = {}
            for k2, v2 in val.items():
                if isinstance(v2, np.ndarray):
                    v[k2] = v2.tolist()
                elif isinstance(v2, (np.floating, np.integer)):
                    v[k2] = float(v2) if isinstance(v2, np.floating) else int(v2)
                else:
                    v[k2] = v2
            serializable[key] = v
        else:
            serializable[key] = val
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
