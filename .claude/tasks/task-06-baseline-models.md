# Task 6: Baseline ML Models

**Dependencies**: Task 5 (Physics Layer)
**Estimated effort**: Medium
**PRD Reference**: Phase 4 — ML Model Architecture (§3.4.1); Evaluation Framework (§4)

## Objective

Train and evaluate baseline models: Logistic Regression (L1+L2 features) and Random Forest (all features). Establish the statistical lower bound for all future models.

## Files to Create

- `src/models/__init__.py`
- `src/models/baselines.py`
- `src/models/evaluate.py`
- `src/models/train.py`
- `tests/test_models.py`

## Implementation Steps

### 1. Data Preparation

```python
def prepare_training_data(
    feature_path: str = "data/processed/feature_matrix.parquet",
    label_path: str = "data/processed/labels.npy",
    config_path: str = "data/processed/split_config.yaml",
    test_months: float = 2.0
) -> Dict:
    """
    Load features and labels, perform temporal train/test split.

    CRITICAL: Use temporal split, NOT random split.
    The last `test_months` of data become the test set.

    Returns:
        {
            'X_train', 'X_test',
            'y_train', 'y_test',
            'feature_names': List[str],
            'train_indices', 'test_indices',
            'class_distribution_train': Dict,
            'class_distribution_test': Dict
        }
    """
    ...
```

**Edge cases**:
- Test set has no positive examples → log warning but proceed (evaluation will show TSS=0)
- Extreme class imbalance in train → use class_weight in models
- Fewer than 10 positive examples → flag for human review

### 2. Baseline: Logistic Regression

```python
def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str = 'balanced',
    max_iter: int = 1000
) -> LogisticRegression:
    """
    Train Logistic Regression on L1+L2 features only.

    PRD: "Establishes statistical lower bound"
    """
    ...
```

**Implementation details**:
- Use `solver='lbfgs'` (good for small datasets)
- `multi_class='multinomial'` for severity prediction
- Standardize features with `StandardScaler` before training
- For binary (flare vs no flare), use `class_weight='balanced'`

### 3. Baseline: Random Forest

```python
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 15,
    class_weight: str = 'balanced'
) -> RandomForestClassifier:
    """
    Train Random Forest on ALL feature layers.

    PRD: "Strong classical baseline, interpretable feature importance"
    """
    ...
```

**Implementation details**:
- `n_jobs=-1` for parallel training
- Use `oob_score=True` for out-of-bag evaluation
- Set `random_state=42` for reproducibility
- Store feature importance for later explainability

### 4. Evaluation

```python
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    compute_ci: bool = True
) -> Dict:
    """
    Compute all required metrics per PRD §4.

    Returns:
        {
            'model_name': str,
            'tss': float,                    # True Skill Statistic
            'brier_score': float,            # Probability Calibration
            'brier_skill_score': float,
            'mae_lead_time': float,          # For future use
            'false_alarm_rate': float,
            'detection_rate': float,         # Recall for M+X
            'auc_roc': float,
            'confusion_matrix': np.ndarray,
            'classification_report': str,
            'feature_importance': Optional[Dict],
            'calibration_curve': Dict        # (prob_true, prob_pred)
        }
    """
    ...

def print_metrics_table(metrics: Dict) -> None:
    """Pretty-print evaluation metrics as a table."""
    ...

def plot_calibration_curve(metrics: Dict, save_path: str = None) -> None:
    """Plot calibration curve (reliability diagram)."""
    ...

def plot_confusion_matrix(metrics: Dict, save_path: str = None) -> None:
    """Plot normalized confusion matrix."""
    ...
```

**Metric implementations**:
- **TSS** = True Positive Rate - False Positive Rate (aka Youden's J statistic)
- **Brier Score** = mean((p - y)²) — lower is better
- **False Alarm Rate** = FP / (FP + TN)
- **Detection Rate** = TP / (TP + FN) for M+X class specifically

### 5. Training Script

```python
# src/models/train.py
"""
Entry point: train and evaluate all baseline models.

Usage: python -m src.models.train

Saves:
- models/logistic_regression.pkl
- models/random_forest.pkl
- models/evaluation_results.json
- models/feature_importance.csv
"""
...
```

### 6. MLflow Integration

Log all training runs to MLflow:
- Parameters: model type, feature layers, hyperparameters
- Metrics: TSS, Brier Score, FAR, Detection Rate, AUC-ROC
- Artifacts: model weights, confusion matrix plot, calibration curve
- Tags: baseline, <model_name>, experiment version

### 7. Write Tests

| Test | Description |
|------|-------------|
| `test_temporal_split` | Test set timestamps are all after train set |
| `test_logistic_regression_trains` | LR trains without error on valid data |
| `test_random_forest_trains` | RF trains without error on valid data |
| `test_tss_range` | TSS is in [-1, 1] |
| `test_brier_score_range` | Brier score is in [0, 1] |
| `test_model_save_load` | Saved model can be loaded and predicts |
| `test_class_distribution_preserved` | Split preserves class distribution info |
| `test_majority_class_baseline` | Always-predict-quiet gives TSS=0 |

## Acceptance Criteria

- [ ] Both LR and RF train without errors
- [ ] Temporal split: test data is strictly later than training data
- [ ] TSS > 0.3 for both models (reasonable for synthetic data)
- [ ] Brier Score < 0.15 for both models
- [ ] Feature importance extracted from RF (top 10 features)
- [ ] Models saved as `.pkl` files in `models/` directory
- [ ] Evaluation results saved as JSON
- [ ] MLflow run logged with all metrics
- [ ] All tests pass: `pytest tests/test_models.py -v`

## Verification

```bash
python -m src.models.train
echo "---"
python -c "
import json
results = json.load(open('models/evaluation_results.json'))
for model, metrics in results.items():
    print(f'{model}: TSS={metrics[\"tss\"]:.4f}, Brier={metrics[\"brier_score\"]:.4f}, FAR={metrics[\"false_alarm_rate\"]:.4f}')
"
```
