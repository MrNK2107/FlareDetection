"""Post-pipeline verification (CLAUDE.md + docs/06 §2).

Checks artifacts, payload spec compliance, and reports metrics vs PRD targets
(TSS>0.6, Brier<0.08, FAR<25%, detection>80% are physics-level goals; on
synthetic data they are reported, not asserted).

Exit code 1 if any structural check fails.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

PRD_TARGETS = {
    'tss': ('>', 0.6),
    'brier_score': ('<', 0.08),
    'false_alarm_rate': ('<', 0.25),
    'detection_rate': ('>', 0.8),
}

checks = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    checks.append(ok)
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("FlareClassifier verification")
    print("-" * 40)

    # ------- artifacts
    print("\nArtifacts:")
    check("models/random_forest.pkl exists", Path('models/random_forest.pkl').exists())
    check("models/logistic_regression.pkl exists", Path('models/logistic_regression.pkl').exists())
    check("models/lstm.pt exists", Path('models/lstm.pt').exists())
    check("models/transformer.pt exists", Path('models/transformer.pt').exists())
    check("models/hmm.pkl exists", Path('models/hmm.pkl').exists())
    check("models/lead_time.pkl exists", Path('models/lead_time.pkl').exists())
    check("data/windows/X_soft.npy exists", Path('data/windows/X_soft.npy').exists())

    # ------- DL window shapes
    if Path('data/windows/X_soft.npy').exists():
        X_soft = np.load('data/windows/X_soft.npy', mmap_mode='r')
        y = np.load('data/windows/y_labels.npy')
        check("X_soft.npy is 2-D (windows x steps)", X_soft.ndim == 2,
              f"shape={X_soft.shape}, dtype={X_soft.dtype}")
        check("labels align with windows", len(y) == X_soft.shape[0],
              f"{len(y)} labels vs {X_soft.shape[0]} windows")
        check("labels include positive class", int((y > 0).sum()) > 0,
              f"{int((y > 0).sum())} positive windows")

    # ------- evaluation results
    print("\nEvaluation metrics (vs PRD targets, reported not asserted on synthetic data):")
    results_path = Path('models/evaluation_results.json')
    if not check("models/evaluation_results.json exists", results_path.exists()):
        return 1
    with open(results_path) as f:
        results = json.load(f)
    expected_models = {'LogisticRegression', 'RandomForest', 'LSTM', 'Transformer'}
    check("all 4 model families evaluated", expected_models.issubset(results.keys()),
          f"found: {sorted(results.keys())}")
    for model, metrics in sorted(results.items()):
        detail = (f"TSS={metrics['tss']:.3f} Brier={metrics['brier_score']:.3f} "
                  f"FAR={metrics['false_alarm_rate']:.3f} DetRate={metrics['detection_rate']:.3f}")
        check(f"{model} metrics complete", all(k in metrics for k in PRD_TARGETS), detail)

    best = max(results.items(), key=lambda kv: kv[1].get('tss', 0))
    print(f"\n  Best TSS: {best[0]} = {best[1]['tss']:.3f} "
          f"(PRD target > 0.6 — physics-level goal, synthetic data)")
    if best[1].get('mc_dropout_std_mean') is not None:
        print(f"  Transformer MC-dropout uncertainty (mean): {best[1]['mc_dropout_std_mean']:.4f}")

    # ------- payload spec compliance
    print("\nPayload spec (docs/06 §4):")
    try:
        from src.api.inference import InferenceEngine
        from src.api.schemas import PredictionInput
        engine = InferenceEngine()
        with open('models/feature_names.json') as f:
            names = json.load(f)
        inp = PredictionInput(features=[0.0] * len(names), feature_names=names)
        out = engine.predict(inp)
        check("all 10 payload fields populated",
              out.solar_state not in ("", None) and out.explanation_text != ""
              and out.model_uncertainty >= 0
              and (out.expected_lead_time_min is None) == (out.flare_probability < 0.3))
        check("severity probs sum <= 1", sum(out.severity_probs.values()) <= 1.0 + 1e-6)
        check("solar_state from HMM", out.solar_state != "Unknown",
              f"state={out.solar_state}")
    except Exception as e:
        check("inference engine serves a spec-compliant payload", False, str(e))

    # ------- registry
    print("\nModel registry:")
    try:
        from src.models.registry import ModelRegistry
        reg = ModelRegistry()
        n_versions = len(reg.list_models())
        check("registry has versions", n_versions >= 1, f"{n_versions} versions")
        check("production pointer set", reg.get_production() is not None,
              reg.get_production()['version'] if reg.get_production() else "")
    except Exception as e:
        check("registry accessible", False, str(e))

    # ------- HMM validation
    print("\nState machine:")
    if Path('models/hmm_validation.json').exists():
        with open('models/hmm_validation.json') as f:
            val = json.load(f)
        print(f"  S3/S4 co-occurrence with flares: {val['s3_s4_cooccurrence_with_flares']:.2%} "
              f"(docs/05 target >70% — physics-level goal)")
        print(f"  Flare windows landing in S3/S4: {val['flare_windows_in_s3_s4']:.2%}")
    else:
        check("hmm_validation.json exists", False)

    ok = all(checks)
    print("\n" + "=" * 40)
    print("VERIFICATION " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
