"""Model promotion gate (docs/08 §5.2).

Registers a candidate model version and promotes it only if it beats the
current production model on ALL 5 metrics (TSS, Brier, FAR, detection rate,
lead-time MAE). Otherwise the candidate is recorded but rejected.

Usage:
  python scripts/promote_model.py --model-name RandomForest \
      --metrics models/evaluation_results.json --artifacts models/random_forest.pkl \
      [--metrics-key RandomForest] [--lead-time-mae X.X]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import ModelRegistry, GATE_METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--metrics', default='models/evaluation_results.json')
    parser.add_argument('--metrics-key', default=None,
                        help='Key inside the metrics JSON (defaults to model name)')
    parser.add_argument('--artifacts', nargs='+', required=True)
    parser.add_argument('--lead-time-mae', type=float, default=None)
    parser.add_argument('--force', action='store_true',
                        help='Promote without gate (first production model)')
    args = parser.parse_args()

    with open(args.metrics) as f:
        all_metrics = json.load(f)
    key = args.metrics_key or args.model_name
    metrics = dict(all_metrics.get(key, {}))
    if args.lead_time_mae is not None:
        metrics['lead_time_mae_min'] = args.lead_time_mae
    else:
        metrics.setdefault('lead_time_mae_min', None)

    registry = ModelRegistry()
    version = registry.register(
        model_name=args.model_name,
        metrics=metrics,
        artifacts={Path(a).name: a for a in args.artifacts},
        metadata={'artifacts': {Path(a).name: a for a in args.artifacts}},
    )
    print(f"Registered candidate version: {version}")

    production = registry.get_production()
    if args.force or production is None:
        registry.promote(version)
        print(f"Promoted {version} to production (no prior production model).")
        return

    passed, report = ModelRegistry.promotion_check(metrics, production.get('metrics', {}))
    print("\nDeployment gate (all 5 metrics must improve):")
    for metric, res in report.items():
        if 'reason' in res:
            print(f"  {metric}: MISSING ({res['reason']})")
        else:
            print(f"  {metric}: candidate={res['candidate']:.4f} "
                  f"production={res['production']:.4f} -> {'PASS' if res['pass'] else 'FAIL'}")
    if passed:
        registry.promote(version)
        print(f"\nPROMOTED: {version}")
    else:
        print(f"\nREJECTED: {version} does not outperform production on all metrics.")


if __name__ == "__main__":
    main()
