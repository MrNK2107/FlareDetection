import argparse
import json
import time
from pathlib import Path

import yaml


def run_pipeline(config_path: str = "config/config.yaml", skip_deep: bool = False):
    t0 = time.time()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("FlareClassifier Pipeline v1.0.0")
    print("=" * 60)

    print("\n[1/8] Generating synthetic data...")
    from src.data_generation.synthetic_flare_generator import SyntheticFlareGenerator
    gen = SyntheticFlareGenerator(config_path)
    df_labels = gen.generate_dataset()
    n_days = config['synthetic_data']['total_days']
    print(f"  -> {n_days} days generated; flare rows: {int((df_labels.flare_class != 'None').sum()):,}")
    cat = gen.get_flare_catalogue()
    print(f"  -> Catalogue: {len(cat)} events "
          f"({cat.flare_class.value_counts().to_dict() if len(cat) else ''})")

    print("\n[2/8] Running ingestion pipeline...")
    from src.ingestion.pipeline import run_ingestion_pipeline
    stats = run_ingestion_pipeline(config_path=config_path)
    print(f"  -> {stats['n_windows']:,} candidate windows, "
          f"{stats['n_selected_windows']:,} selected for features")
    print(f"  -> {stats['n_dl_windows']:,} DL windows; "
          f"class distribution: {stats['class_distribution']}")
    print(f"  -> Missing data rate: {stats['missing_data_rate']:.4%}")

    print("\n[3/8] Training baselines (Logistic Regression + Random Forest)...")
    from src.models.train import train_all_baselines
    results = train_all_baselines(config_path=config_path)
    for model_name, metrics in results.items():
        print(f"  -> {model_name}: TSS={metrics['tss']:.4f}, "
              f"Brier={metrics['brier_score']:.4f}")

    if not skip_deep:
        print("\n[4/8] Training LSTM baseline...")
        from src.models.lstm import train_lstm
        lstm_results = train_lstm(config_path=config_path)
        print(f"  -> LSTM: TSS={lstm_results['tss']:.4f}, Brier={lstm_results['brier_score']:.4f}")

        print("\n[5/8] Training Dual-Stream Transformer...")
        from src.models.transformer import train_transformer
        tr_results = train_transformer(config_path=config_path)
        print(f"  -> Transformer: TSS={tr_results['tss']:.4f}, "
              f"Brier={tr_results['brier_score']:.4f}")
    else:
        print("\n[4-5/8] Skipping deep models (--skip-deep)")

    print("\n[6/8] Training Solar State Machine (HMM)...")
    from src.models.state_machine import train_state_machine
    hmm_val = train_state_machine(config_path=config_path)
    print(f"  -> S3/S4 co-occurrence: {hmm_val['s3_s4_cooccurrence_with_flares']:.2%}")

    print("\n[7/8] Training lead-time model...")
    from src.models.lead_time import train_lead_time
    lead_metrics = train_lead_time(config_path=config_path)
    if lead_metrics:
        print(f"  -> Lead-time MAE: {lead_metrics.get('lead_time_mae_min', float('nan')):.1f} min")

    print("\n[8/8] Registering + promoting models...")
    from src.models.registry import ModelRegistry
    with open('models/evaluation_results.json') as f:
        all_results = json.load(f)
    registry = ModelRegistry()
    lead_mae = lead_metrics.get('lead_time_mae_min') if lead_metrics else None
    for model_name, artifact in [
        ('RandomForest', 'models/random_forest.pkl'),
        ('LogisticRegression', 'models/logistic_regression.pkl'),
    ] + ([] if skip_deep else [('Transformer', 'models/transformer.pt'), ('LSTM', 'models/lstm.pt')]):
        metrics = dict(all_results.get(model_name, {}))
        metrics.setdefault('lead_time_mae_min', lead_mae)
        version = registry.register(
            model_name=model_name,
            metrics=metrics,
            artifacts={Path(artifact).name: artifact},
            metadata={'artifacts': {Path(artifact).name: artifact}},
        )
        print(f"  -> registered {version}")
    production = registry.get_production()
    if production is None:
        best = max(
            (r for r in registry.list_models() if r['metrics'].get('tss') is not None),
            key=lambda r: r['metrics'].get('tss', 0), default=None,
        )
        if best:
            registry.promote(best['version'])
            print(f"  -> promoted first production model: {best['version']}")
    else:
        print(f"  -> production remains: {production['version']} "
              f"(use scripts/promote_model.py for the gated promotion)")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed / 60:.1f} min")
    print("=" * 60)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlareClassifier Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-deep", action="store_true",
                        help="Skip LSTM + Transformer training")
    args = parser.parse_args()
    run_pipeline(args.config, skip_deep=args.skip_deep)
