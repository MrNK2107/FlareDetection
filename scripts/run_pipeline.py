import argparse
import yaml
from pathlib import Path


def run_pipeline(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("FlareClassifier Pipeline v0.1.0")
    print("=" * 60)

    print("\n[1/5] Generating synthetic data...")
    from src.data_generation.synthetic_flare_generator import SyntheticFlareGenerator
    gen = SyntheticFlareGenerator(config_path)
    df = gen.generate_dataset()
    print(f"  -> {len(df):,} rows generated")

    flare_dist = df[df.flare_class != 'None'].flare_class.value_counts().to_dict()
    print(f"  -> Flare distribution: {flare_dist}")

    print("\n[2/5] Running ingestion pipeline...")
    from src.ingestion.pipeline import run_ingestion_pipeline
    stats = run_ingestion_pipeline(config_path=config_path)
    print(f"  -> {stats['n_windows']:,} windows created")
    print(f"  -> Missing data rate: {stats['missing_data_rate']:.4%}")

    print("\n[3/5] Extracting features...")
    import numpy as np
    import pandas as pd
    windows_dir = config['data']['windows_dir']
    processed_dir = config['data']['processed_dir']
    X_soft = np.load(str(Path(windows_dir) / 'X_soft.npy'))
    X_hard = np.load(str(Path(windows_dir) / 'X_hard.npy'))
    from src.features.pipeline import extract_all_features
    features = extract_all_features(X_soft, X_hard, config_path=config_path)
    print(f"  -> {features.shape[1]} features for {features.shape[0]} windows")
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    features.to_parquet(str(Path(processed_dir) / 'feature_matrix.parquet'))
    print(f"  -> Saved to {Path(processed_dir)/'feature_matrix.parquet'}")

    print("\n[4/5] Computing physics features...")
    dsoft_dt_all = np.gradient(X_soft, axis=1)
    dhard_dt_all = np.gradient(X_hard, axis=1)
    d2hard_dt2_all = np.gradient(dhard_dt_all, axis=1)
    from src.physics.pipeline import compute_physics_features
    physics_features = compute_physics_features(
        X_soft, X_hard,
        dsoft_dt_all=dsoft_dt_all,
        d2hard_dt2_all=d2hard_dt2_all,
        config=config,
    )
    print(f"  -> {physics_features.shape[1]} physics features added")
    full_features = pd.concat([features.reset_index(drop=True), physics_features.reset_index(drop=True)], axis=1)
    full_features.to_parquet(str(Path(processed_dir) / 'full_feature_matrix.parquet'))

    print("\n[5/5] Training baseline models...")
    from src.models.train import train_all_baselines
    results = train_all_baselines(config_path=config_path)
    for model_name, metrics in results.items():
        print(f"  -> {model_name}: TSS={metrics['tss']:.4f}, "
              f"Brier={metrics['brier_score']:.4f}, "
              f"FAR={metrics['false_alarm_rate']:.4f}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlareClassifier Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
