"""Model registry (docs/08 §5.2, docs/09): immutable timestamped version dirs,
metadata + metrics per version, production pointer, and the deployment gate
(new model must beat production on ALL 5 metrics before promotion)."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# metric name -> direction ('higher' or 'lower')
METRIC_DIRECTIONS = {
    'tss': 'higher',
    'brier_score': 'lower',
    'false_alarm_rate': 'lower',
    'detection_rate': 'higher',
    'lead_time_mae_min': 'lower',
}
GATE_METRICS = list(METRIC_DIRECTIONS.keys())


class ModelRegistry:
    def __init__(
        self,
        registry_dir: str = "models/registry",
        production_path: str = "models/production.json",
    ):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.production_path = Path(production_path)

    def register(
        self,
        model_name: str,
        metrics: Dict,
        artifacts: Dict[str, str],
        metadata: Optional[Dict] = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        version = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z-{model_name}"
        vdir = self.registry_dir / version
        vdir.mkdir(parents=True, exist_ok=False)
        for src in artifacts.values():
            shutil.copy2(src, vdir / Path(src).name)
        meta = {
            'version': version,
            'model_name': model_name,
            'registered_at': datetime.now(timezone.utc).isoformat(),
            'metrics': metrics,
            **(metadata or {}),
        }
        with open(vdir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return version

    def list_models(self) -> List[Dict]:
        out = []
        for vdir in sorted(self.registry_dir.iterdir()):
            meta_path = vdir / 'metadata.json'
            if meta_path.exists():
                with open(meta_path) as f:
                    out.append(json.load(f))
        return out

    def get_version(self, version: str) -> Optional[Dict]:
        meta_path = self.registry_dir / version / 'metadata.json'
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)

    def get_production(self) -> Optional[Dict]:
        if not self.production_path.exists():
            return None
        with open(self.production_path) as f:
            return json.load(f)

    def promote(self, version: str) -> Dict:
        meta = self.get_version(version)
        if meta is None:
            raise ValueError(f"Unknown version: {version}")
        with open(self.production_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return meta

    @staticmethod
    def promotion_check(candidate_metrics: Dict, production_metrics: Dict) -> tuple:
        """Deployment gate: candidate must outperform production on ALL 5
        metrics (docs/08 §5.2 Rule 1)."""
        report = {}
        passed = True
        for metric, direction in METRIC_DIRECTIONS.items():
            c = candidate_metrics.get(metric)
            p = production_metrics.get(metric)
            if c is None or p is None:
                report[metric] = {'candidate': c, 'production': p, 'pass': False, 'reason': 'missing'}
                passed = False
                continue
            ok = (c >= p) if direction == 'higher' else (c <= p)
            report[metric] = {'candidate': float(c), 'production': float(p), 'pass': bool(ok)}
            passed = passed and ok
        return passed, report


def load_registered_model(version: str, registry_dir: str = "models/registry"):
    """Load the primary model artifact from a registry version (RF pickle)."""
    import joblib
    vdir = Path(registry_dir) / version
    meta_path = vdir / 'metadata.json'
    if not meta_path.exists():
        raise ValueError(f"Unknown version: {version}")
    with open(meta_path) as f:
        meta = json.load(f)
    artifact = meta.get('artifacts', {}).get('model', 'random_forest.pkl')
    return joblib.load(vdir / Path(artifact).name)
