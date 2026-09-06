"""Train the Dual-Stream Transformer. Usage: python scripts/train_transformer.py [config]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.transformer import train_transformer

if __name__ == "__main__":
    train_transformer(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
