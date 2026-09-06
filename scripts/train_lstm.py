"""Train the LSTM baseline. Usage: python scripts/train_lstm.py [config]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lstm import train_lstm

if __name__ == "__main__":
    train_lstm(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
