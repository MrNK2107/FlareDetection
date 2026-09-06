"""Train the 6-state HMM solar state machine. Usage: python scripts/train_hmm.py [config]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.state_machine import train_state_machine

if __name__ == "__main__":
    train_state_machine(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
