"""Inference history (docs/08 §4): SQLite persistence for predictions and
alerts, powering the hindcast overlay and CSV export endpoints."""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.api.schemas import AlertPayload, PredictionOutput

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    flare_probability REAL NOT NULL,
    severity_json TEXT,
    expected_lead_time_min REAL,
    lead_time_ci_json TEXT,
    solar_state TEXT,
    model_uncertainty REAL,
    model_version TEXT,
    explanation_text TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    flare_probability REAL NOT NULL,
    expected_lead_time_min REAL,
    explanation_text TEXT,
    deeplink TEXT,
    channels TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts_utc);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts_utc);
"""


class InferenceHistory:
    def __init__(self, db_path: str = "data/processed/inference_history.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def insert_prediction(self, output: PredictionOutput) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO predictions (ts_utc, flare_probability, severity_json, "
                "expected_lead_time_min, lead_time_ci_json, solar_state, model_uncertainty, "
                "model_version, explanation_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    output.inference_timestamp_utc.isoformat(),
                    float(output.flare_probability),
                    json.dumps(output.severity_probs),
                    output.expected_lead_time_min,
                    json.dumps(output.lead_time_ci_90) if output.lead_time_ci_90 else None,
                    output.solar_state,
                    float(output.model_uncertainty),
                    output.model_version,
                    output.explanation_text,
                ),
            )
            self.conn.commit()

    def insert_alert(self, payload: AlertPayload, channels: Optional[List[str]] = None) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO alerts (ts_utc, flare_probability, expected_lead_time_min, "
                "explanation_text, deeplink, channels) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    payload.timestamp_utc.isoformat(),
                    float(payload.flare_probability),
                    payload.expected_lead_time_min,
                    payload.explanation_text,
                    payload.deeplink,
                    json.dumps(channels or []),
                ),
            )
            self.conn.commit()

    def query_predictions(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM predictions"
        conditions, params = [], []
        if start:
            conditions.append("ts_utc >= ?")
            params.append(start)
        if end:
            conditions.append("ts_utc <= ?")
            params.append(end)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY ts_utc"
        return pd.read_sql_query(query, self.conn, params=params)

    def query_alerts(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM alerts"
        conditions, params = [], []
        if start:
            conditions.append("ts_utc >= ?")
            params.append(start)
        if end:
            conditions.append("ts_utc <= ?")
            params.append(end)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY ts_utc"
        return pd.read_sql_query(query, self.conn, params=params)

    def predictions_csv(self, start: Optional[str] = None, end: Optional[str] = None) -> str:
        df = self.query_predictions(start, end)
        return df.to_csv(index=False)

    def count_since(self, since_iso: str) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE ts_utc >= ?", (since_iso,)
        )
        return int(cur.fetchone()[0])
