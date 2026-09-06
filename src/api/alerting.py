"""Alert system (docs/08 §3): rising-edge trigger at 0.5, configurable quiet
window (default 15 min), and channels: email (SMTP), webhook (JSON POST),
browser (WebSocket push via registered callback)."""
import json
import smtplib
import urllib.request
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Callable, List, Optional

import yaml

from src.api.schemas import PredictionOutput, AlertPayload


class AlertChannel:
    name = "base"

    def send(self, payload: AlertPayload) -> bool:
        raise NotImplementedError


class EmailChannel(AlertChannel):
    name = "email"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    def send(self, payload: AlertPayload) -> bool:
        c = self.cfg
        if not c.get('enabled') or not c.get('smtp_host') or not c.get('to_addrs'):
            return False
        try:
            msg = MIMEText(
                f"Flare probability {payload.flare_probability:.0%}\n"
                f"Lead time: {payload.expected_lead_time_min} min\n"
                f"{payload.explanation_text}\n{payload.deeplink}"
            )
            msg['Subject'] = f"[FlareClassifier] Alert {payload.flare_probability:.0%}"
            msg['From'] = c.get('from_addr', '')
            msg['To'] = ", ".join(c['to_addrs'])
            with smtplib.SMTP(c['smtp_host'], int(c.get('smtp_port', 587)), timeout=10) as s:
                if c.get('username'):
                    s.starttls()
                    s.login(c['username'], c.get('password', ''))
                s.send_message(msg)
            return True
        except Exception as e:
            print(f"Email alert failed: {e}")
            return False


class WebhookChannel(AlertChannel):
    name = "webhook"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    def send(self, payload: AlertPayload) -> bool:
        c = self.cfg
        if not c.get('enabled') or not c.get('url'):
            return False
        try:
            body = json.dumps(payload.model_dump(mode='json')).encode()
            req = urllib.request.Request(
                c['url'], data=body,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300
        except Exception as e:
            print(f"Webhook alert failed: {e}")
            return False


class BrowserPushChannel(AlertChannel):
    """Browser notifications via the WebSocket stream: the server registers a
    broadcaster callback; VAPID Web Push is a stretch goal (MVP_PLAN.md)."""
    name = "browser"

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        self.broadcaster: Optional[Callable[[dict], None]] = None

    def register_broadcaster(self, fn: Callable[[dict], None]) -> None:
        self.broadcaster = fn

    def send(self, payload: AlertPayload) -> bool:
        if not self.cfg.get('enabled', True) or self.broadcaster is None:
            return False
        try:
            self.broadcaster(payload.model_dump(mode='json'))
            return True
        except Exception as e:
            print(f"Browser alert failed: {e}")
            return False


class AlertManager:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        history_callback: Optional[Callable[[AlertPayload], None]] = None,
    ):
        self.threshold = 0.5
        self.quiet_window = timedelta(minutes=15)
        self.channels: List[AlertChannel] = []
        self.history_callback = history_callback
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f).get('alerting', {})
            self.threshold = float(cfg.get('threshold', 0.5))
            self.quiet_window = timedelta(minutes=float(cfg.get('quiet_window_minutes', 15)))
            ch = cfg.get('channels', {})
            self.channels = [
                EmailChannel(ch.get('email')),
                WebhookChannel(ch.get('webhook')),
                BrowserPushChannel(ch.get('browser')),
            ]
        except Exception:
            pass
        self.last_alert_time: Optional[datetime] = None
        self.was_above_threshold = False

    def should_alert(self, probability: float, timestamp: datetime) -> bool:
        now_above = probability >= self.threshold
        rising_edge = now_above and not self.was_above_threshold
        if not rising_edge:
            self.was_above_threshold = now_above
            return False
        if (
            self.last_alert_time is not None
            and timestamp - self.last_alert_time < self.quiet_window
        ):
            # Suppressed: stay armed so the alert fires once the quiet window
            # elapses if the condition persists (docs/08 §3.4).
            return False
        self.was_above_threshold = now_above
        self.last_alert_time = timestamp
        return True

    def evaluate_and_alert(self, prediction: PredictionOutput) -> Optional[AlertPayload]:
        now = prediction.inference_timestamp_utc or datetime.utcnow()
        if self.should_alert(prediction.flare_probability, now):
            self.last_alert_time = now
            payload = AlertPayload(
                flare_probability=prediction.flare_probability,
                severity_probs=prediction.severity_probs,
                expected_lead_time_min=prediction.expected_lead_time_min,
                explanation_text=prediction.explanation_text,
                deeplink=f"/dashboard?time={now.isoformat()}",
                timestamp_utc=now,
            )
            delivered = []
            for channel in self.channels:
                if channel.send(payload):
                    delivered.append(channel.name)
            if self.history_callback is not None:
                try:
                    self.history_callback(payload)
                except Exception as e:
                    print(f"Alert history write failed: {e}")
            print(f"ALERT: Flare probability {payload.flare_probability:.2%} "
                  f"via [{', '.join(delivered) or 'no channels'}]")
            return payload
        return None
