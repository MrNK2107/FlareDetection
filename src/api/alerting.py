from datetime import datetime, timedelta
from typing import Optional

from src.api.schemas import PredictionOutput, AlertPayload


class AlertManager:
    def __init__(self, threshold: float = 0.5, quiet_window_minutes: float = 15.0):
        self.threshold = threshold
        self.quiet_window = timedelta(minutes=quiet_window_minutes)
        self.last_alert_time: Optional[datetime] = None
        self.was_above_threshold = False

    def should_alert(self, probability: float, timestamp: datetime) -> bool:
        now_above = probability >= self.threshold
        rising_edge = now_above and not self.was_above_threshold
        self.was_above_threshold = now_above
        if not rising_edge:
            return False
        if self.last_alert_time is not None:
            if timestamp - self.last_alert_time < self.quiet_window:
                return False
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
            print(f"ALERT: Flare probability {payload.flare_probability:.2%}")
            print(f"  Lead time: {payload.expected_lead_time_min} min")
            print(f"  Explanation: {payload.explanation_text[:100]}...")
            return payload
        return None
