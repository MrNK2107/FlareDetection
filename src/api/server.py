"""FlareClassifier API server: /predict, WebSocket stream (telemetry +
prediction + alert push), history endpoints (docs/08 §4), replay control, and
optional shadow model (docs/08 §5.2)."""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.api.schemas import PredictionInput, PredictionOutput
from src.api.inference import InferenceEngine
from src.api.alerting import AlertManager, BrowserPushChannel
from src.api.history import InferenceHistory

inference_engine: InferenceEngine = None
shadow_engine: InferenceEngine = None
alert_manager: AlertManager = None
history: InferenceHistory = None
streamer = None
broadcast_task = None
connected_ws: Set[WebSocket] = set()
TICK_SECONDS = 10.0


def _broadcast(message_json: str) -> None:
    for ws in list(connected_ws):
        try:
            asyncio.get_running_loop().create_task(ws.send_text(message_json))
        except Exception:
            pass


def _broadcast_alert(payload_json: dict) -> None:
    """Sync callback invoked by AlertManager; pushes to all WS clients."""
    _broadcast(json.dumps({"type": "alert", "payload": payload_json}))


def _run_tick() -> None:
    """One replay step: real features -> prediction -> history/alerts/broadcast."""
    if streamer is None or inference_engine is None:
        return
    payload = streamer.step()
    if payload is None:
        return
    points = payload.pop('points')
    _broadcast(json.dumps({
        "type": "telemetry",
        "window_timestamp_utc": payload['window_timestamp_utc'],
        "points": points,
    }))
    try:
        input_data = PredictionInput(**payload)
        result = inference_engine.predict(input_data)
        try:
            history.insert_prediction(result)
        except Exception:
            pass
        if shadow_engine is not None:
            try:
                shadow_result = shadow_engine.predict(input_data)
                shadow_result.model_version = "shadow"
                shadow_result.inference_timestamp_utc = result.inference_timestamp_utc
                history.insert_prediction(shadow_result)
            except Exception:
                pass
        alert_manager.evaluate_and_alert(result)
        # typed prediction message (includes attention weights + model version)
        _broadcast(json.dumps({
            "type": "prediction",
            **json.loads(result.model_dump_json()),
        }))
    except Exception as e:
        _broadcast(json.dumps({"type": "error", "error": str(e)}))


async def _broadcast_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(_run_tick)
        except Exception as e:
            print(f"Tick failed: {e}")
        await asyncio.sleep(TICK_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, shadow_engine, alert_manager, history, streamer, broadcast_task
    history = InferenceHistory()
    try:
        inference_engine = InferenceEngine()
        print("Inference engine loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load inference engine: {e}")
        inference_engine = None
    # Shadow deployment (docs/08 §5.2): set SHADOW_MODEL to an RF pickle path;
    # its predictions are computed and stored but never served.
    shadow_path = os.getenv("SHADOW_MODEL")
    if shadow_path:
        try:
            shadow_engine = InferenceEngine(rf_model_path=shadow_path)
            print(f"Shadow model loaded from {shadow_path}")
        except Exception as e:
            print(f"Warning: could not load shadow model: {e}")
            shadow_engine = None
    alert_manager = AlertManager(history_callback=lambda p: history.insert_alert(p))
    for ch in alert_manager.channels:
        if isinstance(ch, BrowserPushChannel):
            ch.register_broadcaster(_broadcast_alert)
    try:
        from src.api.streaming import TelemetryStreamer
        streamer = TelemetryStreamer()
        print("Telemetry streamer loaded (replay mode)")
    except Exception as e:
        print(f"Warning: telemetry streamer unavailable: {e}")
        streamer = None
    if streamer is not None:
        broadcast_task = asyncio.create_task(_broadcast_loop())
    yield
    if broadcast_task is not None:
        broadcast_task.cancel()
    connected_ws.clear()


app = FastAPI(
    title="FlareClassifier — Solar Flare Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": inference_engine is not None,
        "shadow_loaded": shadow_engine is not None,
        "streamer_loaded": streamer is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = inference_engine.predict(input_data)
    try:
        history.insert_prediction(result)
    except Exception as e:
        print(f"History write failed: {e}")
    if shadow_engine is not None:
        try:
            shadow_result = shadow_engine.predict(input_data)
            shadow_result.model_version = "shadow"
            shadow_result.inference_timestamp_utc = result.inference_timestamp_utc
            history.insert_prediction(shadow_result)
        except Exception as e:
            print(f"Shadow prediction failed: {e}")
    alert_manager.evaluate_and_alert(result)
    return result


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    connected_ws.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type")
                if msg_type == "control":
                    action = msg.get("action")
                    if streamer is not None:
                        if action == "pause":
                            streamer.paused = True
                        elif action == "resume":
                            streamer.paused = False
                        elif action == "set_speed":
                            streamer.speed = float(msg.get("value", 1.0))
                        await websocket.send_text(json.dumps({
                            "type": "control_ack", "action": action,
                            "speed": streamer.speed, "paused": streamer.paused,
                        }))
                elif "features" in msg:
                    # client-initiated inference (kept for compatibility)
                    input_data = PredictionInput(**msg)
                    if inference_engine is not None:
                        result = inference_engine.predict(input_data)
                        try:
                            history.insert_prediction(result)
                        except Exception:
                            pass
                        await websocket.send_text(result.model_dump_json())
                        alert_manager.evaluate_and_alert(result)
                    else:
                        await websocket.send_text(json.dumps({"error": "Model not loaded"}))
                else:
                    await websocket.send_text(json.dumps({"error": "Unknown message type"}))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        connected_ws.discard(websocket)


@app.post("/stream/control")
async def stream_control(action: str, value: float = 1.0):
    if streamer is None:
        raise HTTPException(status_code=503, detail="Streamer not loaded")
    if action == "pause":
        streamer.paused = True
    elif action == "resume":
        streamer.paused = False
    elif action == "set_speed":
        streamer.speed = float(value)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    return {"status": "ok", "action": action, "speed": streamer.speed, "paused": streamer.paused}


@app.get("/history/predictions")
async def history_predictions(start: str = None, end: str = None, format: str = "json"):
    if format == "csv":
        csv_text = history.predictions_csv(start, end)
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
    df = history.query_predictions(start, end)
    return {"status": "ok", "count": len(df), "predictions": df.to_dict(orient="records")}


@app.get("/history/alerts")
async def history_alerts(start: str = None, end: str = None):
    df = history.query_alerts(start, end)
    return {"status": "ok", "count": len(df), "alerts": df.to_dict(orient="records")}


@app.get("/history/events")
async def history_events(start: str = None, end: str = None):
    """Flare events for the hindcast overlay (from the generator catalogue)."""
    catalogue_path = "data/external/flare_catalogue.parquet"
    try:
        import pandas as pd
        cat = pd.read_parquet(catalogue_path)
        if start:
            cat = cat[cat['peak_utc'] >= start]
        if end:
            cat = cat[cat['peak_utc'] <= end]
        return {"status": "ok", "count": len(cat), "events": cat.to_dict(orient="records")}
    except Exception as e:
        return {"status": "no_events", "message": str(e)}


@app.get("/metrics")
async def get_metrics():
    try:
        with open('models/evaluation_results.json') as f:
            metrics = json.load(f)
        return {"status": "ok", "metrics": metrics}
    except FileNotFoundError:
        return {"status": "no_metrics", "message": "Evaluation results not found"}
