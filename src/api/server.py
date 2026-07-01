from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import json
import os

from src.api.schemas import PredictionInput, PredictionOutput
from src.api.inference import InferenceEngine
from src.api.alerting import AlertManager

inference_engine: InferenceEngine = None
alert_manager: AlertManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, alert_manager
    try:
        inference_engine = InferenceEngine()
        print("Inference engine loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load inference engine: {e}")
        inference_engine = None
    alert_manager = AlertManager()
    yield


app = FastAPI(
    title="FlareClassifier — Solar Flare Forecasting API",
    version="0.1.0",
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
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if inference_engine is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = inference_engine.predict(input_data)
    alert_manager.evaluate_and_alert(result)
    return result


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                input_data = PredictionInput(**msg)
                if inference_engine is not None:
                    result = inference_engine.predict(input_data)
                    await websocket.send_text(result.model_dump_json())
                else:
                    await websocket.send_text(json.dumps({"error": "Model not loaded"}))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/metrics")
async def get_metrics():
    import json as j
    try:
        with open('models/evaluation_results.json') as f:
            metrics = j.load(f)
        return {"status": "ok", "metrics": metrics}
    except FileNotFoundError:
        return {"status": "no_metrics", "message": "Evaluation results not found"}
