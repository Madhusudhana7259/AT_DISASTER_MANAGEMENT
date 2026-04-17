from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.app.engine import SurveillanceEngine


app = FastAPI(title="Crowd Safety Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: SurveillanceEngine | None = None


@app.on_event("startup")
def startup_event():
    global engine
    engine = SurveillanceEngine(
        video_path="data/raw_videos/video4.mp4",
        model_path="models/abnormal_cnn_lstm1.pth",
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def api_status():
    if engine is None:
        return {"error": "Engine not initialized"}
    return engine.get_latest_status()


@app.get("/api/snapshot")
def api_snapshot():
    if engine is None:
        return {"error": "Engine not initialized"}
    return engine.get_snapshot_base64()


@app.get("/api/stream")
def api_stream():
    if engine is None:
        return {"error": "Engine not initialized"}
    return StreamingResponse(
        engine.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

