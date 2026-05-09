from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.app.engine import SurveillanceEngine
from backend.app.notifier import AlertReportNotifier, NotificationConfigError


app = FastAPI(title="Crowd Safety Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: SurveillanceEngine | None = None
notifier = AlertReportNotifier()


@app.on_event("startup")
def startup_event():
    global engine
    engine = SurveillanceEngine(
        video_path="data/raw_videos/video5.mp4",
        model_path="models/abnormal_cnn_lstm1.pth",
    )
    engine.start()


@app.on_event("shutdown")
def shutdown_event():
    global engine
    if engine is not None:
        engine.stop()
        engine = None


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


@app.post("/api/report/send")
def api_send_report():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    status = engine.get_latest_status()
    try:
        result = notifier.send_email_report(
            status=status,
        )
    except NotificationConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to send report: {exc}") from exc

    return {
        "ok": True,
        "message": f"Report sent via email to {result['recipient']}",
        "subject": result["subject"],
    }
