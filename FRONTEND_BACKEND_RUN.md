## Run Backend (FastAPI)

From project root (`E:\Project`):

```powershell
pip install -r requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend endpoints:

- `GET /health`
- `GET /api/status` -> latest agent levels, disaster type, alerts
- `GET /api/stream` -> MJPEG live stream
- `GET /api/snapshot` -> one frame as base64 + latest status

## Run Frontend (React)

```powershell
cd Frontend
npm install
npm run dev
```

Open:

- `http://localhost:5173`

Notes:

- Frontend reads backend from `http://127.0.0.1:8000`.
- Backend currently uses `data/raw_videos/sample1.mp4` as input stream.
