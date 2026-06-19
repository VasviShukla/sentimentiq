from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from model import load_model, predict_sentiment

app = FastAPI(
    title="SentimentIQ API",
    description="Real-time sentiment analysis powered by Machine Learning",
    version="1.0.0"
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
try:
    model, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError as e:
    print(f"Warning: {e}")
    model_loaded = False

# ── Request schemas ──────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "SentimentIQ API is running",
        "model_loaded": model_loaded
    }

@app.post("/predict")
def predict(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first using the Jupyter notebook.")
    result = predict_sentiment(req.text, model, vectorizer)
    return result

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first using the Jupyter notebook.")
    results = [predict_sentiment(t, model, vectorizer) for t in req.texts if t.strip()]
    return {"count": len(results), "results": results}
