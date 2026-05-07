from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Các biến cấu hình từ môi trường
GCS_BUCKET = os.environ.get("GCS_BUCKET", "justlab-mlops-data")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        
        # Chỉ tải nếu model tồn tại trên GCS
        if blob.exists():
            blob.download_to_filename(MODEL_PATH)
            print(f"✅ Đã tải model từ gs://{GCS_BUCKET}/{GCS_MODEL_KEY} về {MODEL_PATH}")
        else:
            print(f"⚠️ Model chưa tồn tại trên GCS tại {GCS_MODEL_KEY}. Vui lòng chạy pipeline trước.")
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")

# Tải model khi khởi động ứng dụng
download_model()

# Load model vào bộ nhớ (nếu file tồn tại)
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Kiểm tra trạng thái server."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    """Đầu vào: JSON {'features': [f1, ..., f12]}"""
    global model
    
    # Reload model nếu trước đó chưa load được
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model is not available yet.")

    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thấp", 1: "trung bình", 2: "cao"}
    
    return {
        "prediction": prediction,
        "label": labels.get(prediction, "không xác định")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
