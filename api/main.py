import io
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurovision-api")

# Initialize FastAPI application
app = FastAPI(
    title="NeuroVision AI - Brain Tumor Classification API",
    description="API for brain MRI classification using InceptionV3 (Glioma, Meningioma, No Tumor, Pituitary).",
    version="1.0.0",
)

# Enable CORS for frontend integration (Streamlit, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CLS_DIR = BASE_DIR / "models" / "classification"

# Default class mapping
DEFAULT_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Global state
classification_model = None
classification_status = "Not initialized"
class_names = DEFAULT_CLASSES



class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_status: str
    classes: List[str]


class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float
    is_tumor: bool
    probabilities: Dict[str, float]



@app.on_event("startup")
def load_models():
    """Load class labels and InceptionV3 model on startup."""
    global classification_model, classification_status, class_names

    # Load Class Names
    class_file = MODEL_CLS_DIR / "class_names.json"
    if class_file.exists():
        try:
            with open(class_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "classes" in data:
                        class_names = data["classes"]
                    else:
                        class_names = [data[str(i)] for i in range(len(data))]
            logger.info(f"Loaded class names: {class_names}")
        except Exception as e:
            logger.warning(f"Could not parse class_names.json ({e}), using default classes.")

    # Load InceptionV3 Model
    candidates = [
        MODEL_CLS_DIR / "best_model.keras",
        MODEL_CLS_DIR / "best_inceptionv3.keras",
    ]
    model_file = next((p for p in candidates if p.exists()), None)

    if not model_file:
        classification_status = f"Model file not found in {MODEL_CLS_DIR} (expected best_model.keras)"
        logger.warning(classification_status)
    else:
        try:
            import tensorflow as tf
            classification_model = tf.keras.models.load_model(str(model_file))
            classification_status = f"Loaded successfully from {model_file.name}"
            logger.info(f"✓ {classification_status}")
        except ImportError:
            classification_status = "TensorFlow is not installed (run: pip install tensorflow)"
            logger.warning(classification_status)
        except Exception as e:
            classification_status = f"Failed to load model file: {e}"
            logger.error(classification_status)



# ================= Preprocessing Helper ===============
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((299, 299))
    img_array = np.array(image, dtype=np.float32)

    # InceptionV3 scaling: [0, 255] -> [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 299, 299, 3)


# ================= API Endpoints ==================

@app.get("/", tags=["General"])
def root():
    
    return {
        "service": "NeuroVision AI - Brain Tumor Classification API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    
    return HealthResponse(
        status="healthy",
        model_loaded=classification_model is not None,
        model_status=classification_status,
        classes=class_names,
    )


@app.post("/predict", response_model=ClassificationResult, tags=["Classification"])
@app.post("/predict/classification", response_model=ClassificationResult, include_in_schema=False)
async def predict_mri(file: UploadFile = File(...)):
    
    if classification_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Classification model not loaded: {classification_status}",
        )

    try:
        contents = await file.read()
        tensor = preprocess_image(contents)

        # Run inference
        raw_preds = classification_model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(raw_preds))
        pred_class = class_names[pred_idx]
        confidence = float(raw_preds[pred_idx])

        # All class probabilities
        probabilities = {
            class_names[i]: round(float(raw_preds[i]), 4)
            for i in range(len(class_names))
        }
        is_tumor = pred_class.lower() != "no tumor"

        return ClassificationResult(
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            is_tumor=is_tumor,
            probabilities=probabilities,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
