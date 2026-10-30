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
    title="NeuroVision AI - Brain Tumor Diagnostic API",
    description="API for Brain Tumor Classification (InceptionV3) and Tumor Localization (YOLOv8).",
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
MODEL_DET_DIR = BASE_DIR / "models" / "detection"

# Default class mapping
DEFAULT_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Global state
classification_model = None
classification_status = "Not initialized"
detection_model = None
detection_status = "Not initialized"
class_names = DEFAULT_CLASSES


# ================= Response Schemas ==================

class HealthResponse(BaseModel):
    status: str
    classification_loaded: bool
    detection_loaded: bool
    classification_status: str
    detection_status: str
    classes: List[str]


class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float
    is_tumor: bool
    probabilities: Dict[str, float]


class BoundingBox(BaseModel):
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str


class DetectionResult(BaseModel):
    num_detections: int
    boxes: List[BoundingBox]


# ================= Model Loading ==================

@app.on_event("startup")
def load_models():
    """Load trained models and class labels on startup."""
    global classification_model, classification_status
    global detection_model, detection_status
    global class_names

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
    cls_candidates = [
        MODEL_CLS_DIR / "best_model.keras",
        MODEL_CLS_DIR / "best_inceptionv3.keras",
    ]
    cls_file = next((p for p in cls_candidates if p.exists()), None)

    if not cls_file:
        classification_status = f"Model file not found in {MODEL_CLS_DIR} (expected best_model.keras)"
        logger.warning(classification_status)
    else:
        try:
            import tensorflow as tf
            classification_model = tf.keras.models.load_model(str(cls_file))
            classification_status = f"Loaded successfully from {cls_file.name}"
            logger.info(f"✓ {classification_status}")
        except ImportError:
            classification_status = "TensorFlow is not installed (run: pip install tensorflow)"
            logger.warning(classification_status)
        except Exception as e:
            classification_status = f"Failed to load model file: {e}"
            logger.error(classification_status)

    # Load YOLOv8 Detection Model
    det_candidates = [
        MODEL_DET_DIR / "best_yolov8.pt",
        MODEL_DET_DIR / "best.pt",
    ]
    det_file = next((p for p in det_candidates if p.exists()), None)

    if not det_file:
        detection_status = f"Model file not found in {MODEL_DET_DIR} (expected best_yolov8.pt)"
        logger.warning(detection_status)
    else:
        try:
            from ultralytics import YOLO
            detection_model = YOLO(str(det_file))
            detection_status = f"Loaded successfully from {det_file.name}"
            logger.info(f"{detection_status}")
        except ImportError:
            detection_status = "Ultralytics is not installed (run: pip install ultralytics)"
            logger.warning(detection_status)
        except Exception as e:
            detection_status = f"Failed to load detection model: {e}"
            logger.error(detection_status)


# ================= Preprocessing Helper ==================

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
        "service": "NeuroVision AI - Diagnostic API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    return HealthResponse(
        status="healthy",
        classification_loaded=classification_model is not None,
        detection_loaded=detection_model is not None,
        classification_status=classification_status,
        detection_status=detection_status,
        classes=class_names,
    )


@app.post("/predict/classification", response_model=ClassificationResult, tags=["Classification"])
@app.post("/predict", response_model=ClassificationResult, include_in_schema=False)
async def predict_classification(file: UploadFile = File(...)):
    """Classify brain MRI scan into 4 classes: Glioma, Meningioma, No Tumor, Pituitary."""
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
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/detection", response_model=DetectionResult, tags=["Detection"])
async def predict_detection(file: UploadFile = File(...)):
    """Detect and localize tumor lesions with bounding boxes using YOLOv8."""
    if detection_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Detection model not loaded: {detection_status}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run YOLO inference
        results = detection_model.predict(source=image, conf=0.25, verbose=False)
        boxes_list: List[BoundingBox] = []

        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
                conf = round(float(box.conf[0]), 4)
                cls_id = int(box.cls[0])
                name = result.names.get(cls_id, "tumor")

                boxes_list.append(BoundingBox(box=coords, confidence=conf, class_name=name))

        return DetectionResult(num_detections=len(boxes_list), boxes=boxes_list)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
