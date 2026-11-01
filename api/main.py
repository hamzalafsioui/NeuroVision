import os
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_VERBOSE"] = "false"
try:
    import torch
    from ultralytics import YOLO
except ImportError:
    pass

import base64
import io
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Set KERAS_HOME inside project to avoid permission errors
BASE_DIR = Path(__file__).resolve().parent.parent
keras_dir = BASE_DIR / ".keras"
keras_dir.mkdir(exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(keras_dir))

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

import threading

# Global state
classification_model = None
classification_status = "Initializing in background..."
detection_model = None
detection_status = "Initializing in background..."
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
    class_probabilities: Dict[str, float]


class BoundingBox(BaseModel):
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str


class DetectionResult(BaseModel):
    num_detections: int
    boxes: List[BoundingBox]
    detections: List[Dict[str, Any]]
    annotated_image_base64: Optional[str] = None


# ================= Model Loading Worker ==================

def _load_models_worker():
    """Load trained models and class labels in a background thread."""
    global classification_model, classification_status
    global detection_model, detection_status
    global class_names

    # 1) Load Class Names
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

    # 2) Load and Warm up YOLOv8 FIRST (PyTorch must initialize before TensorFlow)
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
            logger.info(f"✓ {detection_status}")

            # Warm up YOLO model graph and plotting assets in memory
            try:
                dummy_det = np.zeros((640, 640, 3), dtype=np.uint8)
                warm_res = detection_model.predict(source=dummy_det, conf=0.25, verbose=False)
                if warm_res and len(warm_res) > 0:
                    _ = warm_res[0].plot()
                logger.info("✓ YOLOv8 model and plotting assets warmed up.")
            except Exception as e_w:
                logger.warning(f"Detection warmup skipped: {e_w}")
        except ImportError:
            detection_status = "Ultralytics is not installed (run: pip install ultralytics)"
            logger.warning(detection_status)
        except Exception as e:
            detection_status = f"Failed to load detection model: {e}"
            logger.error(detection_status)

    # 3) Load and Warm up InceptionV3 SECOND
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

            # Warm up classification model graph in memory
            try:
                dummy_cls = np.zeros((1, 299, 299, 3), dtype=np.float32)
                _ = classification_model.predict(dummy_cls, verbose=0)
                logger.info("✓ InceptionV3 model graph warmed up.")
            except Exception as e_w:
                logger.warning(f"Classification warmup skipped: {e_w}")
        except ImportError:
            classification_status = "TensorFlow is not installed (run: pip install tensorflow)"
            logger.warning(classification_status)
        except Exception as e:
            classification_status = f"Failed to load model file: {e}"
            logger.error(classification_status)


@app.on_event("startup")
def startup_event():
    """Start model loading in background thread so FastAPI opens port 8000 instantly."""
    t = threading.Thread(target=_load_models_worker, daemon=True)
    t.start()


# ================= Preprocessing Helper ==================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((299, 299))
    img_array = np.array(image, dtype=np.float32)

    # InceptionV3 scaling: [0, 255] -> [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 299, 299, 3)


def perform_classification(contents: bytes) -> Dict[str, Any]:
    if classification_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Classification model not loaded: {classification_status}",
        )

    tensor = preprocess_image(contents)
    raw_preds = classification_model.predict(tensor, verbose=0)[0]
    pred_idx = int(np.argmax(raw_preds))
    pred_class = class_names[pred_idx]
    confidence = float(raw_preds[pred_idx])

    probabilities = {
        class_names[i]: round(float(raw_preds[i]), 4)
        for i in range(len(class_names))
    }
    is_tumor = pred_class.lower() != "no tumor"

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "is_tumor": is_tumor,
        "probabilities": probabilities,
        "class_probabilities": probabilities,
    }


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


# Classification Endpoints (supporting multiple route names for compatibility)
@app.post("/predict/classification", response_model=ClassificationResult, tags=["Classification"])
@app.post("/predict/classify", response_model=ClassificationResult, tags=["Classification"])
@app.post("/predict", response_model=ClassificationResult, include_in_schema=False)
async def predict_classification_endpoint(file: UploadFile = File(...)):
    """Classify brain MRI scan into 4 classes: Glioma, Meningioma, No Tumor, Pituitary."""
    try:
        contents = await file.read()
        return perform_classification(contents)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Detection Endpoints (supporting /detection and /detect)
@app.post("/predict/detection", response_model=DetectionResult, tags=["Detection"])
@app.post("/predict/detect", response_model=DetectionResult, tags=["Detection"])
async def predict_detection_endpoint(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, alias="conf_threshold"),
    iou_threshold: float = Query(0.45, alias="iou_threshold"),
):
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
        results = detection_model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        boxes_list: List[BoundingBox] = []
        annotated_b64 = None

        if results and len(results) > 0:
            result = results[0]
            # Convert plotted image to base64
            plotted_bgr = result.plot()
            plotted_rgb = Image.fromarray(plotted_bgr[:, :, ::-1])
            buf = io.BytesIO()
            plotted_rgb.save(buf, format="JPEG")
            annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            for box in result.boxes:
                coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
                conf = round(float(box.conf[0]), 4)
                cls_id = int(box.cls[0])
                name = result.names.get(cls_id, "tumor")

                boxes_list.append(BoundingBox(box=coords, confidence=conf, class_name=name))

        detections_table = [
            {
                "box_id": f"Tumor #{i+1}",
                "confidence": b.confidence,
                "box": b.box,
                "class_name": b.class_name,
            }
            for i, b in enumerate(boxes_list)
        ]

        return DetectionResult(
            num_detections=len(boxes_list),
            boxes=boxes_list,
            detections=detections_table,
            annotated_image_base64=annotated_b64,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


# Multimodal Endpoint (combines InceptionV3 and YOLOv8)
@app.post("/predict/multimodal", tags=["Multimodal Diagnosis"])
async def predict_multimodal_endpoint(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, alias="conf_threshold"),
):
    """Unified diagnosis: histological classification + lesion localization overlay."""
    contents = await file.read()

    # 1) Classification
    cls_res = perform_classification(contents)

    # 2) Detection (if available)
    regions: List[Dict[str, Any]] = []
    annotated_b64 = None

    if detection_model is not None:
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            results = detection_model.predict(source=image, conf=conf_threshold, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                plotted_bgr = result.plot()
                plotted_rgb = Image.fromarray(plotted_bgr[:, :, ::-1])
                buf = io.BytesIO()
                plotted_rgb.save(buf, format="JPEG")
                annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                for box in result.boxes:
                    coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]
                    conf = round(float(box.conf[0]), 4)
                    name = result.names.get(int(box.cls[0]), "tumor")
                    regions.append({
                        "confidence": conf,
                        "box_2d": coords,
                        "class_name": name,
                    })
        except Exception as e:
            logger.warning(f"Detection step failed in multimodal pipeline: {e}")

    # 3) Clinical Synthesis
    if not cls_res["is_tumor"]:
        synthesis = f"Negative for intracranial neoplasm ({cls_res['confidence']:.1%} confidence)."
    else:
        loc_str = f" with {len(regions)} localized lesion(s)" if regions else ""
        synthesis = f"Positive: identified {cls_res['predicted_class']} tumor ({cls_res['confidence']:.1%} confidence){loc_str}."

    return {
        "patient_diagnosis": {
            "has_tumor": cls_res["is_tumor"],
            "tumor_type": cls_res["predicted_class"],
            "classification_confidence": cls_res["confidence"],
            "class_probabilities": cls_res["probabilities"],
            "clinical_synthesis": synthesis,
        },
        "detection_localization": {
            "num_lesions": len(regions),
            "regions": regions,
        },
        "annotated_image_base64": annotated_b64,
    }
