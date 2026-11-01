import base64
import io
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import requests
import streamlit as st

# Configure page
st.set_page_config(
    page_title="NeuroVision | Diagnostic Multimodal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Dark Theme CSS
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #090D16 0%, #0F172A 100%);
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .badge-positive {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-negative {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")
ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = ROOT_DIR / "data" / "prepared" / "classification" / "test"
REPORT_DIR = ROOT_DIR / "outputs" / "reports"

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=64)
st.sidebar.title("NeuroVision AI")
st.sidebar.caption("Multimodal Medical Deep Learning Suite")

# API Health Check in Sidebar
health_data = {}
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=5)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        cls_ok = health_data.get("classification_loaded", False)
        det_ok = health_data.get("detection_loaded", False)
        if cls_ok and det_ok:
            st.sidebar.success("● API Online (InceptionV3 ✓, YOLO ✓)")
        elif cls_ok or det_ok:
            st.sidebar.success(f"● API Online ({'InceptionV3 ✓' if cls_ok else 'No Cls ⚠️'}, {'YOLO ✓' if det_ok else 'No Det'})")
        else:
            st.sidebar.info("⏳ API Initializing (Loading neural nets in background...)")
    else:
        st.sidebar.warning(f"▲ API Status: {health_resp.status_code}")
except Exception:
    st.sidebar.error("✖ API Offline (Start Docker: docker compose up -d)")

st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["Diagnostic Multimodal", "InceptionV3 Classification", "YOLOv8 Tumor Localization", "Dataset Manifest & MLflow", "Architecture"]
)

st.sidebar.markdown("---")
st.sidebar.caption("NeuroVision Biomedical MRI Workstation")


# Helper function to find sample MRI images
def get_sample_images():
    samples = {}
    if TEST_IMG_DIR.exists():
        for class_dir in sorted(TEST_IMG_DIR.iterdir()):
            if class_dir.is_dir():
                imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                if imgs:
                    samples[f"Sample: {class_dir.name}"] = imgs[0]
    return samples

SAMPLE_IMAGES = get_sample_images()


# =====================================================================
# TAB 1: DIAGNOSTIC MULTIMODAL
# =====================================================================
if menu == "Diagnostic Multimodal":
    st.title("Unified Multimodal Brain MRI Diagnosis")
    st.markdown("Integrates **InceptionV3** histological tumor classification and **YOLOv8** lesion localization in a single automated pass.")

    col_input, col_preview = st.columns([1, 1])

    with col_input:
        st.subheader("1. Input Brain MRI Slice")
        source_mode = st.radio("Image Source", ["Upload File", "Choose Preloaded Sample"], horizontal=True, key="multi_source")

        selected_image = None
        img_bytes = None

        if source_mode == "Upload File":
            uploaded_file = st.file_uploader("Upload Axial Brain MRI Scan (JPG/PNG)", type=["jpg", "jpeg", "png"], key="multi_upload")
            if uploaded_file:
                selected_image = Image.open(uploaded_file).convert("RGB")
                img_bytes = uploaded_file.getvalue()
        else:
            if SAMPLE_IMAGES:
                choice = st.selectbox("Select Benchmark Test Sample", list(SAMPLE_IMAGES.keys()), key="multi_choice")
                img_path = SAMPLE_IMAGES[choice]
                selected_image = Image.open(img_path).convert("RGB")
                buf = io.BytesIO()
                selected_image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
            else:
                st.info("No local test samples found in data/prepared/classification/test/.")

        conf_slider = st.slider("Detection Confidence Threshold", 0.10, 0.90, 0.25, 0.05, key="multi_conf")
        run_btn = st.button("Run Full Multimodal Analysis", type="primary", use_container_width=True)

    with col_preview:
        if selected_image is not None:
            st.subheader("Scan Preview")
            st.image(selected_image, caption=f"Original MRI ({selected_image.width}x{selected_image.height})", use_container_width=True)

    if run_btn:
        if img_bytes is None:
            st.warning("Please upload an MRI scan or select a preloaded sample first.")
        else:
            with st.spinner("Analyzing scan with InceptionV3 and YOLOv8 neural networks..."):
                try:
                    files = {"file": ("mri_scan.jpg", img_bytes, "image/jpeg")}
                    params = {"conf_threshold": conf_slider}
                    res = requests.post(f"{API_URL}/predict/multimodal", files=files, params=params, timeout=60)
                    
                    if res.status_code == 200:
                        data = res.json()
                        diag = data.get("patient_diagnosis", {})
                        det = data.get("detection_localization", {})
                        annotated_b64 = data.get("annotated_image_base64")

                        st.markdown("---")
                        st.subheader("2. Automated Clinical Diagnostic Report")

                        has_tumor = diag.get("has_tumor", False)
                        tumor_type = diag.get("tumor_type", "Unknown")
                        cls_conf = diag.get("classification_confidence", 0.0)

                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Pathological Status", "POSITIVE (Tumor Detected)" if has_tumor else "NEGATIVE (No Tumor)")
                        with col_m2:
                            st.metric("Primary Tumor Class", tumor_type)
                        with col_m3:
                            st.metric("Confidence", f"{cls_conf:.2%}")

                        col_raw, col_annot = st.columns(2)
                        with col_raw:
                            st.image(selected_image, caption="Original Raw MRI Scan", use_container_width=True)
                        with col_annot:
                            if annotated_b64:
                                ann_img = Image.open(io.BytesIO(base64.b64decode(annotated_b64)))
                                st.image(ann_img, caption="YOLOv8 Tumor Localization Overlay", use_container_width=True)
                            else:
                                st.info("YOLOv8 localization overlay not available (detection weights not loaded).")

                        col_probs, col_boxes = st.columns(2)
                        with col_probs:
                            st.subheader("Classification Class Distribution")
                            probs = diag.get("class_probabilities", {})
                            for cls_name, p_val in probs.items():
                                st.write(f"**{cls_name}**: {p_val:.2%}")
                                st.progress(min(max(float(p_val), 0.0), 1.0))

                        with col_boxes:
                            st.subheader("Lesion Localization Coordinates")
                            regions = det.get("regions", [])
                            if regions:
                                df_boxes = pd.DataFrame([
                                    {
                                        "Box ID": f"Tumor #{i+1}",
                                        "Confidence": f"{r['confidence']:.2%}",
                                        "Coordinates [X1, Y1, X2, Y2]": str(r['box_2d']),
                                    }
                                    for i, r in enumerate(regions)
                                ])
                                st.dataframe(df_boxes, use_container_width=True)
                            else:
                                st.info("No bounding boxes detected above threshold.")

                        st.info(f"**Diagnostic Summary:** {diag.get('clinical_synthesis')}")
                    else:
                        st.error(f"API Error ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with API backend: {e}")


# =====================================================================
# TAB 2: INCEPTIONV3 CLASSIFICATION
# =====================================================================
elif menu == "InceptionV3 Classification":
    st.title("Brain Tumor Histological Classification")
    st.markdown("Fine-tuned **InceptionV3** architecture predicting among: **Glioma, Meningioma, No Tumor, and Pituitary**.")

    col_input, col_preview = st.columns([1, 1])

    with col_input:
        st.subheader("Select Input Image")
        source_mode = st.radio("Image Source", ["Choose Preloaded Sample", "Upload File"], horizontal=True, key="cls_source")

        selected_image = None
        img_bytes = None

        if source_mode == "Choose Preloaded Sample":
            if SAMPLE_IMAGES:
                choice = st.selectbox("Select Benchmark Test Sample", list(SAMPLE_IMAGES.keys()), key="cls_choice")
                img_path = SAMPLE_IMAGES[choice]
                selected_image = Image.open(img_path).convert("RGB")
                buf = io.BytesIO()
                selected_image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
            else:
                st.info("No local test samples found in data/prepared/classification/test/.")
        else:
            uploaded_file = st.file_uploader("Upload MRI Scan (JPG/PNG)", type=["jpg", "png", "jpeg"], key="cls_upload")
            if uploaded_file:
                selected_image = Image.open(uploaded_file).convert("RGB")
                img_bytes = uploaded_file.getvalue()

        run_cls_btn = st.button("Run InceptionV3 Classification", type="primary", use_container_width=True)

    with col_preview:
        if selected_image is not None:
            st.subheader("Input MRI Preview")
            st.image(selected_image, caption=f"MRI Slice ({selected_image.width}x{selected_image.height})", use_container_width=True)

    if run_cls_btn:
        if img_bytes is None:
            st.warning("Please upload an image or choose a preloaded sample first.")
        else:
            with st.spinner("Classifying scan with InceptionV3..."):
                try:
                    files = {"file": ("mri.jpg", img_bytes, "image/jpeg")}
                    resp = requests.post(f"{API_URL}/predict/classification", files=files, timeout=60)
                    
                    if resp.status_code == 200:
                        res = resp.json()
                        pred_class = res.get("predicted_class", "Unknown")
                        confidence = res.get("confidence", 0.0)
                        is_tumor = res.get("is_tumor", False)
                        probs = res.get("probabilities") or res.get("class_probabilities", {})

                        st.markdown("---")
                        st.subheader("Classification Results")

                        col_r1, col_r2 = st.columns([1, 1])
                        with col_r1:
                            st.metric("Predicted Category", pred_class)
                            st.metric("Model Confidence", f"{confidence:.2%}")
                            if is_tumor:
                                st.error(f"Tumor Detected: {pred_class}")
                            else:
                                st.success("No Tumor Detected (Healthy / Normal Scan)")

                        with col_r2:
                            st.subheader("Class Probabilities")
                            for c_name, p_val in probs.items():
                                st.write(f"**{c_name}**: {p_val:.2%}")
                                st.progress(min(max(float(p_val), 0.0), 1.0))
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend at {API_URL}: {e}")


# =====================================================================
# TAB 3: YOLOV8 DETECTION
# =====================================================================
elif menu == "YOLOv8 Tumor Localization":
    st.title("YOLOv8 Tumor Object Detection")
    st.markdown("High-resolution object detection model for localizing neoplastic lesions with bounding boxes.")

    col_input, col_preview = st.columns([1, 1])

    with col_input:
        st.subheader("Select Input Image")
        source_mode = st.radio("Image Source", ["Choose Preloaded Sample", "Upload File"], horizontal=True, key="det_source")

        selected_image = None
        img_bytes = None

        if source_mode == "Choose Preloaded Sample":
            if SAMPLE_IMAGES:
                choice = st.selectbox("Select Benchmark Test Sample", list(SAMPLE_IMAGES.keys()), key="det_choice")
                img_path = SAMPLE_IMAGES[choice]
                selected_image = Image.open(img_path).convert("RGB")
                buf = io.BytesIO()
                selected_image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
            else:
                st.info("No local test samples found.")
        else:
            uploaded_file = st.file_uploader("Upload MRI for Detection", type=["jpg", "png", "jpeg"], key="det_upload")
            if uploaded_file:
                selected_image = Image.open(uploaded_file).convert("RGB")
                img_bytes = uploaded_file.getvalue()

        conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05, key="yolo_conf")
        iou = st.slider("IoU NMS Threshold", 0.10, 0.90, 0.45, 0.05, key="yolo_iou")
        run_det_btn = st.button("Run YOLOv8 Detection", type="primary", use_container_width=True)

    with col_preview:
        if selected_image is not None:
            st.subheader("Input MRI Preview")
            st.image(selected_image, caption="MRI Slice", use_container_width=True)

    if run_det_btn:
        if img_bytes is None:
            st.warning("Please upload an image or choose a preloaded sample first.")
        else:
            with st.spinner("Detecting tumor bounding boxes..."):
                try:
                    files = {"file": ("mri.jpg", img_bytes, "image/jpeg")}
                    params = {"conf_threshold": conf, "iou_threshold": iou}
                    resp = requests.post(f"{API_URL}/predict/detection", files=files, params=params, timeout=60)
                    
                    if resp.status_code == 200:
                        res = resp.json()
                        st.markdown("---")
                        st.subheader(f"Detection Results: {res.get('num_detections', 0)} Lesion(s) Found")

                        b64_img = res.get("annotated_image_base64")
                        if b64_img:
                            det_img = Image.open(io.BytesIO(base64.b64decode(b64_img)))
                            st.image(det_img, caption="Detected Tumors Overlay", use_container_width=True)
                        if res.get("detections"):
                            st.dataframe(pd.DataFrame(res["detections"]), use_container_width=True)
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend at {API_URL}: {e}")


# =====================================================================
# TAB 4: DATASET & MLFLOW
# =====================================================================
elif menu == "Dataset Manifest & MLflow":
    st.title("Dataset Lineage & MLflow Tracking")
    st.markdown("Full traceability of training, validation, and test sets according to ISO/IEEE AI standards.")

    manifest_path = REPORT_DIR / "global_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Classification Images", manifest_data.get("classification", {}).get("exploration", {}).get("total_valid", "N/A"))
        with col2:
            st.metric("YOLO Annotation Pairs", manifest_data.get("detection", {}).get("quality_report", {}).get("valid_pairs", "N/A"))
        with col3:
            st.metric("Git Commit Hash", manifest_data.get("git_commit", "unknown"))

        st.subheader("Data Manifest Specification")
        st.json(manifest_data)
    else:
        st.info("Global manifest not generated yet.")

    sample_box_img = REPORT_DIR / "yolo_sample_boxes.png"
    if sample_box_img.exists():
        st.subheader("Ground Truth YOLO Annotations Sample")
        st.image(str(sample_box_img), use_container_width=True)


# =====================================================================
# TAB 5: ARCHITECTURE
# =====================================================================
elif menu == "Architecture":
    st.title("NeuroVision System Architecture")
    st.markdown("""
    ### End-to-End Multimodal Diagnostic Pipeline
    - **InceptionV3 Classification**: Transfer learning on ImageNet, adapted for 4 brain MRI classes (Glioma, Meningioma, Pituitary, Normal).
    - **YOLOv8 Detection**: Real-time bounding box regression for tumor segmentation/localization.
    - **FastAPI Core**: Microservice providing unified `/predict/classification`, `/predict/detection`, and `/predict/multimodal` inference endpoints.
    - **Streamlit Interface**: Interactive radiological workstation.
    - **MLflow Tracking**: Experiment versioning, hyperparameter logging, and model registry.
    """)
