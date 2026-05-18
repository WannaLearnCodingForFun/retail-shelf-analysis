#!/usr/bin/env python3
"""
Streamlit web app for oil bottle image classification.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from inference import get_inference_transform, load_model
from src.config import BEST_MODEL_PATH
from src.utils import get_device

# Page config
st.set_page_config(
    page_title="Oil Bottle Classifier",
    page_icon="🛢️",
    layout="centered",
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@st.cache_resource
def get_cached_model():
    """Load model once and reuse across reruns."""
    device = get_device()
    model, class_names = load_model(BEST_MODEL_PATH, device)
    return model, class_names, device


def main() -> None:
    st.title("Oil Bottle Image Classifier")
    st.markdown(
        "Upload a **cropped bottle image** (or a clear product photo). "
        "The model predicts: **parachute**, **saffola**, or **other**."
    )

    if not BEST_MODEL_PATH.exists():
        st.error(
            f"Trained model not found at `{BEST_MODEL_PATH}`. "
            "Run `python train.py` first."
        )
        st.stop()

    try:
        model, class_names, device = get_cached_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    st.sidebar.markdown("### Model")
    st.sidebar.code(str(BEST_MODEL_PATH.name))
    st.sidebar.markdown(f"**Device:** `{device}`")
    st.sidebar.markdown(f"**Classes:** {', '.join(class_names)}")

    uploaded = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    # Validate extension from filename
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in VALID_EXTENSIONS:
        st.warning(f"Unexpected file type `{suffix}`. Trying to open as image anyway.")

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the uploaded file as an image: {exc}")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        # Save to temp path for predict() or run tensor path inline
        transform = get_inference_transform()
        tensor = transform(image).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        elapsed = time.perf_counter() - t0

        scores = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}
        pred_class = max(scores, key=scores.get)
        top_conf = scores[pred_class]

        st.metric("Predicted class", pred_class.upper())
        st.metric("Confidence", f"{top_conf:.1%}")
        st.caption(f"Inference time: **{elapsed * 1000:.1f} ms**")

        st.markdown("**Confidence by class**")
        for name in sorted(scores, key=scores.get, reverse=True):
            st.progress(scores[name], text=f"{name}: {scores[name]:.1%}")

    st.divider()
    st.markdown("#### Detailed scores")
    for name in sorted(scores, key=scores.get, reverse=True):
        st.write(f"- **{name}**: {scores[name]:.4f}")


if __name__ == "__main__":
    main()
