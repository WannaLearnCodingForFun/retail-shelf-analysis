from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import json

import cv2
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.analyze import analyze_image
from src.utils.config import load_config
from src.utils.paths import repo_root


def _model_path(cfg: dict) -> Path:
    root = repo_root()
    det = root / cfg.get("inference", {}).get("weights", {}).get("detector", "models/weights/shampoo_detector.pt")
    if det.exists():
        return det
    pre = root / "models/weights/detector_pretrained.pt"
    if pre.exists():
        return pre
    return Path("yolov8m.pt")


st.set_page_config(page_title="Shampoo Shelf Analysis", layout="wide")
st.title("FMCG Retail Intelligence — Shampoo")

cfg = load_config()
defaults = cfg.get("inference", {}).get("defaults", {})
conf_default = float(defaults.get("confidence_threshold", 0.45))
iou_default = float(defaults.get("iou_threshold", 0.5))
imgsz_default = int(defaults.get("image_size", 960))
max_det_default = int(defaults.get("max_detections", 100))

with st.sidebar:
    st.subheader("Run")
    run = st.button("Analyze", type="primary", use_container_width=True)
    st.caption("Stable defaults applied internally")

st.subheader("Upload")
uploaded = st.file_uploader("Upload shelf image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a shelf image to begin.")
    st.stop()
if not run:
    st.stop()

weights = _model_path(cfg)
classes = cfg["classes"]["names"]
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp.write(uploaded.read())
    tmp_path = Path(tmp.name)

result = analyze_image(
    tmp_path,
    weights,
    classes,
    conf=conf_default,
    iou=iou_default,
    imgsz=imgsz_default,
    max_det=max_det_default,
)
img = result.pop("rendered_bgr")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
st.subheader("Annotated image")
st.image(img_rgb, caption="Detected products with brand overlays", use_container_width=True)

det_df = pd.DataFrame(result["detections"])
total_products = int(result.get("total_products", len(det_df)))
dominant_brand = str(result["insights"].get("dominant_brand", "others"))
dominant_share = float(result["insights"].get("dominant_share", 0.0))
avg_conf = float(result.get("avg_confidence", 0.0))
unique_brands = int(det_df["detected_brand"].nunique()) if not det_df.empty else 0

st.subheader("KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total products", total_products)
c2.metric("Dominant brand", dominant_brand, f"{dominant_share:.2f}%")
c3.metric("Avg confidence", f"{avg_conf:.2f}")
c4.metric("Unique brands", unique_brands)

shelf = result["shelf_share"]
st.subheader("Shelf share")
shelf_df = pd.DataFrame({"brand": list(shelf.keys()), "share_pct": list(shelf.values())})
shelf_df = shelf_df[shelf_df["share_pct"] > 0].sort_values("share_pct", ascending=False)
fig_share = px.pie(
    shelf_df,
    names="brand",
    values="share_pct",
    hole=0.35,
    title="Shelf share by brand (%)",
)
fig_share.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig_share, use_container_width=True)

st.subheader("Facings (visible counts)")
facings = result.get("facings", {})
fac_df = pd.DataFrame({"brand": list(facings.keys()), "facings": list(facings.values())})
fac_df = fac_df[fac_df["facings"] > 0].sort_values("facings", ascending=False)
fig_fac = px.bar(fac_df, x="facings", y="brand", orientation="h", title="Facing counts by brand")
st.plotly_chart(fig_fac, use_container_width=True)

st.subheader("Insights")
st.write(result["insights"]["summary"])
for obs in result["insights"]["observations"]:
    st.write(f"- {obs}")

st.subheader("Detections")
if det_df.empty:
    st.warning("No detections found for this image.")
else:
    cols = ["detected_brand", "confidence", "shelf_share_area", "shelf_zone", "bbox_width", "bbox_height"]
    show = det_df.copy()
    for c in cols:
        if c not in show.columns:
            show[c] = None
    show = show[cols].sort_values(["shelf_share_area", "confidence"], ascending=[False, False])
    st.dataframe(show, use_container_width=True)

st.subheader("Export")
csv_bytes = det_df.to_csv(index=False).encode("utf-8") if not det_df.empty else b""
st.download_button("Download detections CSV", data=csv_bytes, file_name="detections.csv", mime="text/csv")
st.download_button("Download analysis JSON", data=json.dumps(result, indent=2).encode("utf-8"), file_name="analysis.json", mime="application/json")

