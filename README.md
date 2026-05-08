## Retail Shelf Intelligence for Shampoo Brands

An AI-powered retail analytics system that analyzes shampoo shelf images to detect products, identify major brands, estimate shelf share, and generate business insights through an interactive dashboard.

## Features

* Shampoo bottle detection using YOLOv8
* Brand recognition using CLIP zero-shot classification
* Shelf-share estimation based on occupied shelf area
* Facing count analytics
* Shelf position analysis (top / middle / bottom)
* Interactive Streamlit dashboard
* FastAPI backend for API-based inference
* Clean visual analytics and annotated detections

## Supported Brands

* Head & Shoulders
* Pantene
* Dove
* Sunsilk
* Others

## System Pipeline

Shelf Image
→ Bottle Detection
→ Product Cropping
→ Brand Classification
→ Shelf Analytics
→ Interactive Dashboard

## Project Structure

```bash
project_root/
├── app/
├── data/
├── models/
├── outputs/
├── src/
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Dashboard

```bash
streamlit run app/frontend/app.py
```

## Backend API

Run the FastAPI server:

```bash
uvicorn app.backend.main:app --reload --port 8000
```

API endpoint:

```bash
POST /analyze
```

Returns:

* detections
* shelf share
* facing counts
* shelf zones
* insights
* annotated image path

## Outputs

Generated results are stored in:

```bash
outputs/predictions/
outputs/metrics/
```

## Technologies Used

* Python
* PyTorch
* YOLOv8
* OpenAI CLIP
* Streamlit
* FastAPI
* Plotly

## Screenshots
<img width="1470" height="798" alt="image" src="https://github.com/user-attachments/assets/279e8de6-55c1-4a52-a626-f10f60139fa5" />
<img width="1212" height="625" alt="image" src="https://github.com/user-attachments/assets/8dc9ab83-ac44-4588-95c2-b2438d47f0f4" />
<img width="1464" height="747" alt="image" src="https://github.com/user-attachments/assets/bba667bb-3d6a-4da7-a2e1-97b901e33248" />




## Notes

* The system is optimized for retail shelf images containing shampoo products.
* Core brands are prioritized for more stable predictions.
* The project is designed for fast inference and demo reliability.


