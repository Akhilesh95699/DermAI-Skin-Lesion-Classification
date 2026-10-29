
# DermAI — Skin Lesion Classification

An AI-powered skin lesion classification research project using **ConvNeXt-Tiny**, **PyTorch**, **FastAPI**, and **HTML/CSS/JavaScript**.

## Overview

DermAI is a web-based research prototype that analyzes an uploaded skin image and returns the model's top three predicted skin lesion categories with confidence scores.

## Features

- Upload JPG and PNG images
- Preview the selected image
- Predict among seven skin lesion classes
- Display the top three predictions and confidence scores
- FastAPI backend for model inference
- Simple, responsive web interface
- Medical disclaimer for responsible use

## Technology Stack

- **Deep Learning:** PyTorch, ConvNeXt-Tiny
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Image Processing:** Pillow, torchvision

## Skin Lesion Classes

| Code | Category |
|---|---|
| `akiec` | Actinic keratosis / intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesion |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevus |
| `vasc` | Vascular lesion |

## Project Structure

```text
DermAI-Skin-Lesion-Classification/
├── frontend/
│   └── index.html
└── backend/
    ├── main.py
    └── requirements.txt
```

## Local Setup

### 1. Install dependencies

Create or activate a Python environment, then run:

```bash
pip install -r requirements.txt
```

### 2. Prepare the trained model

Place the exported TorchScript model `dermai_convnext_tiny_ts.pt` in the backend directory, as expected by the API configuration.

### 3. Start the backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to test the API.

### 4. Open the frontend

Run a local static server from the frontend directory:

```bash
python -m http.server 5500
```

Open `http://127.0.0.1:5500` in your browser.

Ensure the frontend API URL points to the running backend.

## Medical Disclaimer

DermAI is an educational research prototype, not a medical diagnostic device. Predictions and confidence scores must not be used as a substitute for professional medical assessment. A low-risk prediction does not rule out skin cancer. Consult a qualified dermatologist for medical concerns.

## Responsible Use

Use appropriate, authorized images and protect personal and medical information. Model performance can vary across image quality, lesion types, and populations.
