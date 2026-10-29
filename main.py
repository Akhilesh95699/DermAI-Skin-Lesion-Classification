import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FULL_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus (mole)",
    "vasc": "Vascular lesion",
}
MODEL_PATH = "dermai_convnext_tiny_ts.pt"
DISCLAIMER = (
    "This is an AI screening aid, not a medical diagnosis. "
    "A low-risk result does not rule out skin cancer. "
    "Please consult a dermatologist."
)

model = torch.jit.load(MODEL_PATH, map_location="cpu").eval()

transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="DermAI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG image.")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        p = 0
        for xf in (x, x.flip(3), x.flip(2), x.flip(2).flip(3)):
            p = p + model(xf).softmax(1)
        p = (p / 4)[0]

    probs, idx = p.topk(3)
    return {
        "predictions": [
            {
                "code": CLASS_NAMES[i],
                "name": FULL_NAMES[CLASS_NAMES[i]],
                "confidence": round(float(pr) * 100, 1),
            }
            for pr, i in zip(probs, idx)
        ],
        "disclaimer": DISCLAIMER,
    }
