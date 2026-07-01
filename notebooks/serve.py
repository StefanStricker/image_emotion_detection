import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "convnext_small_rafdb.pth"

CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

_state: dict = {}


def _load_model(device: torch.device) -> nn.Module:
    model = models.convnext_small(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state["device"] = device
    _state["model"] = _load_model(device)
    _state["use_amp"] = device.type == "cuda"
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Emotion Detection API",
    description="Classifies facial expressions into one of 7 emotion categories.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EmotionPrediction(BaseModel):
    emotion: str
    confidence: float
    scores: dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess(image_bytes: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")
    return INFERENCE_TRANSFORMS(img)


@torch.inference_mode()
def _run_inference(tensor: torch.Tensor, use_tta: bool) -> torch.Tensor:
    device: torch.device = _state["device"]
    use_amp: bool = _state["use_amp"]
    model: nn.Module = _state["model"]

    batch = tensor.unsqueeze(0).to(device)

    with torch.autocast(device.type, enabled=use_amp):
        logits = model(batch)
        if use_tta:
            logits = (logits + model(TF.hflip(batch))) / 2

    return F.softmax(logits.squeeze(0), dim=0).cpu()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "device": str(_state.get("device", "not loaded"))}


@app.post("/predict", response_model=EmotionPrediction)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, …)"),
    tta: bool = Query(False, description="Apply test-time augmentation (horizontal flip average)"),
):
    """Return the predicted emotion and per-class confidence scores."""
    image_bytes = await file.read()
    tensor = _preprocess(image_bytes)
    probs = _run_inference(tensor, use_tta=tta)

    scores = {cls: round(probs[i].item(), 6) for i, cls in enumerate(CLASSES)}
    top_idx = int(probs.argmax())

    return EmotionPrediction(
        emotion=CLASSES[top_idx],
        confidence=round(probs[top_idx].item(), 6),
        scores=scores,
    )
