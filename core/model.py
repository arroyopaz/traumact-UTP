"""Modelo EfficientNet-B0 del notebook y carga del ensemble de 3 ventanas Hounsfield."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .preprocessing import WINDOW_ORDER, normalize, window_rgb

LABEL_COLUMNS = [
    "bowel_healthy", "bowel_injury",
    "extravasation_healthy", "extravasation_injury",
    "kidney_healthy", "kidney_low", "kidney_high",
    "liver_healthy", "liver_low", "liver_high",
    "spleen_healthy", "spleen_low", "spleen_high",
]


class EfficientNetModel(nn.Module):
    """Misma arquitectura que en final-ensemble.ipynb (claves del state_dict compatibles).
    Se instancia sin descargar pesos de ImageNet: los pesos vienen de model_{ventana}.pth."""

    def __init__(self, num_classes: int = 13, dropout: float = 0.4):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.efficientnet = nn.Sequential(*list(base.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(1280, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clean_state_dict(sd: dict) -> dict:
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_model(source, device: torch.device) -> EfficientNetModel:
    """source: ruta a .pth o bytes del archivo subido."""
    if isinstance(source, (bytes, bytearray)):
        sd = torch.load(io.BytesIO(source), map_location=device, weights_only=True)
    else:
        sd = torch.load(Path(source), map_location=device, weights_only=True)
    m = EfficientNetModel(len(LABEL_COLUMNS))
    m.load_state_dict(_clean_state_dict(sd), strict=True)
    return m.to(device).eval()


def load_ensemble(sources: dict, device: torch.device) -> dict:
    """sources: {ventana: ruta|bytes}. Devuelve {ventana: modelo}."""
    return {w: load_model(sources[w], device) for w in WINDOW_ORDER}


@torch.inference_mode()
def predict_series(ensemble: dict, hu_slices: list, device: torch.device, batch_size: int = 16,
                   progress=None) -> dict:
    """Predice por corte con cada modelo y promedia (Ecuación (9)).
    Devuelve {'soft_tissue': [n,13], 'liver': [n,13], 'abdomen': [n,13], 'ensemble': [n,13]}."""
    out = {w: [] for w in WINDOW_ORDER}
    n = len(hu_slices)
    use_amp = device.type == "cuda"
    for start in range(0, n, batch_size):
        chunk = hu_slices[start:start + batch_size]
        for w in WINDOW_ORDER:
            x = torch.from_numpy(np.stack([normalize(window_rgb(h, w)) for h in chunk])).to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = ensemble[w](x)
            out[w].append(torch.sigmoid(logits.float()).cpu().numpy())
        if progress:
            progress(min(start + batch_size, n) / n)
    res = {w: np.concatenate(v, axis=0) for w, v in out.items()}
    res["ensemble"] = np.mean([res[w] for w in WINDOW_ORDER], axis=0)
    return res


def gradcam(model: EfficientNetModel, hu: np.ndarray, window: str, label_idx: int, device: torch.device) -> np.ndarray:
    """Mapa Grad-CAM (384x384, valores 0-1) sobre el último bloque convolucional."""
    feats = {}
    target_layer = model.efficientnet[0][-1]

    def fwd_hook(_, __, output):
        feats["a"] = output
        output.register_hook(lambda g: feats.__setitem__("g", g))

    h = target_layer.register_forward_hook(fwd_hook)
    try:
        x = torch.from_numpy(normalize(window_rgb(hu, window))[None]).to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logit = model(x)[0, label_idx]
        logit.backward()
        a, g = feats["a"][0], feats["g"][0]
        weights = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * a).sum(0)).detach().cpu().numpy()
    finally:
        h.remove()
    import cv2
    cam = cv2.resize(cam, (hu.shape[1], hu.shape[0]))
    return cam / (cam.max() + 1e-8)
