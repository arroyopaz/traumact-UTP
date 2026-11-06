"""Preprocesamiento idéntico al del notebook final-ensemble.ipynb.

DICOM -> Unidades Hounsfield -> ventana -> uint8 RGB 384x384 -> normalización ImageNet.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = 384

WINDOWS = {
    "soft_tissue": {"center": 40, "width": 400, "label": "Tejidos blandos (40/400)"},
    "liver": {"center": 30, "width": 150, "label": "Hígado (30/150)"},
    "abdomen": {"center": 60, "width": 400, "label": "Abdomen (60/400)"},
}
WINDOW_ORDER = ["soft_tissue", "liver", "abdomen"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_hu(pixels: np.ndarray, slope: float = 1.0, intercept: float = 0.0) -> np.ndarray:
    """Ecuación (1) de la tesis: HU = PV x RescaleSlope + RescaleIntercept."""
    return pixels.astype(np.float32) * float(slope) + float(intercept)


def window_limits(window: str) -> tuple[int, int]:
    w = WINDOWS[window]
    return w["center"] - w["width"] // 2, w["center"] + w["width"] // 2


def apply_window(hu: np.ndarray, window: str) -> np.ndarray:
    """Ecuación (2): devuelve una imagen uint8 en escala de grises (H x W)."""
    lower, upper = window_limits(window)
    x = (np.clip(hu, lower, upper) - lower) / (upper - lower)
    return (x * 255).astype(np.uint8)


def window_rgb(hu: np.ndarray, window: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Igual que RSNATraumaDataset._load_dicom: gris -> 3 canales -> resize."""
    img = apply_window(hu, window)
    rgb = np.stack([img, img, img], axis=-1)
    return cv2.resize(rgb, (size, size))


def normalize(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW normalizado con media/desviación de ImageNet."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def display_image(hu: np.ndarray, window: str | None) -> np.ndarray:
    """Imagen para el visor (no se redimensiona). window=None -> rango completo."""
    if window is None:
        x = (np.clip(hu, -1000, 1000) + 1000) / 2000
        return (x * 255).astype(np.uint8)
    return apply_window(hu, window)
