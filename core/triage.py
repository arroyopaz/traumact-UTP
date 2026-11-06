"""Agregación de probabilidades por estudio y priorización de la lista de trabajo.

El modelo de la tesis clasifica CORTES. Para priorizar ESTUDIOS completos el prototipo
agrega las probabilidades por corte con una regla configurable (no validada clínicamente):
- etiquetas de lesión (injury/low/high): media de las k probabilidades más altas;
- etiquetas de estado sano (healthy): media sobre todos los cortes analizados.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import LABEL_COLUMNS

URGENT_LABELS = ["liver_high", "spleen_high", "kidney_high", "extravasation_injury", "bowel_injury"]
LOW_LABELS = ["liver_low", "spleen_low", "kidney_low"]

PRETTY = {
    "bowel_healthy": "Intestino sano", "bowel_injury": "Lesión intestinal",
    "extravasation_healthy": "Sin extravasación", "extravasation_injury": "Extravasación activa",
    "kidney_healthy": "Riñón sano", "kidney_low": "Riñón · bajo grado", "kidney_high": "Riñón · alto grado",
    "liver_healthy": "Hígado sano", "liver_low": "Hígado · bajo grado", "liver_high": "Hígado · alto grado",
    "spleen_healthy": "Bazo sano", "spleen_low": "Bazo · bajo grado", "spleen_high": "Bazo · alto grado",
}


@dataclass
class TriageResult:
    study_probs: dict          # etiqueta -> probabilidad agregada
    score: float               # puntuación de prioridad
    level: str                 # ALTO / MEDIO / BAJO
    top_label: str
    top_slice: int             # índice del corte con mayor probabilidad del hallazgo principal


def aggregate(slice_probs: np.ndarray, top_k: int = 3) -> dict:
    n = slice_probs.shape[0]
    k = max(1, min(top_k, n))
    out = {}
    for j, lab in enumerate(LABEL_COLUMNS):
        col = slice_probs[:, j]
        out[lab] = float(col.mean()) if lab.endswith("healthy") else float(np.sort(col)[-k:].mean())
    return out


def triage(slice_probs: np.ndarray, top_k: int = 3, t_high: float = 0.5, t_med: float = 0.3) -> TriageResult:
    probs = aggregate(slice_probs, top_k)
    urgent = {l: probs[l] for l in URGENT_LABELS}
    top_label = max(urgent, key=urgent.get)
    score = urgent[top_label]
    low_max = max(probs[l] for l in LOW_LABELS)
    if score >= t_high:
        level = "ALTO"
    elif score >= t_med or low_max >= t_high:
        level = "MEDIO"
        if low_max > score:
            top_label = max(LOW_LABELS, key=lambda l: probs[l])
            score = low_max
    else:
        level = "BAJO"
    top_slice = int(np.argmax(slice_probs[:, LABEL_COLUMNS.index(top_label)]))
    return TriageResult(probs, float(score), level, top_label, top_slice)


def sort_worklist(results: dict) -> list:
    """results: {study_id: TriageResult}. Orden: nivel y luego puntuación descendente."""
    rank = {"ALTO": 0, "MEDIO": 1, "BAJO": 2}
    return sorted(results, key=lambda s: (rank[results[s].level], -results[s].score))
