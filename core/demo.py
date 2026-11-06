"""Estudios SINTÉTICOS de demostración (fantomas generados por software, no son pacientes).

Sirven para mostrar el flujo de la interfaz cuando todavía no se cargan los pesos entrenados
(model_soft_tissue.pth, model_liver.pth, model_abdomen.pth). En ese modo las probabilidades
son SIMULADAS a partir de la lesión que se dibujó en el fantoma.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from .dicom_io import Series, Study
from .model import LABEL_COLUMNS

DEMO_CASES = [
    ("DEMO-01", {"liver_high": 1}, "Laceración hepática extensa"),
    ("DEMO-02", {"spleen_low": 1, "extravasation_injury": 1}, "Lesión esplénica con extravasación activa"),
    ("DEMO-03", {}, "Sin lesión"),
    ("DEMO-04", {"kidney_low": 1}, "Contusión renal de bajo grado"),
    ("DEMO-05", {}, "Sin lesión"),
    ("DEMO-06", {"kidney_high": 1}, "Lesión renal de alto grado"),
]


def _ell(xx, yy, cx, cy, rx, ry, ang=0.0):
    a = np.deg2rad(ang)
    X, Y = xx - cx, yy - cy
    xr = X * np.cos(a) + Y * np.sin(a)
    yr = -X * np.sin(a) + Y * np.cos(a)
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1


def phantom_slice(z: float, lesions: dict, n: int = 256, rng=None) -> np.ndarray:
    """Corte axial sintético en HU. z en [0, 1] (0 = craneal, 1 = caudal)."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n] / n
    hu = np.full((n, n), -1000.0, dtype=np.float32)
    hu[_ell(xx, yy, .5, .52, .45, .33)] = -100
    hu[_ell(xx, yy, .5, .52, .42, .30)] = 45
    hu[_ell(xx, yy, .5, .52, .39, .27)] = -90
    # tamaño de cada órgano según la altura del corte
    liv = max(0.0, 1 - abs(z - 0.3) / 0.35)
    spl = max(0.0, 1 - abs(z - 0.3) / 0.25)
    kid = max(0.0, 1 - abs(z - 0.65) / 0.22)
    liver = _ell(xx, yy, .30, .46, .17 * liv + 1e-3, .15 * liv + 1e-3, -15)
    spleen = _ell(xx, yy, .76, .44, .08 * spl + 1e-3, .10 * spl + 1e-3, 20)
    kl = _ell(xx, yy, .36, .66, .05 * kid + 1e-3, .07 * kid + 1e-3, 20)
    kr = _ell(xx, yy, .66, .66, .05 * kid + 1e-3, .07 * kid + 1e-3, -20)
    hu[liver] = 80
    hu[spleen] = 95
    hu[kl | kr] = 160
    hu[_ell(xx, yy, .51, .64, .025, .025)] = 220
    hu[_ell(xx, yy, .51, .74, .05, .045)] = 700
    hu[_ell(xx, yy, .51, .74, .03, .025)] = 250
    for cx, cy in [(.52, .40), (.60, .36), (.45, .33), (.58, .50), (.47, .50)]:
        hu[_ell(xx, yy, cx, cy, .045, .04)] = 30
        hu[_ell(xx, yy, cx, cy - .01, .03, .02)] = -950
    if lesions.get("liver_high") and liv > 0.5:
        hu[_ell(xx, yy, .27, .44, .09 * liv, .015, 35) & liver] = 35
        hu[_ell(xx, yy, .33, .52, .05 * liv, .012, -20) & liver] = 35
    if lesions.get("liver_low") and liv > 0.7:
        hu[_ell(xx, yy, .25, .40, .02, .006, 30) & liver] = 50
    if lesions.get("spleen_low") and spl > 0.6:
        hu[_ell(xx, yy, .77, .44, .03, .006, 40) & spleen] = 50
    if lesions.get("extravasation_injury") and spl > 0.6:
        hu[_ell(xx, yy, .79, .46, .015, .015)] = 260
        hu[_ell(xx, yy, .80, .56, .05, .02, 10) & ~spleen] = 25
    if lesions.get("kidney_low") and kid > 0.6:
        hu[_ell(xx, yy, .67, .64, .015, .015) & kr] = 60
    if lesions.get("kidney_high") and kid > 0.5:
        hu[_ell(xx, yy, .36, .66, .05 * kid, .012, 60) & kl] = 30
        hu[_ell(xx, yy, .31, .70, .04, .03) & ~kl] = 55
    hu += rng.normal(0, 12, hu.shape).astype(np.float32)
    return hu


def _lesion_window(label: str):
    if label.startswith("liver"):
        return 0.30, 0.35, 0.5
    if label.startswith("spleen") or label.startswith("extravasation"):
        return 0.30, 0.25, 0.6
    if label.startswith("kidney"):
        return 0.65, 0.22, 0.5
    return 0.5, 0.2, 0.6


def simulated_probs(lesions: dict, zs: np.ndarray, seed: int) -> np.ndarray:
    """Probabilidades SIMULADAS por corte (no provienen de ningún modelo)."""
    rng = np.random.default_rng(seed)
    n = len(zs)
    p = rng.uniform(0.02, 0.18, size=(n, len(LABEL_COLUMNS)))
    idx = {l: i for i, l in enumerate(LABEL_COLUMNS)}
    for organ in ["bowel", "extravasation", "kidney", "liver", "spleen"]:
        p[:, idx[f"{organ}_healthy"]] = rng.uniform(0.80, 0.95, n)
    for lab in lesions:
        c, half, _ = _lesion_window(lab)
        presence = np.clip(1 - np.abs(zs - c) / half, 0, 1)
        peak = 0.88 if lab.endswith(("high", "injury")) else 0.62
        p[:, idx[lab]] = np.maximum(p[:, idx[lab]], presence * peak + rng.normal(0, 0.03, n))
        organ = lab.split("_")[0]
        p[:, idx[f"{organ}_healthy"]] = np.minimum(p[:, idx[f"{organ}_healthy"]], 1 - presence * 0.7)
    return np.clip(p, 0, 1)


def make_demo_studies(n_slices: int = 48, size: int = 256) -> list:
    studies = []
    for k, (sid, lesions, desc) in enumerate(DEMO_CASES):
        rng = np.random.default_rng(100 + k)
        zs = np.linspace(0.05, 0.95, n_slices)
        s = Series(sid, f"demo-series-{k}", description=desc)
        s.hu = [phantom_slice(float(z), lesions, size, rng) for z in zs]
        s.positions = list(-zs * 300)
        s.instance_numbers = list(range(1, n_slices + 1))
        st = Study(sid, s, "demo", ["Estudio sintético (fantoma generado por software); no corresponde a un paciente."],
                   ground_truth={"lesiones": list(lesions) or ["ninguna"], "descripcion": desc})
        st.sim_probs = simulated_probs(lesions, zs, 200 + k)
        studies.append(st)
    return studies


def write_demo_dicoms(out_dir, n_slices: int = 48, size: int = 256) -> list:
    """Escribe los estudios de demostración como DICOM anonimizados (para probar la carga)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
    from pathlib import Path

    out = Path(out_dir)
    paths = []
    for k, (sid, lesions, desc) in enumerate(DEMO_CASES):
        rng = np.random.default_rng(100 + k)
        study_uid, series_uid = generate_uid(), generate_uid()
        for i, z in enumerate(np.linspace(0.05, 0.95, n_slices)):
            hu = phantom_slice(float(z), lesions, size, rng)
            px = np.clip(hu + 1024, 0, 4095).astype(np.uint16)
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            p = out / sid / "serie_1" / f"{i + 1}.dcm"
            p.parent.mkdir(parents=True, exist_ok=True)
            ds = FileDataset(str(p), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
            ds.Modality = "CT"
            ds.PatientID = sid
            ds.PatientName = ""
            ds.SeriesDescription = f"FANTOMA SINTETICO - {desc}"
            ds.StudyDate = _dt.date.today().strftime("%Y%m%d")
            ds.InstanceNumber = i + 1
            ds.ImagePositionPatient = [0.0, 0.0, float(-z * 300)]
            ds.Rows, ds.Columns = px.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
            ds.PixelRepresentation = 0
            ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
            ds.PixelData = px.tobytes()
            ds.save_as(str(p), enforce_file_format=True)
            paths.append(p)
    return paths
