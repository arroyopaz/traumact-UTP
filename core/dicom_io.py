"""Lectura de estudios DICOM subidos (archivos .dcm sueltos o .zip) y agrupación por estudio/serie."""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pydicom

from .preprocessing import to_hu

# Atributos que, si contienen datos, indican que el archivo podría no estar anonimizado
# (subconjunto del Basic Application Level Confidentiality Profile, DICOM PS3.15).
IDENTIFYING_TAGS = [
    "PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "InstitutionName", "InstitutionAddress", "ReferringPhysicianName",
    "AccessionNumber",
]


@dataclass
class Series:
    study_key: str
    series_uid: str
    hu: list = field(default_factory=list)        # lista de matrices HU (float32)
    positions: list = field(default_factory=list)  # posición para ordenar los cortes
    instance_numbers: list = field(default_factory=list)
    identifying: set = field(default_factory=set)
    description: str = ""

    def sorted(self) -> "Series":
        order = np.argsort(-np.asarray(self.positions))  # de craneal a caudal
        s = Series(self.study_key, self.series_uid, description=self.description, identifying=set(self.identifying))
        s.hu = [self.hu[i] for i in order]
        s.positions = [self.positions[i] for i in order]
        s.instance_numbers = [self.instance_numbers[i] for i in order]
        return s

    @property
    def n_slices(self) -> int:
        return len(self.hu)


@dataclass
class Study:
    study_id: str            # identificador mostrado (seudonimizado)
    series: Series
    source: str              # "dicom" o "demo"
    warnings: list = field(default_factory=list)
    ground_truth: dict | None = None  # solo para estudios de demostración


def pseudonym(uid: str) -> str:
    """Identificador corto no reversible para mostrar en pantalla."""
    return "EST-" + hashlib.sha256(uid.encode()).hexdigest()[:6].upper()


def _iter_files(uploaded):
    """uploaded: lista de (nombre, bytes). Expande archivos .zip."""
    for name, data in uploaded:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    if info.is_dir() or info.file_size == 0:
                        continue
                    base = info.filename.rsplit("/", 1)[-1]
                    if base.startswith(".") or "__MACOSX" in info.filename:
                        continue
                    yield info.filename, z.read(info)
        else:
            yield name, data


def read_uploaded(uploaded, max_files: int = 3000) -> tuple[list[Study], list[str]]:
    """Devuelve (estudios, errores). Agrupa por StudyInstanceUID y elige, por estudio,
    la serie axial con mayor número de cortes."""
    series_map: dict[tuple[str, str], Series] = {}
    errors: list[str] = []
    n = 0
    for name, data in _iter_files(uploaded):
        n += 1
        if n > max_files:
            errors.append(f"Se alcanzó el máximo de {max_files} archivos; el resto se ignoró.")
            break
        try:
            ds = pydicom.dcmread(io.BytesIO(data), force=True)
            px = ds.pixel_array
        except Exception as exc:  # archivo no DICOM o sin píxeles
            errors.append(f"{name}: no se pudo leer como imagen DICOM ({type(exc).__name__}).")
            continue
        if px.ndim != 2:
            errors.append(f"{name}: imagen multiframe o a color no soportada en el prototipo.")
            continue
        folder = name.rsplit("/", 2)[0] if name.count("/") >= 2 else "carga"
        study_uid = str(getattr(ds, "StudyInstanceUID", "") or getattr(ds, "PatientID", "") or folder)
        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or name.rsplit("/", 1)[0])
        key = (study_uid, series_uid)
        if key not in series_map:
            series_map[key] = Series(study_uid, series_uid, description=str(getattr(ds, "SeriesDescription", "")))
        s = series_map[key]
        hu = to_hu(px, getattr(ds, "RescaleSlope", 1), getattr(ds, "RescaleIntercept", 0))
        s.hu.append(hu)
        ipp = getattr(ds, "ImagePositionPatient", None)
        inst = int(getattr(ds, "InstanceNumber", len(s.hu)) or len(s.hu))
        s.positions.append(float(ipp[2]) if ipp is not None and len(ipp) == 3 else float(inst))
        s.instance_numbers.append(inst)
        for tag in IDENTIFYING_TAGS:
            v = getattr(ds, tag, None)
            if v not in (None, "") and str(v).strip() not in ("", "ANONYMOUS", "Anonymized"):
                s.identifying.add(tag)

    by_study: dict[str, list[Series]] = {}
    for s in series_map.values():
        by_study.setdefault(s.study_key, []).append(s)

    studies = []
    for uid, lst in by_study.items():
        best = max(lst, key=lambda x: x.n_slices).sorted()
        warns = []
        if len(lst) > 1:
            warns.append(f"El estudio tiene {len(lst)} series; se analiza la de más cortes ({best.n_slices}).")
        if best.identifying:
            warns.append("Posibles datos identificables en atributos DICOM: " + ", ".join(sorted(best.identifying))
                         + ". Anonimice los estudios antes de usarlos (DICOM PS3.15).")
        studies.append(Study(pseudonym(uid), best, "dicom", warns))
    return studies, errors
