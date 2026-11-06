"""Pruebas del prototipo (ejecutar: python -m pytest -q desde la carpeta traumact_triage)."""
import io
import zipfile

import numpy as np

from core.preprocessing import apply_window, normalize, to_hu, window_limits, window_rgb
from core.triage import triage, sort_worklist
from core.demo import make_demo_studies, write_demo_dicoms
from core.dicom_io import read_uploaded


def test_hu_and_windows():
    assert np.allclose(to_hu(np.array([0, 1000, 2000]), 1, -1024), [-1024, -24, 976])
    assert window_limits("soft_tissue") == (-160, 240)
    assert window_limits("liver") == (-45, 105)
    assert window_limits("abdomen") == (-140, 260)
    img = apply_window(np.array([[-500.0, 30.0, 500.0]]), "liver")
    assert img.tolist() == [[0, 127, 255]]


def test_tensor_shape():
    rgb = window_rgb(np.random.default_rng(0).normal(0, 300, (512, 512)), "abdomen")
    assert rgb.shape == (384, 384, 3)
    assert normalize(rgb).shape == (3, 384, 384)


def test_triage_order():
    studies = make_demo_studies(n_slices=24, size=128)
    res = {s.study_id: triage(s.sim_probs) for s in studies}
    order = sort_worklist(res)
    assert res[order[0]].level == "ALTO"
    assert res["DEMO-03"].level == "BAJO" and res["DEMO-05"].level == "BAJO"


def test_dicom_roundtrip(tmp_path):
    write_demo_dicoms(tmp_path, n_slices=6, size=64)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in tmp_path.rglob("*.dcm"):
            z.write(p, p.relative_to(tmp_path))
    studies, errors = read_uploaded([("estudios.zip", buf.getvalue())])
    assert not errors and len(studies) == 6
    assert all(s.series.n_slices == 6 for s in studies)
    assert studies[0].series.hu[0].min() >= -1024


def test_model_architecture():
    import torch
    from core.model import EfficientNetModel
    m = EfficientNetModel().eval()
    assert sum(p.numel() for p in m.parameters()) == 4_671_113
    with torch.no_grad():
        assert m(torch.zeros(2, 3, 384, 384)).shape == (2, 13)
