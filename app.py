"""TraumaCT-Triage · Prototipo funcional de apoyo al triaje de traumatismo abdominal en TC.

Ejecutar:  streamlit run app.py
Pesos:     coloque model_soft_tissue.pth, model_liver.pth y model_abdomen.pth en ./weights
           (generados por final-ensemble.ipynb) o súbalos desde la barra lateral.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.demo import make_demo_studies
from core.dicom_io import read_uploaded
from core.preprocessing import WINDOWS, WINDOW_ORDER, display_image
from core.triage import PRETTY, URGENT_LABELS, sort_worklist, triage

APP_DIR = Path(__file__).parent
WEIGHTS_DIR = APP_DIR / "weights"
LEVEL_STYLE = {"ALTO": ("#b00020", "#fde7ea"), "MEDIO": ("#b85c00", "#fff3e0"), "BAJO": ("#2e7d32", "#eef7ee")}
THESIS_AUC = {"kidney_high": 0.9564, "liver_high": 0.9536, "bowel_healthy": 0.9294, "bowel_injury": 0.9283,
              "spleen_high": 0.9148, "spleen_healthy": 0.7628, "kidney_healthy": 0.7507, "extravasation_injury": 0.7452,
              "extravasation_healthy": 0.7406, "liver_healthy": 0.7037, "spleen_low": 0.6788, "liver_low": 0.6741,
              "kidney_low": 0.5242}

st.set_page_config(page_title="TraumaCT-Triage", page_icon="🩻", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.tct-header {background:#1f4e79; color:#fff; padding:.7rem 1rem; border-radius:6px; margin-bottom:.8rem;}
.tct-header b {font-size:1.15rem;} .tct-header span {opacity:.85; font-size:.85rem;}
.tct-card {border:1px solid #c9ced6; border-radius:6px; padding:.45rem .6rem; margin-bottom:.35rem;}
.tct-bar {display:flex; align-items:center; gap:.5rem; margin:.18rem 0; font-size:.86rem;}
.tct-bar .lab {width:46%;} .tct-bar .trk {flex:1; background:#eceff3; height:.85rem; border-radius:3px; overflow:hidden;}
.tct-bar .val {width:2.6rem; text-align:right; font-variant-numeric:tabular-nums;}
.tct-note {border:1px solid #e0a040; background:#fff8e6; color:#8a4b00; padding:.5rem .7rem; border-radius:6px; font-size:.85rem;}
.tct-demo {border:1px dashed #7a5af8; background:#f3f0ff; color:#3d2a8a; padding:.4rem .7rem; border-radius:6px; font-size:.85rem; margin-bottom:.6rem;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- estado
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


ss("studies", [])
ss("slice_probs", {})       # study_id -> {'ensemble': [n,13], ...}
ss("analyzed_idx", {})      # study_id -> índices de cortes analizados
ss("results", {})
ss("selected", None)
ss("audit", [])
ss("mode_used", {})


def log(event: str, detail: str = ""):
    st.session_state.audit.append({"fecha_hora": dt.datetime.now().isoformat(timespec="seconds"), "evento": event, "detalle": detail})


# ----------------------------------------------------------------------------- modelo (perezoso)
@st.cache_resource(show_spinner="Cargando los 3 modelos EfficientNet-B0…")
def get_ensemble(fingerprint: str, _sources: dict):
    from core.model import get_device, load_ensemble
    dev = get_device()
    return load_ensemble(_sources, dev), str(dev)


def weight_sources(uploaded_w):
    """Prioridad: archivos subidos > carpeta ./weights. Devuelve (fuentes, faltantes)."""
    src, missing = {}, []
    for w in WINDOW_ORDER:
        up = uploaded_w.get(w)
        p = WEIGHTS_DIR / f"model_{w}.pth"
        if up is not None:
            src[w] = up.getvalue()
        elif p.exists():
            src[w] = str(p)
        else:
            missing.append(f"model_{w}.pth")
    return src, missing


def fingerprint(src: dict) -> str:
    h = hashlib.sha256()
    for w in WINDOW_ORDER:
        v = src[w]
        h.update(v if isinstance(v, (bytes, bytearray)) else (v + str(Path(v).stat().st_mtime)).encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("**1 · Pesos del ensemble**")
    uploaded_w = {}
    with st.expander("Subir model_{ventana}.pth", expanded=False):
        for w in WINDOW_ORDER:
            uploaded_w[w] = st.file_uploader(f"model_{w}.pth", type=["pth", "pt"], key=f"w_{w}")
    src, missing = weight_sources(uploaded_w)
    have_weights = not missing
    if have_weights:
        st.success("Pesos disponibles para las 3 ventanas.")
    else:
        st.warning("Faltan: " + ", ".join(missing) + ". Se usará el **modo demostración** (probabilidades simuladas).")

    st.markdown("**2 · Estudios**")
    source = st.radio("Origen", ["Estudios de demostración (sintéticos)", "Cargar DICOM (.dcm o .zip)"], label_visibility="collapsed")
    uploads = None
    if source.startswith("Cargar"):
        uploads = st.file_uploader("Series de TC anonimizadas", type=["dcm", "zip"], accept_multiple_files=True,
                                   help="Un .zip por estudio o varios .dcm. Se agrupan por StudyInstanceUID.")
    if st.button("📂 Cargar estudios", use_container_width=True):
        st.session_state.update(studies=[], slice_probs={}, analyzed_idx={}, results={}, selected=None, mode_used={})
        if source.startswith("Estudios"):
            st.session_state.studies = make_demo_studies()
            log("carga", "6 estudios sintéticos de demostración")
        elif uploads:
            with st.spinner("Leyendo archivos DICOM…"):
                studies, errors = read_uploaded([(u.name, u.getvalue()) for u in uploads])
            st.session_state.studies = studies
            for e in errors[:20]:
                st.error(e)
            log("carga", f"{len(studies)} estudio(s) DICOM; {len(errors)} archivo(s) con error")
        else:
            st.info("Seleccione archivos DICOM primero.")

    st.markdown("**3 · Parámetros de triaje**")
    stride = st.slider("Analizar 1 de cada N cortes", 1, 10, 2 if have_weights else 1,
                       help="Reduce el tiempo de inferencia en CPU. 1 = todos los cortes.")
    top_k = st.slider("Agregación: media de los k cortes más probables", 1, 10, 3)
    t_high = st.slider("Umbral prioridad ALTA", 0.05, 0.95, 0.50, 0.05)
    t_med = st.slider("Umbral prioridad MEDIA", 0.05, 0.95, 0.30, 0.05)
    st.caption("Umbrales y regla de agregación no validados clínicamente; ajustar con radiólogos.")
    run = st.button("▶️ Analizar y priorizar", type="primary", use_container_width=True,
                    disabled=not st.session_state.studies)

# ----------------------------------------------------------------------------- análisis
if run:
    studies = st.session_state.studies
    ensemble = None
    if have_weights:
        try:
            ensemble, dev = get_ensemble(fingerprint(src), src)
            log("modelo", f"ensemble cargado en {dev}")
        except Exception as exc:
            st.error(f"No se pudieron cargar los pesos: {exc}")
            st.stop()
    bar = st.progress(0.0, text="Analizando…")
    t0 = time.perf_counter()
    n_done = 0
    for i, s in enumerate(studies):
        idx = list(range(0, s.series.n_slices, stride))
        if ensemble is not None:
            from core.model import get_device, predict_series
            res = predict_series(ensemble, [s.series.hu[j] for j in idx], get_device(),
                                 progress=lambda f, i=i: bar.progress((i + f) / len(studies), text=f"Analizando {s.study_id}…"))
            st.session_state.mode_used[s.study_id] = "modelo"
        elif s.source == "demo":
            res = {"ensemble": s.sim_probs[idx]}
            st.session_state.mode_used[s.study_id] = "simulado"
        else:
            st.session_state.mode_used[s.study_id] = "sin_modelo"
            bar.progress((i + 1) / len(studies))
            continue
        n_done += len(idx)
        st.session_state.slice_probs[s.study_id] = res
        st.session_state.analyzed_idx[s.study_id] = idx
        st.session_state.results[s.study_id] = triage(res["ensemble"], top_k, t_high, t_med)
        bar.progress((i + 1) / len(studies), text=f"{s.study_id} listo")
    elapsed = time.perf_counter() - t0
    bar.empty()
    for k in [k for k in st.session_state if str(k).startswith("slider_")]:
        st.session_state.pop(k, None)
    if n_done and ensemble is not None:
        msg = f"{n_done} cortes analizados en {elapsed:.1f} s ({1000 * elapsed / n_done:.1f} ms/corte, incluye preprocesamiento)."
        st.toast(msg)
        log("analisis", msg + f" stride={stride}, k={top_k}, umbrales={t_high}/{t_med}")
    elif n_done:
        log("analisis", f"modo demostración (probabilidades simuladas); k={top_k}, umbrales={t_high}/{t_med}")
    order = sort_worklist(st.session_state.results)
    st.session_state.selected = order[0] if order else (studies[0].study_id if studies else None)

# re-calcular triaje si cambian umbrales/k sin re-inferir
for sid, res in st.session_state.slice_probs.items():
    st.session_state.results[sid] = triage(res["ensemble"], top_k, t_high, t_med)

# ----------------------------------------------------------------------------- encabezado
st.markdown('<div class="tct-header"><b>TraumaCT-Triage</b> · apoyo al triaje de traumatismo abdominal en TC<br>'
            '<span>Ensemble de 3 EfficientNet-B0 especializados en ventanas Hounsfield: tejidos blandos · hígado · abdomen</span></div>',
            unsafe_allow_html=True)
modes = set(st.session_state.mode_used.values())
if "simulado" in modes:
    st.markdown('<div class="tct-demo"><b>MODO DEMOSTRACIÓN.</b> No se encontraron los pesos entrenados: las probabilidades '
                'mostradas son <b>simuladas</b> y las imágenes son fantomas sintéticos. Sirve solo para ilustrar el flujo de la interfaz.</div>',
                unsafe_allow_html=True)
if "sin_modelo" in modes:
    st.warning("Los estudios DICOM cargados solo pueden visualizarse: para analizarlos cargue los tres archivos de pesos.")

studies = {s.study_id: s for s in st.session_state.studies}
if not studies:
    st.info("Use la barra lateral: **Cargar estudios** y luego **Analizar y priorizar**. "
            "Sin pesos entrenados la aplicación funciona en modo demostración con estudios sintéticos.")
    st.stop()

results = st.session_state.results
if st.session_state.selected not in studies:
    st.session_state.selected = next(iter(studies))

col_list, col_view, col_prob = st.columns([1.05, 1.6, 1.25], gap="medium")

# ----------------------------------------------------------------------------- lista de trabajo
with col_list:
    st.markdown("#### Lista de trabajo priorizada")
    ordered = sort_worklist(results) + [s for s in studies if s not in results]
    for sid in ordered:
        r = results.get(sid)
        if r:
            fg, bg = LEVEL_STYLE[r.level]
            sub = (f"{r.level} · {PRETTY[r.top_label]} {r.study_probs[r.top_label]:.2f}" if r.level != "BAJO"
                   else f"BAJO · sin hallazgo urgente probable (máx. {r.score:.2f})")
        else:
            fg, bg, sub = "#555", "#f4f4f4", "Pendiente de análisis"
        sel = "outline:2px solid #1f4e79;" if sid == st.session_state.selected else ""
        st.markdown(f'<div class="tct-card" style="background:{bg};{sel}"><b>{sid}</b><br>'
                    f'<span style="color:{fg};font-size:.85rem">{sub}</span></div>', unsafe_allow_html=True)
        if st.button("Abrir", key=f"open_{sid}", use_container_width=True):
            st.session_state.selected = sid
            log("abrir_estudio", sid)
            st.rerun()

study = studies[st.session_state.selected]
r = results.get(study.study_id)
probs_slices = st.session_state.slice_probs.get(study.study_id)
idx_an = st.session_state.analyzed_idx.get(study.study_id, [])

# ----------------------------------------------------------------------------- visor
with col_view:
    st.markdown(f"#### Visor · {study.study_id}")
    for w in study.warnings:
        st.caption("⚠️ " + w)
    wopt = {"Tejidos blandos": "soft_tissue", "Hígado": "liver", "Abdomen": "abdomen", "Rango completo": None}
    wsel = st.radio("Ventana", list(wopt), horizontal=True, label_visibility="collapsed")
    n = study.series.n_slices
    default = idx_an[r.top_slice] if r else n // 2
    skey = f"slider_{study.study_id}"
    if skey not in st.session_state or not 1 <= st.session_state[skey] <= n:
        st.session_state[skey] = int(min(default, n - 1)) + 1

    def _goto(k=skey, v=default + 1):
        st.session_state[k] = v

    c1, c2 = st.columns([2.3, 1.2])
    with c2:
        st.button("Ir al hallazgo", use_container_width=True, on_click=_goto, disabled=r is None)
    with c1:
        sl = st.slider("Corte", 1, n, key=skey) - 1
    hu = study.series.hu[sl]
    img = display_image(hu, wopt[wsel])
    show_cam = False
    if r and st.session_state.mode_used.get(study.study_id) == "modelo":
        show_cam = st.toggle(f"Mapa de atención Grad-CAM ({PRETTY[r.top_label]})", value=False)
    if show_cam:
        from core.model import LABEL_COLUMNS, get_device, gradcam
        ens, _ = get_ensemble(fingerprint(src), src)
        wcam = wopt[wsel] or "soft_tissue"
        cam = gradcam(ens[wcam], hu, wcam, LABEL_COLUMNS.index(r.top_label), get_device())
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        rgb = np.stack([img] * 3, -1)
        img = (0.6 * rgb + 0.4 * heat).astype(np.uint8)
        st.caption(f"Grad-CAM del modelo {wcam}: zonas que más influyen en la probabilidad del hallazgo. Solo orientativo.")
    st.image(img, use_container_width=True, clamp=True)
    pos_note = f"Corte {sl + 1} / {n}"
    if probs_slices is not None and sl in idx_an:
        pos_note += " · analizado"
    elif probs_slices is not None:
        pos_note += " · no analizado (stride)"
    st.caption(pos_note)

# ----------------------------------------------------------------------------- probabilidades
def bars(prob_dict: dict, labels: list):
    html = ""
    for lab in labels:
        p = prob_dict[lab]
        color = "#9aa3ad" if lab.endswith("healthy") else ("#b00020" if p >= t_high else "#c55a11" if p >= t_med else "#2e7d32")
        html += (f'<div class="tct-bar"><div class="lab">{PRETTY[lab]}</div><div class="trk">'
                 f'<div style="width:{p * 100:.1f}%;background:{color};height:100%"></div></div>'
                 f'<div class="val">{p:.2f}</div></div>')
    st.markdown(html, unsafe_allow_html=True)


with col_prob:
    st.markdown("#### Probabilidades del ensemble")
    if r is None:
        st.info("Estudio aún no analizado.")
    else:
        scope = st.radio("Ámbito", ["Estudio (agregado)", "Corte actual"], horizontal=True, label_visibility="collapsed")
        show_healthy = st.checkbox("Mostrar etiquetas de estado sano", value=False)
        from core.model import LABEL_COLUMNS
        labels = [l for l in LABEL_COLUMNS if show_healthy or not l.endswith("healthy")]
        labels = sorted(labels, key=lambda l: (l not in URGENT_LABELS, l))
        if scope.startswith("Estudio"):
            bars(r.study_probs, labels)
        else:
            near = int(np.argmin(np.abs(np.array(idx_an) - sl)))
            st.caption(f"Corte analizado más cercano: {idx_an[near] + 1}")
            bars(dict(zip(LABEL_COLUMNS, probs_slices["ensemble"][near])), labels)
        fg, bg = LEVEL_STYLE[r.level]
        main = (f"Hallazgo principal: {PRETTY[r.top_label]} (AUC en validación de la tesis: {THESIS_AUC[r.top_label]:.2f})"
                if r.level != "BAJO" else "Sin hallazgo urgente por encima de los umbrales configurados.")
        st.markdown(f'<div class="tct-card" style="background:{bg};margin-top:.6rem"><b style="color:{fg}">Prioridad {r.level}</b>'
                    f' · puntuación {r.score:.2f}<br><span style="font-size:.85rem">{main}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="tct-note">Resultado de apoyo. La decisión diagnóstica corresponde al médico radiólogo. '
                'Sistema de investigación sin validación clínica externa.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- pestañas inferiores
st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["📈 Perfil por cortes", "🧩 Detalle por modelo", "📄 Reporte y auditoría", "ℹ️ Acerca del prototipo"])
from core.model import LABEL_COLUMNS

with tab1:
    if probs_slices is None:
        st.info("Analice el estudio para ver la evolución de las probabilidades a lo largo de los cortes.")
    else:
        sel_labels = st.multiselect("Etiquetas", [l for l in LABEL_COLUMNS if not l.endswith("healthy")],
                                    default=[r.top_label] if r else URGENT_LABELS[:2], format_func=lambda l: PRETTY[l])
        df = pd.DataFrame(probs_slices["ensemble"], columns=LABEL_COLUMNS, index=[i + 1 for i in idx_an])
        df.index.name = "corte"
        if sel_labels:
            st.line_chart(df[sel_labels].rename(columns=PRETTY), height=260)
        st.caption("Probabilidad del ensemble en cada corte analizado; el estudio se resume con la media de los k cortes más altos.")

with tab2:
    if probs_slices is None or "soft_tissue" not in probs_slices:
        st.info("Disponible cuando se analiza con los pesos reales (en modo demostración no hay modelos individuales).")
    else:
        from core.triage import aggregate
        rows = {w: aggregate(probs_slices[w], top_k) for w in WINDOW_ORDER}
        rows["ensemble"] = r.study_probs
        tab = pd.DataFrame(rows).T[LABEL_COLUMNS].rename(columns=PRETTY)
        tab.index = [WINDOWS[w]["label"] if w in WINDOWS else "Ensemble (promedio)" for w in tab.index]
        st.dataframe(tab.style.format("{:.3f}").background_gradient(axis=None, cmap="Reds", vmin=0, vmax=1), use_container_width=True)
        st.caption("Probabilidades agregadas por estudio de cada modelo especializado y del ensemble (Ecuación 9 de la tesis).")

with tab3:
    if results:
        rep = []
        for sid in sort_worklist(results):
            rr = results[sid]
            row = {"estudio": sid, "prioridad": rr.level, "puntuacion": round(rr.score, 4),
                   "hallazgo_principal": rr.top_label, "corte_hallazgo": st.session_state.analyzed_idx[sid][rr.top_slice] + 1,
                   "modo": st.session_state.mode_used.get(sid)}
            row.update({k: round(v, 4) for k, v in rr.study_probs.items()})
            gt = studies[sid].ground_truth
            if gt:
                row["verdad_demo"] = ";".join(gt["lesiones"])
            rep.append(row)
        rep_df = pd.DataFrame(rep)
        st.dataframe(rep_df, use_container_width=True, hide_index=True)
        meta = {"generado": dt.datetime.now().isoformat(timespec="seconds"), "parametros": {"stride": stride, "top_k": top_k,
                "umbral_alto": t_high, "umbral_medio": t_med}, "aviso": "Resultado de apoyo; decisión del médico radiólogo."}
        c1, c2, c3 = st.columns(3)
        c1.download_button("Descargar reporte CSV", rep_df.to_csv(index=False).encode("utf-8"), "triaje_reporte.csv", "text/csv")
        c2.download_button("Descargar reporte JSON", json.dumps({"meta": meta, "estudios": rep}, ensure_ascii=False, indent=2).encode("utf-8"),
                           "triaje_reporte.json", "application/json")
        c3.download_button("Descargar registro de auditoría", pd.DataFrame(st.session_state.audit).to_csv(index=False).encode("utf-8"),
                           "registro_auditoria.csv", "text/csv")
    st.markdown("**Registro de auditoría de la sesión**")
    st.dataframe(pd.DataFrame(st.session_state.audit), use_container_width=True, hide_index=True)

with tab4:
    st.markdown("""
**Qué hace.** Lee series de TC en DICOM, convierte los píxeles a Unidades Hounsfield, aplica las ventanas de tejidos blandos (40/400),
hígado (30/150) y abdomen (60/400), ejecuta los tres modelos EfficientNet-B0 entrenados en `final-ensemble.ipynb`, promedia sus
probabilidades por corte, agrega por estudio y ordena la lista de trabajo por la probabilidad de lesión urgente
(alto grado de hígado, bazo o riñón; extravasación activa; lesión intestinal).

**Limitaciones que deben declararse.**
- Los modelos se entrenaron con cortes **anotados** de 269 pacientes; aplicarlos a todos los cortes de un estudio es un cambio de dominio
  y la regla de agregación por estudio (media de los k cortes más probables) es una decisión de diseño **no validada**.
- Los umbrales de prioridad son configurables y deben fijarse con radiólogos sobre datos de validación.
- AUC-ROC promedio del ensemble en validación: 0,7894; alto grado ≥ 0,91; bajo grado 0,52–0,68 (Capítulo 5 de la tesis).
- No es un dispositivo médico; no debe usarse para decisiones clínicas.

**Seguridad de la información.** El prototipo procesa los archivos en memoria, muestra identificadores seudonimizados (hash del
StudyInstanceUID), advierte si detecta atributos DICOM identificables y registra un log de auditoría descargable.
""")
