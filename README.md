# TraumaCT-Triage · Prototipo funcional (Streamlit)

Prototipo de apoyo al triaje de traumatismo abdominal en tomografía computarizada, construido sobre el
ensemble de tres EfficientNet-B0 especializados en ventanas Hounsfield de la tesis
(tejidos blandos 40/400, hígado 30/150, abdomen 60/400). **Uso exclusivamente académico; no es un dispositivo médico.**

## Instalación y ejecución

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                     # abre http://localhost:8501
```

## Modos de funcionamiento

| Modo | Requisitos | Qué muestra |
|---|---|---|
| Demostración | ninguno | 6 estudios sintéticos (fantomas) con probabilidades **simuladas**; ilustra el flujo de la interfaz |
| Modelo real | `weights/model_soft_tissue.pth`, `model_liver.pth`, `model_abdomen.pth` | Inferencia real del ensemble sobre estudios DICOM o sobre los fantomas |

Los pesos se obtienen del notebook `final-ensemble.ipynb` (ver `descargar_pesos_kaggle.py`).

## Flujo

1. **Cargar estudios**: demostración o archivos DICOM (`.dcm` sueltos o un `.zip` por estudio). Se agrupan por
   `StudyInstanceUID` y se usa la serie con más cortes, ordenada de craneal a caudal.
2. **Analizar y priorizar**: DICOM → HU (RescaleSlope/Intercept) → 3 ventanas → 3 modelos → promedio (ensemble) por corte →
   agregación por estudio (media de los *k* cortes más probables para las etiquetas de lesión) → nivel ALTO/MEDIO/BAJO.
3. **Revisar**: lista de trabajo priorizada, visor con selector de ventana y de corte, salto al corte del hallazgo,
   mapa Grad-CAM (solo con pesos reales), perfil de probabilidades por corte, detalle por modelo, reporte CSV/JSON y
   registro de auditoría.

## Estructura

```
app.py                  interfaz Streamlit
core/preprocessing.py   HU, ventanas y normalización (idénticos al notebook)
core/model.py           EfficientNetModel, carga del ensemble, inferencia, Grad-CAM
core/triage.py          agregación por estudio y priorización
core/dicom_io.py        lectura/agrupación DICOM, alerta de atributos identificables, seudónimos
core/demo.py            fantomas sintéticos y probabilidades simuladas
tests/test_core.py      pruebas (python -m pytest -q)
```

## Limitaciones

- Los modelos se entrenaron con cortes anotados de 269 pacientes (lesión intestinal o extravasación activa). Aplicarlos a
  **todos** los cortes de un estudio supone un cambio de dominio; la regla de agregación y los umbrales son decisiones de
  diseño no validadas, configurables en la barra lateral.
- Rendimiento de referencia (validación, tesis): AUC-ROC promedio 0,7894; alto grado ≥ 0,91; bajo grado 0,52–0,68.
- En CPU la inferencia del ensemble toma del orden de 150 ms por corte (medido en un servidor sin GPU); use el parámetro
  «Analizar 1 de cada N cortes» o una GPU.
- Use únicamente estudios anonimizados. La app advierte si detecta atributos DICOM identificables, pero no anonimiza.
