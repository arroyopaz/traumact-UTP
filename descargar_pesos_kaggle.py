# Ejecutar como ÚLTIMA celda del notebook final-ensemble.ipynb en Kaggle (después del entrenamiento).
# Empaqueta los pesos para descargarlos desde el panel "Output" (/kaggle/working).
import zipfile, os, json
files = ["model_soft_tissue.pth", "model_liver.pth", "model_abdomen.pth", "ensemble_results.json"]
with zipfile.ZipFile("/kaggle/working/traumact_pesos.zip", "w") as z:
    for f in files:
        assert os.path.exists(f), f"No existe {f}: vuelva a ejecutar el entrenamiento"
        z.write(f)
print("Listo: descargue /kaggle/working/traumact_pesos.zip (Output → traumact_pesos.zip)")
# Nota: si la sesión de Kaggle ya se cerró y no se guardó una versión ("Save Version → Save & Run All"),
# los .pth se perdieron y hay que volver a ejecutar el notebook completo (~1 h 45 min en GPU T4).
