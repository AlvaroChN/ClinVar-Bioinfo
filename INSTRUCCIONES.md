# VarAI Detect — Guía de ejecución en VS Code

## 1. Instalar dependencias (una sola vez)

Abre la terminal de VS Code (`Ctrl + J`) y ejecuta:

```bash
pip install -r requirements.txt
```

---

## 2. Ejecutar los módulos en orden

Cada módulo se corre individualmente desde la terminal:

### Módulo 1 — Descargar ClinVar (~5–10 min, archivo ~500 MB)
```bash
python 01_fetch_clinvar.py
```
**Genera:** `data/brca1_labeled.csv` y `data/brca1_vus.csv`

---

### Módulo 2 — Anotar con VEP y gnomAD (~20–40 min según conexión)
```bash
python 02_annotate_vep_gnomad.py
```
**Genera:** `data/brca1_labeled_annotated.csv` y `data/brca1_vus_annotated.csv`

> ⚠️ Requiere internet. La variable `MAX_VARIANTS = 3000` limita la cantidad
> para pruebas rápidas. Ponla en `None` para procesar todo el dataset.

---

### Módulo 3 — Entrenar el modelo (~1–2 min)
```bash
python 03_train_model.py
```
**Genera:** `models/rf_model.pkl` y `models/confusion_matrix.png`

---

### Módulo 4 — Predecir y priorizar VUS (~5 seg)
```bash
python 04_predict_vus.py
```
**Genera:** `data/vus_prioritized.csv` y `models/vus_priority_distribution.png`

---

### Interfaz web Streamlit
```bash
streamlit run app.py
```
Se abre automáticamente en `http://localhost:8501`

---

## 3. Estructura de archivos generados

```
ClinVar/
├── data/
│   ├── variant_summary.txt.gz          # descargado de ClinVar FTP
│   ├── variant_summary.txt             # descomprimido
│   ├── brca1_labeled.csv               # benignas + patogénicas (sin anotar)
│   ├── brca1_vus.csv                   # VUS (sin anotar)
│   ├── brca1_labeled_annotated.csv     # con CADD, REVEL, AF
│   ├── brca1_vus_annotated.csv         # VUS con features
│   └── vus_prioritized.csv             # resultado final
├── models/
│   ├── rf_model.pkl                    # modelo serializado
│   ├── confusion_matrix.png
│   └── vus_priority_distribution.png
├── 01_fetch_clinvar.py
├── 02_annotate_vep_gnomad.py
├── 03_train_model.py
├── 04_predict_vus.py
├── app.py
└── requirements.txt
```

## 4. Correr como Jupyter Notebook en VS Code

Si prefieres ver los resultados celda por celda:
1. Instala la extensión **Jupyter** en VS Code
2. Abre cualquier `.py` y usa `# %%` para dividirlo en celdas (ya está soportado)
3. O crea un `.ipynb` y copia el código de cada módulo en celdas separadas
