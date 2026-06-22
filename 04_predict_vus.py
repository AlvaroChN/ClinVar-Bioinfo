# =============================================================================
# MÓDULO 4 — Predicción y priorización de VUS
# =============================================================================
# Carga el modelo entrenado y lo aplica sobre las VUS anotadas.
# Genera la tabla final con probabilidad de patogenicidad y nivel de prioridad.
# =============================================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- configuración ----------
DATA_DIR  = Path("data")
MODEL_DIR = Path("models")

VUS_CSV    = DATA_DIR / "brca1_vus_annotated.csv"
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
OUT_RESULT = DATA_DIR / "vus_prioritized.csv"

FEATURES = ["cadd_phred", "polyphen_score", "sift_score", "af"]

# Umbrales de prioridad (ajustables)
THRESHOLD_HIGH   = 0.70   # probabilidad ≥ 0.70 → prioridad ALTA
THRESHOLD_MEDIUM = 0.40   # probabilidad ≥ 0.40 → prioridad MEDIA
                           # probabilidad  < 0.40 → prioridad BAJA


# =============================================================================
# 1. Carga de datos y modelo
# =============================================================================

def load_artifacts() -> tuple[pd.DataFrame, object]:
    print("[INFO] Cargando VUS y modelo...")
    vus = pd.read_csv(VUS_CSV)
    model = joblib.load(MODEL_PATH)
    print(f"[OK] VUS cargadas: {len(vus):,}")
    print(f"[OK] Modelo cargado: {MODEL_PATH}")
    return vus, model


# =============================================================================
# 2. Predicción
# =============================================================================

def predict(vus: pd.DataFrame, model) -> pd.DataFrame:
    vus = vus.copy()

    # Asegura que los features estén en buen estado
    for f in FEATURES:
        vus[f] = pd.to_numeric(vus[f], errors="coerce").fillna(0.0)

    X = vus[FEATURES].values
    prob_pathogenic = model.predict_proba(X)[:, 1]  # P(clase = 1 | X)

    vus["prob_pathogenic"] = prob_pathogenic
    return vus


# =============================================================================
# 3. Asignación de prioridad
# =============================================================================

def assign_priority(prob: float) -> str:
    if prob >= THRESHOLD_HIGH:
        return "ALTA"
    elif prob >= THRESHOLD_MEDIUM:
        return "MEDIA"
    else:
        return "BAJA"


def build_results(vus: pd.DataFrame) -> pd.DataFrame:
    vus["priority"] = vus["prob_pathogenic"].apply(assign_priority)

    # Ordena de mayor a menor probabilidad de patogenicidad
    result = vus.sort_values("prob_pathogenic", ascending=False).reset_index(drop=True)

    # Selecciona columnas finales (RF-19)
    cols = ["Name", "cadd_phred", "revel_score", "af", "prob_pathogenic", "priority"]
    result = result[[c for c in cols if c in result.columns]]
    result.insert(0, "rank", result.index + 1)
    return result


# =============================================================================
# 4. Visualizaciones
# =============================================================================

def plot_priority_distribution(result: pd.DataFrame) -> None:
    counts = result["priority"].value_counts().reindex(["ALTA", "MEDIA", "BAJA"])
    colors = {"ALTA": "#e74c3c", "MEDIA": "#f39c12", "BAJA": "#27ae60"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Barras de prioridad
    axes[0].bar(counts.index, counts.values,
                color=[colors[p] for p in counts.index])
    axes[0].set_title("Distribución por nivel de prioridad")
    axes[0].set_ylabel("N° de VUS")
    for i, (p, v) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    # Histograma de probabilidades
    axes[1].hist(result["prob_pathogenic"], bins=30, color="#3498db", edgecolor="white")
    axes[1].axvline(THRESHOLD_HIGH,   color="#e74c3c", linestyle="--",
                    label=f"Alta (≥{THRESHOLD_HIGH})")
    axes[1].axvline(THRESHOLD_MEDIUM, color="#f39c12", linestyle="--",
                    label=f"Media (≥{THRESHOLD_MEDIUM})")
    axes[1].set_title("Distribución de probabilidad de patogenicidad")
    axes[1].set_xlabel("P(patogénica)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(MODEL_DIR / "vus_priority_distribution.png", dpi=150)
    print(f"[OK] Gráfico guardado: {MODEL_DIR / 'vus_priority_distribution.png'}")
    plt.close()


def print_top_vus(result: pd.DataFrame, n: int = 20) -> None:
    print(f"\n[INFO] Top {n} VUS de mayor prioridad:")
    print("-" * 80)
    top = result.head(n)[["rank", "Name", "cadd_phred", "polyphen_score",
                           "sift_score", "af", "prob_pathogenic", "priority"]]
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(top.to_string(index=False))
    print("-" * 80)


def print_summary(result: pd.DataFrame) -> None:
    counts = result["priority"].value_counts()
    total  = len(result)
    print(f"\n{'='*50}")
    print("RESUMEN DE PRIORIZACIÓN")
    print(f"{'='*50}")
    print(f"  Total VUS analizadas : {total:,}")
    for p in ["ALTA", "MEDIA", "BAJA"]:
        n = counts.get(p, 0)
        pct = n / total * 100 if total else 0
        print(f"  Prioridad {p:<6}: {n:>5,}  ({pct:.1f}%)")
    print(f"{'='*50}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MÓDULO 4 — Predicción y priorización de VUS")
    print("=" * 60)

    vus, model = load_artifacts()
    vus = predict(vus, model)
    result = build_results(vus)

    result.to_csv(OUT_RESULT, index=False)
    print(f"\n[OK] Tabla de resultados guardada: {OUT_RESULT}")

    print_summary(result)
    print_top_vus(result, n=20)
    plot_priority_distribution(result)

    print("\n[DONE] Módulo 4 completado.")
    print("  → Para la interfaz web: ejecutar  streamlit run app.py")
