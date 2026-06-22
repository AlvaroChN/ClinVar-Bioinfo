# =============================================================================
# MODULO 3 - Entrenamiento del modelo Random Forest
# =============================================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay,
)

# ---------- configuracion ----------
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

ANNOTATED_CSV = DATA_DIR / "brca1_labeled_annotated.csv"
MODEL_PATH    = MODEL_DIR / "rf_model.pkl"
FEATURES      = ["cadd_phred", "polyphen_score", "sift_score", "af"]
TARGET        = "binary_label"

RANDOM_STATE  = 42
N_FOLDS       = 5
TEST_SIZE     = 0.20


# =============================================================================
# 1. Carga y validacion
# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Dataset cargado: {len(df):,} variantes")
    print(f"  Patogenicas (1): {(df[TARGET] == 1).sum():,}")
    print(f"  Benignas    (0): {(df[TARGET] == 0).sum():,}")

    # Convierte features a numerico e imputa nulos con mediana
    defaults = {"cadd_phred": 15.0, "polyphen_score": 0.5,
                "sift_score": 0.05, "af": 0.0}
    for col, default in defaults.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        if pd.isna(med):
            med = default
        df[col] = df[col].fillna(med)

    missing = df[FEATURES].isna().sum()
    if missing.any():
        print(f"[WARN] Nulos residuales:\n{missing}")

    print(f"[OK] Filas listas para entrenamiento: {len(df):,}")
    return df


# =============================================================================
# 2. Validacion cruzada de 5 pliegues
# =============================================================================

def cross_validate_model(X: np.ndarray, y: np.ndarray) -> None:
    print("\n[INFO] Validacion cruzada estratificada (5 pliegues)...")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "precision": "precision",
        "recall":    "recall",
        "f1":        "f1",
        "roc_auc":   "roc_auc",
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    print(f"\n  {'Metrica':<12} {'Media':>8} {'Std':>8}")
    print("  " + "-" * 30)
    for metric in ["precision", "recall", "f1", "roc_auc"]:
        vals = results[f"test_{metric}"]
        print(f"  {metric:<12} {vals.mean():>8.4f} {vals.std():>8.4f}")


# =============================================================================
# 3. Entrenamiento final y evaluacion en test set
# =============================================================================

def train_final_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\n[INFO] Split: train={len(X_train):,}  test={len(X_test):,}")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n[INFO] Reporte en conjunto de prueba:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Benigna (0)", "Patogenica (1)"],
    ))

    auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC: {auc:.4f}")

    print("\n[INFO] Importancia de variables:")
    for feat, imp in zip(FEATURES, model.feature_importances_):
        bar = "#" * int(imp * 40)
        print(f"  {feat:<15} {imp:.4f}  {bar}")

    plot_confusion_matrix(y_test, y_pred)
    return model


def plot_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benigna", "Patogenica"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de Confusion - Conjunto de Prueba")
    plt.tight_layout()
    out = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    print(f"[OK] Grafico guardado: {out}")
    plt.close()


# =============================================================================
# 4. Serializacion
# =============================================================================

def save_model(model: RandomForestClassifier) -> None:
    joblib.dump(model, MODEL_PATH)
    print(f"\n[OK] Modelo guardado: {MODEL_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODULO 3 - Entrenamiento Random Forest")
    print("=" * 60)

    df = load_data(ANNOTATED_CSV)

    X = df[FEATURES].values
    y = df[TARGET].values

    cross_validate_model(X, y)
    model = train_final_model(X, y)
    save_model(model)

    print("\n[DONE] Modulo 3 completado.")
    print("  -> Siguiente paso: ejecutar 04_predict_vus.py")
