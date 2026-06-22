# =============================================================================
# MODULO 2 - Anotacion con Ensembl VEP
# =============================================================================
# Features extraidos de VEP REST API:
#   cadd_phred    - impacto deletéreo general (0-60)
#   polyphen_score - impacto sobre la proteina para missense (0-1)
#   sift_score    - tolerancia del cambio aminoacidico (0-1, menor = mas danino)
#   af            - frecuencia alelica global (desde colocated_variants)
#
# Formato HGVS que acepta VEP:  NM_007294.4:c.190T>G
# Formato de ClinVar:           NM_007294.4(BRCA1):c.190T>G (p.Cys64Gly)
# =============================================================================

import re
import time
import json
import requests
import pandas as pd
from pathlib import Path

# ---------- configuracion ----------
DATA_DIR    = Path("data")
LABELED_CSV = DATA_DIR / "brca1_labeled.csv"
VUS_CSV     = DATA_DIR / "brca1_vus.csv"
OUT_LABELED = DATA_DIR / "brca1_labeled_annotated.csv"
OUT_VUS     = DATA_DIR / "brca1_vus_annotated.csv"

VEP_URL     = "https://rest.ensembl.org/vep/human/hgvs"
VEP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

BATCH_SIZE  = 100
SLEEP_BATCH = 1.5

# Cantidad de variantes a procesar (None = todas)
MAX_LABELED = 500
MAX_VUS     = 200


# =============================================================================
# 1. Limpieza HGVS
# =============================================================================

def clean_hgvs(name: str) -> str | None:
    """
    NM_007294.4(BRCA1):c.190T>G (p.Cys64Gly)  ->  NM_007294.4:c.190T>G
    """
    if pd.isna(name):
        return None
    name = str(name).strip()
    name = name.split(" ")[0]                    # quita descripcion proteica
    name = re.sub(r"\([^)]+\)", "", name)        # quita (BRCA1)
    if ":" not in name:
        return None
    if not (name.startswith("NM_") or name.startswith("NC_") or name.startswith("NG_")):
        return None
    return name


# =============================================================================
# 2. Llamada VEP (batch)
# =============================================================================

def call_vep_batch(hgvs_list: list[str]) -> list[dict]:
    payload = {
        "hgvs_notations": hgvs_list,
        "CADD": 1,
        "canonical": 1,
    }
    try:
        resp = requests.post(
            VEP_URL,
            headers=VEP_HEADERS,
            data=json.dumps(payload),
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        print(f"  [WARN] VEP {resp.status_code}: {resp.text[:150]}")
        return []
    except Exception as e:
        print(f"  [WARN] Error VEP: {e}")
        return []


def extract_features(entry: dict) -> dict:
    """
    Extrae cadd_phred, polyphen_score, sift_score y af de una entrada VEP.
    Recorre todas las transcript_consequences y toma el valor del transcrito
    canonico si esta disponible; de lo contrario el primero con valor.
    """
    cadd, polyphen, sift, af = None, None, None, None

    for tc in entry.get("transcript_consequences", []):
        is_canonical = tc.get("canonical") == 1

        c = tc.get("cadd_phred")
        p = tc.get("polyphen_score")
        s = tc.get("sift_score")

        if is_canonical:
            if c is not None: cadd = c
            if p is not None: polyphen = p
            if s is not None: sift = s
            break
        else:
            if cadd is None and c is not None: cadd = c
            if polyphen is None and p is not None: polyphen = p
            if sift is None and s is not None: sift = s

    # AF desde colocated_variants > frequencies > gnomade
    for cv in entry.get("colocated_variants", []):
        for allele_freqs in cv.get("frequencies", {}).values():
            gnomade = allele_freqs.get("gnomade")
            gnomadg = allele_freqs.get("gnomadg")
            candidate = gnomade if gnomade is not None else gnomadg
            if candidate is not None:
                if af is None or candidate > af:
                    af = candidate

    return {
        "cadd_phred":     cadd,
        "polyphen_score": polyphen,
        "sift_score":     sift,
        "af":             af,
    }


# =============================================================================
# 3. Proceso completo
# =============================================================================

def annotate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["hgvs_clean"] = df["Name"].apply(clean_hgvs)
    valid = df["hgvs_clean"].notna()
    print(f"  HGVS validos: {valid.sum():,} / {len(df):,}")

    df_valid = df[valid].copy().reset_index(drop=True)
    hgvs_list = df_valid["hgvs_clean"].tolist()
    total = len(hgvs_list)

    results: dict[str, dict] = {}

    print(f"  Enviando {total:,} variantes a VEP (batches de {BATCH_SIZE})...")
    for i in range(0, total, BATCH_SIZE):
        batch = hgvs_list[i : i + BATCH_SIZE]
        raw = call_vep_batch(batch)
        for entry in raw:
            hgvs_id = entry.get("input", "")
            results[hgvs_id] = extract_features(entry)
        done = min(i + BATCH_SIZE, total)
        n_cadd = sum(1 for v in results.values() if v.get("cadd_phred") is not None)
        print(f"  {done:>4}/{total}  |  con CADD: {n_cadd}", end="\r")
        time.sleep(SLEEP_BATCH)
    print()

    # Mapear resultados
    for feat in ["cadd_phred", "polyphen_score", "sift_score", "af"]:
        df_valid[feat] = df_valid["hgvs_clean"].map(
            lambda x, f=feat: results.get(x, {}).get(f)
        )

    # Imputacion con medianas
    for col, default in [("cadd_phred", 15.0), ("polyphen_score", 0.5),
                         ("sift_score", 0.05), ("af", 0.0)]:
        med = df_valid[col].median()
        if pd.isna(med):
            med = default
        df_valid[col] = df_valid[col].fillna(med)

    # Resumen
    n_cadd  = (df_valid["cadd_phred"]     != 15.0).sum()
    n_pp    = (df_valid["polyphen_score"] != 0.5).sum()
    n_sift  = (df_valid["sift_score"]     != 0.05).sum()
    n_af0   = (df_valid["af"] == 0.0).sum()
    print(f"  [{label}] Total: {len(df_valid):,}  "
          f"CADD real: {n_cadd:,}  PolyPhen real: {n_pp:,}  "
          f"SIFT real: {n_sift:,}  AF=0: {n_af0:,}")

    return df_valid


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODULO 2 - Anotacion VEP")
    print("=" * 60)

    # --- Labeled ---
    print("\n[PASO 1] Variantes etiquetadas...")
    labeled = pd.read_csv(LABELED_CSV)
    if MAX_LABELED:
        labeled = labeled.sample(n=min(MAX_LABELED, len(labeled)), random_state=42)
        print(f"  Muestra: {len(labeled):,} variantes")

    labeled = annotate(labeled, "labeled")
    cols = ["Name", "hgvs_clean", "label", "binary_label",
            "cadd_phred", "polyphen_score", "sift_score", "af", "Type"]
    labeled[[c for c in cols if c in labeled.columns]].to_csv(OUT_LABELED, index=False)
    print(f"[OK] {OUT_LABELED}")

    # --- VUS ---
    print("\n[PASO 2] VUS...")
    vus = pd.read_csv(VUS_CSV)
    if MAX_VUS:
        vus = vus.sample(n=min(MAX_VUS, len(vus)), random_state=42)
        print(f"  Muestra: {len(vus):,} VUS")

    vus = annotate(vus, "vus")
    cols_vus = ["Name", "hgvs_clean", "cadd_phred", "polyphen_score", "sift_score", "af", "Type"]
    vus[[c for c in cols_vus if c in vus.columns]].to_csv(OUT_VUS, index=False)
    print(f"[OK] {OUT_VUS}")

    print("\n[DONE] Modulo 2 completado.")
    print("  -> Siguiente paso: ejecutar 03_train_model.py")
