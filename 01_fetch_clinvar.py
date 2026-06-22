# =============================================================================
# MÓDULO 1 — Descarga y limpieza de variantes BRCA1 desde ClinVar
# =============================================================================
# Estrategia: descarga el archivo variant_summary.txt.gz del FTP de ClinVar
# (más confiable que la API para volúmenes grandes) y filtra solo BRCA1.
# =============================================================================

import requests
import gzip
import shutil
import pandas as pd
from pathlib import Path

# ---------- configuración ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RAW_FILE_GZ  = DATA_DIR / "variant_summary.txt.gz"
RAW_FILE_TSV = DATA_DIR / "variant_summary.txt"
OUT_LABELED  = DATA_DIR / "brca1_labeled.csv"   # benignas + patogénicas
OUT_VUS      = DATA_DIR / "brca1_vus.csv"        # VUS para inferencia

CLINVAR_FTP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# ---------- mapeo de clasificaciones ----------
PATHOGENIC_LABELS = {
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic",
}
BENIGN_LABELS = {
    "benign",
    "likely benign",
    "benign/likely benign",
}
VUS_LABELS = {
    "uncertain significance",
    "variant of uncertain significance",
}

# =============================================================================
# 1. Descarga
# =============================================================================

def download_clinvar(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[INFO] Archivo ya existe: {dest}. Saltando descarga.")
        return
    print(f"[INFO] Descargando ClinVar ({url}) ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1_048_576):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {pct:.1f}%", end="\r")
    print(f"\n[OK] Descargado → {dest}")


def decompress_gz(src: Path, dest: Path) -> None:
    if dest.exists():
        print(f"[INFO] Archivo descomprimido ya existe: {dest}. Saltando.")
        return
    print(f"[INFO] Descomprimiendo {src} ...")
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"[OK] Descomprimido → {dest}")


# =============================================================================
# 2. Carga y filtrado
# =============================================================================

def load_and_filter(tsv_path: Path) -> pd.DataFrame:
    print("[INFO] Cargando variant_summary.txt (puede tardar ~1 min) ...")
    cols_needed = [
        "GeneSymbol",
        "ClinicalSignificance",
        "ReviewStatus",
        "Name",           # contiene la notación HGVS
        "RS# (dbSNP)",
        "Type",
        "Assembly",
        "Chromosome",
        "Start",
        "Stop",
        "ReferenceAllele",
        "AlternateAllele",
    ]
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        low_memory=False,
        usecols=cols_needed,
        dtype=str,
    )
    print(f"[INFO] Total variantes en ClinVar: {len(df):,}")

    # Filtra solo BRCA1 y ensamble GRCh38
    df = df[
        (df["GeneSymbol"].str.upper() == "BRCA1") &
        (df["Assembly"] == "GRCh38")
    ].copy()
    print(f"[INFO] Variantes BRCA1 (GRCh38): {len(df):,}")
    return df


# =============================================================================
# 3. Normalización de clasificaciones
# =============================================================================

def normalize_classification(df: pd.DataFrame) -> pd.DataFrame:
    df["clinsig_raw"] = df["ClinicalSignificance"].str.strip().str.lower()

    def assign_label(sig: str) -> str | None:
        if pd.isna(sig):
            return None
        for p in PATHOGENIC_LABELS:
            if sig == p:
                return "pathogenic"
        for b in BENIGN_LABELS:
            if sig == b:
                return "benign"
        for v in VUS_LABELS:
            if v in sig:
                return "vus"
        return None  # conflictivas, no clasificadas, etc.

    df["label"] = df["clinsig_raw"].apply(assign_label)

    # Descartar filas sin categoría útil
    df = df[df["label"].notna()].copy()
    print(f"[INFO] Tras normalización: {len(df):,} variantes útiles")

    counts = df["label"].value_counts()
    print(f"  pathogenic : {counts.get('pathogenic', 0):>6,}")
    print(f"  benign     : {counts.get('benign', 0):>6,}")
    print(f"  vus        : {counts.get('vus', 0):>6,}")
    return df


# =============================================================================
# 4. Separación y guardado
# =============================================================================

def split_and_save(df: pd.DataFrame) -> None:
    # Clase binaria solo para entrenamiento
    labeled = df[df["label"].isin(["pathogenic", "benign"])].copy()
    labeled["binary_label"] = labeled["label"].map({"pathogenic": 1, "benign": 0})

    vus = df[df["label"] == "vus"].copy()

    labeled.to_csv(OUT_LABELED, index=False)
    vus.to_csv(OUT_VUS, index=False)

    print(f"\n[OK] Guardado: {OUT_LABELED}  ({len(labeled):,} filas)")
    print(f"[OK] Guardado: {OUT_VUS}  ({len(vus):,} filas)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MÓDULO 1 — Fetch ClinVar BRCA1")
    print("=" * 60)

    download_clinvar(CLINVAR_FTP_URL, RAW_FILE_GZ)
    decompress_gz(RAW_FILE_GZ, RAW_FILE_TSV)

    df = load_and_filter(RAW_FILE_TSV)
    df = normalize_classification(df)
    split_and_save(df)

    print("\n[DONE] Módulo 1 completado.")
    print(f"  → Siguiente paso: ejecutar 02_annotate_vep_gnomad.py")
