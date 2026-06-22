# =============================================================================
# VarAI Detect — Interfaz Streamlit
# =============================================================================
# Ejecutar con:  streamlit run app.py
# =============================================================================

import io
import joblib
import time
import requests
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- configuración ----------
MODEL_PATH = Path("models") / "rf_model.pkl"
FEATURES   = ["cadd_phred", "polyphen_score", "sift_score", "af"]

THRESHOLD_HIGH   = 0.70
THRESHOLD_MEDIUM = 0.40

VEP_URL     = "https://rest.ensembl.org/vep/human/hgvs"
VEP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
GNOMAD_URL  = "https://gnomad.broadinstitute.org/api"


# =============================================================================
# Helpers de anotación (reutilizados del módulo 2)
# =============================================================================

def annotate_vep_batch(hgvs_list):
    results = {}
    try:
        resp = requests.post(
            VEP_URL,
            headers=VEP_HEADERS,
            data=json.dumps({"hgvs_notations": hgvs_list}),
            timeout=60,
        )
        if resp.status_code != 200:
            return results
        for entry in resp.json():
            hgvs_id = entry.get("input", "")
            cadd, polyphen, sift = None, None, None
            for tc in entry.get("transcript_consequences", []):
                if cadd is None and "cadd_phred" in tc:
                    cadd = tc["cadd_phred"]
                if polyphen is None and "polyphen_score" in tc:
                    polyphen = tc["polyphen_score"]
                if sift is None and "sift_score" in tc:
                    sift = tc["sift_score"]
            results[hgvs_id] = {"cadd_phred": cadd, "polyphen_score": polyphen, "sift_score": sift}
    except Exception:
        pass
    return results


GNOMAD_QUERY = """
query VariantAF($variantId: String!, $datasetId: DatasetId!) {
  variant(variantId: $variantId, dataset: $datasetId) {
    genome { af }
    exome  { af }
  }
}
"""

def get_af_gnomad(variant_id):
    try:
        resp = requests.post(
            GNOMAD_URL,
            json={"query": GNOMAD_QUERY,
                  "variables": {"variantId": variant_id, "datasetId": "gnomad_r4"}},
            timeout=20,
        )
        if resp.status_code != 200:
            return None
        v = resp.json().get("data", {}).get("variant")
        if not v:
            return None
        af = (v.get("genome") or {}).get("af") or (v.get("exome") or {}).get("af")
        return float(af) if af is not None else None
    except Exception:
        return None


def parse_vcf(content: str) -> pd.DataFrame:
    """Parsea un VCF básico y retorna las columnas esenciales."""
    rows = []
    for line in content.splitlines():
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        chrom, pos, vid, ref, alt = parts[:5]
        rows.append({
            "CHROM": chrom.replace("chr", ""),
            "POS": pos,
            "ID": vid,
            "REF": ref,
            "ALT": alt,
        })
    return pd.DataFrame(rows)


def assign_priority(prob: float) -> str:
    if prob >= THRESHOLD_HIGH:
        return "ALTA"
    elif prob >= THRESHOLD_MEDIUM:
        return "MEDIA"
    return "BAJA"


PRIORITY_COLOR = {"ALTA": "🔴", "MEDIA": "🟡", "BAJA": "🟢"}


# =============================================================================
# Carga del modelo (cacheado)
# =============================================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# =============================================================================
# UI principal
# =============================================================================

def main():
    st.set_page_config(
        page_title="VarAI Detect",
        page_icon="🧬",
        layout="wide",
    )

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/"
                 "FullText_Search_Specification.pdf/page1-120px-"
                 "FullText_Search_Specification.pdf.jpg",
                 width=80)
        st.title("VarAI Detect")
        st.caption("Clasificación y priorización de variantes VUS en BRCA1")
        st.divider()
        st.markdown("**Umbrales de prioridad**")
        high_thresh = st.slider("Prioridad ALTA ≥", 0.5, 0.95, THRESHOLD_HIGH, 0.05)
        med_thresh  = st.slider("Prioridad MEDIA ≥", 0.2, 0.69, THRESHOLD_MEDIUM, 0.05)
        st.divider()
        st.markdown("**Acerca de**")
        st.caption("Features: CADD, REVEL score, Frecuencia alélica (gnomAD)")
        st.caption("Modelo: Random Forest (scikit-learn)")

    # --- Cabecera ---
    st.title("🧬 VarAI Detect")
    st.markdown(
        "Herramienta bioinformática para la **clasificación y priorización de "
        "variantes de significado incierto (VUS)** en el gen **BRCA1** mediante "
        "aprendizaje automático."
    )

    model = load_model()
    if model is None:
        st.error(
            "⚠️ Modelo no encontrado. Ejecuta primero `03_train_model.py` "
            "para generar `models/rf_model.pkl`."
        )
        st.stop()

    st.success("✅ Modelo cargado correctamente")

    # --- Tabs ---
    tab_upload, tab_manual, tab_results = st.tabs(
        ["📂 Cargar archivo VCF", "✏️ Ingresar variante manualmente", "📊 Resultados"]
    )

    # =========================================================================
    # TAB 1 — Cargar VCF
    # =========================================================================
    with tab_upload:
        st.markdown("### Sube el archivo VCF del paciente")
        uploaded = st.file_uploader(
            "Formato soportado: .vcf", type=["vcf", "txt"]
        )

        if uploaded:
            content = uploaded.read().decode("utf-8", errors="ignore")
            df_vcf = parse_vcf(content)

            if df_vcf.empty:
                st.error("El archivo no contiene variantes válidas o el formato es incorrecto.")
                st.stop()

            st.success(f"✅ {len(df_vcf)} variantes detectadas en el archivo.")
            st.dataframe(df_vcf.head(10))

            if st.button("Anotar y predecir"):
                with st.spinner("Anotando con VEP..."):
                    # Formato HGVS genomico que acepta VEP: 17:g.43106478T>A
                    hgvs_list = [
                        f"{r['CHROM']}:g.{r['POS']}{r['REF']}>{r['ALT']}"
                        for _, r in df_vcf.iterrows()
                    ]
                    vep_res = annotate_vep_batch(hgvs_list)

                    cadds, polyphens, sifts, afs = [], [], [], []
                    prog = st.progress(0)
                    for i, (_, row) in enumerate(df_vcf.iterrows()):
                        hgvs = hgvs_list[i]
                        vep  = vep_res.get(hgvs, {})
                        cadds.append(vep.get("cadd_phred") or 15.0)
                        polyphens.append(vep.get("polyphen_score") or 0.5)
                        sifts.append(vep.get("sift_score") or 0.05)
                        # AF desde gnomAD usando chrom-pos-ref-alt
                        vid = f"{row['CHROM']}-{row['POS']}-{row['REF']}-{row['ALT']}"
                        af = get_af_gnomad(vid)
                        afs.append(af if af is not None else 0.0)
                        prog.progress((i + 1) / len(df_vcf))
                        time.sleep(0.1)

                    df_vcf["cadd_phred"]     = pd.to_numeric(cadds,     errors="coerce").fillna(15.0)
                    df_vcf["polyphen_score"] = pd.to_numeric(polyphens, errors="coerce").fillna(0.5)
                    df_vcf["sift_score"]     = pd.to_numeric(sifts,     errors="coerce").fillna(0.05)
                    df_vcf["af"]             = pd.to_numeric(afs,       errors="coerce").fillna(0.0)

                    X = df_vcf[FEATURES].values
                    prob = model.predict_proba(X)[:, 1]
                    df_vcf["prob_pathogenic"] = prob
                    df_vcf["priority"] = [assign_priority(p) for p in prob]
                    df_vcf = df_vcf.sort_values("prob_pathogenic", ascending=False).reset_index(drop=True)

                    st.session_state["result_df"] = df_vcf
                    st.success("Prediccion completada. Ve a la pestana Resultados.")

    # =========================================================================
    # TAB 2 — Ingreso manual
    # =========================================================================
    with tab_manual:
        st.markdown("### Ingresa los valores de la variante manualmente")
        st.caption(
            "Útil si ya tienes los scores calculados o quieres probar el modelo."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            cadd  = st.number_input("CADD phred score", min_value=0.0, max_value=60.0, value=25.0, step=0.5)
        with col2:
            revel = st.number_input("REVEL score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        with col3:
            af    = st.number_input("Frecuencia alélica (AF)", min_value=0.0, max_value=1.0,
                                    value=0.0001, step=0.0001, format="%.6f")
        name_input = st.text_input("Identificador de variante (opcional)", "NM_007294.4:c.5266dupC")

        if st.button("🔬 Predecir"):
            X_input = np.array([[cadd, revel, af]])
            prob    = model.predict_proba(X_input)[0][1]
            prio    = assign_priority(prob)
            icon    = PRIORITY_COLOR[prio]

            st.divider()
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Probabilidad de patogenicidad", f"{prob:.4f}")
            col_r2.metric("Nivel de prioridad", f"{icon} {prio}")
            col_r3.metric("Variante", name_input)

            # Gauge simple
            fig, ax = plt.subplots(figsize=(4, 0.5))
            ax.barh(0, prob, color="#e74c3c" if prio == "ALTA"
                    else "#f39c12" if prio == "MEDIA" else "#27ae60", height=0.4)
            ax.barh(0, 1 - prob, left=prob, color="#ecf0f1", height=0.4)
            ax.axvline(THRESHOLD_HIGH,   color="red",    linestyle="--", linewidth=0.8)
            ax.axvline(THRESHOLD_MEDIUM, color="orange", linestyle="--", linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("P(patogénica)")
            ax.set_title(f"Probabilidad: {prob:.4f}")
            st.pyplot(fig)
            plt.close()

    # =========================================================================
    # TAB 3 — Resultados
    # =========================================================================
    with tab_results:
        st.markdown("### Tabla de priorización de VUS")

        if "result_df" not in st.session_state:
            # Intenta cargar el CSV pre-generado por el módulo 4
            result_path = Path("data") / "vus_prioritized.csv"
            if result_path.exists():
                st.session_state["result_df"] = pd.read_csv(result_path)
            else:
                st.info("No hay resultados aún. Sube un archivo VCF o ejecuta el módulo 4.")
                st.stop()

        result = st.session_state["result_df"].copy()
        result["priority"] = result["priority"].map(
            lambda p: f"{PRIORITY_COLOR.get(p, '')} {p}"
        )

        # Filtros
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            prio_filter = st.multiselect(
                "Filtrar por prioridad",
                ["🔴 ALTA", "🟡 MEDIA", "🟢 BAJA"],
                default=["🔴 ALTA", "🟡 MEDIA", "🟢 BAJA"],
            )
        with col_f2:
            prob_min = st.slider("Probabilidad mínima", 0.0, 1.0, 0.0, 0.01)

        mask = (
            result["priority"].isin(prio_filter) &
            (result["prob_pathogenic"] >= prob_min)
        )
        filtered = result[mask]

        # Métricas rápidas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total VUS", len(result))
        m2.metric("🔴 Prioridad Alta",  (result["priority"] == "🔴 ALTA").sum())
        m3.metric("🟡 Prioridad Media", (result["priority"] == "🟡 MEDIA").sum())
        m4.metric("🟢 Prioridad Baja",  (result["priority"] == "🟢 BAJA").sum())

        st.dataframe(
            filtered.style.background_gradient(
                subset=["prob_pathogenic"], cmap="RdYlGn_r"
            ),
            use_container_width=True,
            height=500,
        )

        # Descarga CSV
        csv_bytes = filtered.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Descargar resultados CSV",
            data=csv_bytes,
            file_name="vus_prioritizadas.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
