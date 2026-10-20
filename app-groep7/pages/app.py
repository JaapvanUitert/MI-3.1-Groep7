import sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np
import streamlit as st
import prediction

DB_FILE = Path(__file__).parent.parent / "longkanker.db"

st.set_page_config(page_title="CDSS Longkanker Keuzehulp", layout="centered", initial_sidebar_state="collapsed")

st.title("CDSS Longkanker Keuzehulp")
st.caption("Vergelijk eenvoudig het effect van behandelopties op het voorspelde risico.")

if not DB_FILE.exists():
    st.error("ehr.db niet gevonden. Voer eerst `python setup_database.py` uit.")
    st.stop()
if not prediction.MODEL_FILE.exists():
    st.error("model.keras niet gevonden. Train eerst het neuraal netwerk via `python model_longkanker.py`.")
    st.stop()

# model+patienten inladen
@st.cache_resource(max_entries=1)
def get_model(mtime):
    return prediction.load_model()

def load_patients():
    with closing(sqlite3.connect(DB_FILE)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM patient ORDER BY id").fetchall()
    return {row["id"]: dict(row) for row in rows}

patients = load_patients()
if not patients:
    st.warning("Geen patiënten gevonden in de database.")
    st.stop()

try:
    model = get_model(prediction.MODEL_FILE.stat().st_mtime)
except Exception as error:
    st.error(f"Kan model.longkanker.keras niet laden: {error}")
    st.stop()

# patient selecteren
st.subheader("1. Selecteer patiënt")
patient_id = st.selectbox(
    "Kies een patiënt:",
    options=list(patients),
    format_func=lambda pid: patients[pid]["display_name"]
)
patient = patients[patient_id]

st.divider()

# behandelkeuze
st.subheader("2. Kies een behandeloptie")
behandeling_keuze = st.radio(
    "Welke behandeling wilt u overwegen?",
    options=[
        "Geen chirurgie of chemotherapie",
        "Alleen Orgaanchirurgie",
        "Alleen Chemotherapie",
        "Combinatie van Chirurgie en Chemotherapie"
    ],
    index=0
)

# naar binaire waarden voor het neuraal netwerk
org_chir = 1 if "Chirurgie" in behandeling_keuze else 0
chemo = 1 if "Chemotherapie" in behandeling_keuze else 0

st.divider()

# berekend risico
st.subheader("3. Berekend Risico")

# berekening via het model
prob_overlijden = prediction.get_prediction(model, patient, org_chir, chemo)
risico_label = "Hoog Risico" if prob_overlijden >= 0.50 else "Laag Risico"

st.metric(
    label="Risicocategorie", 
    value=risico_label, 
    delta=f"{prob_overlijden*100:.1f}% kans op overlijden", 
    delta_color="inverse"
)
st.progress(prob_overlijden)

st.divider()

# vragen voor de arts
st.subheader("4. Vragen voor uw arts")
st.write("Gebruik deze voorbeeldvragen ter voorbereiding op uw gesprek:")

st.markdown(f"""
* **Geschiktheid:** Is de optie *"{behandeling_keuze}"* geschikt voor mijn persoonlijke situatie?
* **Doel van de behandeling:** Is deze keuze gericht op genezing of op levensverlenging en kwaliteit van leven?
* **Bijwerkingen:** Welke bijwerkingen of hersteltijd kan ik verwachten bij deze behandeling?
""")

st.text_area(
    "Aanvullende eigen vragen of opmerkingen:",
    placeholder="Typ hier uw eigen vragen om mee te nemen naar de arts...",
    height=120
)