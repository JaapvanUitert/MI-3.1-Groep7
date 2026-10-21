import sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np
import streamlit as st
import prediction
import math
import pandas as pd

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


# st.subheader("1. Vul patiëntgegevens in")
# stadium = st.radio(
#     "Wat is het stadium van uw longkanker?",
#     options=["I", "II", "III", "IV"],
#     index=None,
#     horizontal=True,
# )

# st.caption(
#     "Vul het stadium in dat uw behandelaar heeft vastgesteld. "
#     "Weet u het niet? Vraag dit dan na bij uw behandelaar."
# )

# leeftijd = st.number_input(
#     "Wat is uw leeftijd in jaren?",
#     min_value=0,
#     value=None,
#     step=1,
#     placeholder="Vul uw leeftijd in",
# )

# geslacht = st.radio(
#     "Wat is uw geslacht?",
#     options=["Vrouw", "Man", "Anders / onbekend"],
#     index=None,
#     horizontal=True,
# )

# if stadium is not None and leeftijd is not None and geslacht is not None:
#     patientgegevens = {
#         "stadium": stadium,
#         "leeftijd": int(leeftijd),
#         "geslacht": geslacht,
#     }

#     # Gebruik patientgegevens als invoer voor jullie voorspelmodel.
# else:
#     st.info("Vul alle patiëntgegevens in om verder te gaan.")

st.divider()

# Expliciete mapping: behandeloptie -> (org_chir, chemo)
behandelopties = {
    "Geen chirurgie of chemotherapie": (0, 0),
    "Alleen Orgaanchirurgie": (1, 0),
    "Alleen Chemotherapie": (0, 1),
    "Combinatie van Chirurgie en Chemotherapie": (1, 1),
}

st.subheader("2. Vergelijk behandelopties")
st.write(
    "Voorspeld overlijdensrisico per behandeloptie, "
    "gerangschikt van laag naar hoog."
)

resultaten = []

# Vier voorspellingen voor dezelfde patiënt
for behandeling, (org_chir, chemo) in behandelopties.items():
    kans = float(
        prediction.get_prediction(model, patient, org_chir, chemo)
    )

    if not math.isfinite(kans) or not 0 <= kans <= 1:
        st.error(f"Ongeldige modeluitkomst voor: {behandeling}")
        st.stop()

    resultaten.append({
        "Behandeloptie": behandeling,
        "Kans": kans,
        "Risicocategorie": (
            "Hoog risico" if kans >= 0.50 else "Laag risico"
        ),
    })

# Sorteer op de ongeronde voorspellingen
ranking = (
    pd.DataFrame(resultaten)
    .sort_values("Kans", ascending=True, kind="stable")
    .reset_index(drop=True)
)

ranking.insert(0, "Positie", range(1, len(ranking) + 1))
ranking["Voorspeld overlijdensrisico"] = ranking["Kans"].map(
    lambda kans: f"{kans * 100:.1f}%"
)

st.dataframe(
    ranking[
        [
            "Positie",
            "Behandeloptie",
            "Voorspeld overlijdensrisico",
            "Risicocategorie",
        ]
    ],
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Dit is een vergelijking van modelvoorspellingen, "
    "geen behandeladvies. Een lagere voorspelling bewijst niet "
    "dat die behandeling het overlijdensrisico verlaagt."
)

st.divider()

# vragen voor de arts
st.subheader("4. Vragen voor uw arts")
st.write("Gebruik deze voorbeeldvragen ter voorbereiding op uw gesprek:")

st.markdown(f"""
* **Geschiktheid:** Is de optie *"{behandeling}"* geschikt voor mijn persoonlijke situatie?
* **Doel van de behandeling:** Is deze keuze gericht op genezing of op levensverlenging en kwaliteit van leven?
* **Bijwerkingen:** Welke bijwerkingen of hersteltijd kan ik verwachten bij deze behandeling?
""")

st.text_area(
    "Aanvullende eigen vragen of opmerkingen:",
    placeholder="Typ hier uw eigen vragen om mee te nemen naar de arts...",
    height=120
)