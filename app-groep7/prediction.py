from pathlib import Path
import numpy as np

MODEL_FILE = Path(__file__).parent / "model.longkanker.keras"

FEATURES = [
    "leeft",
    "gesl",
    "tumsoort",
    "org_chir",
    "chemo"
]

def load_model():
    import keras
    return keras.models.load_model(str(MODEL_FILE), compile=False)

def preprocess(x):
    return x

def make_input(patient, org_chir=None, chemo=None):
    values = dict(patient)
    if org_chir is not None:
        values["org_chir"] = org_chir
    if chemo is not None:
        values["chemo"] = chemo
        
    for f in FEATURES:
        if f not in values or values[f] is None:
            raise ValueError(f"Geen waarde gevonden voor feature '{f}' bij patiënt {values.get('display_name')}.")
            
    raw = np.array([[float(values[f]) for f in FEATURES]])
    return preprocess(raw)

def check_model(model, X):
    try:
        out = np.asarray(model.predict(X, verbose=0))
    except Exception as error:
        return f"Het model accepteert geen invoer van {len(FEATURES)} variabelen. (Fout: {error})"
    if out.shape != (len(X), 1):
        return f"Verwachte outputvorm is ({len(X)}, 1), gekregen: {out.shape}"
    if not np.all((out >= 0) & (out <= 1)):
        return "De modeluitvoer valt niet netjes tussen 0 en 1."
    return None

def get_prediction(model, patient, org_chir, chemo):
    x = make_input(patient, org_chir, chemo)
    return float(model.predict(x, verbose=0)[0, 0])


if __name__ == "__main__":
    # Testscript voor prediction.py
    sample_patient = {
        "display_name": "Test Patiënt",
        "leeft": 65,
        "gesl": 1,
        "tumsoort": 302310,
        "org_chir": 0,
        "chemo": 0
    }
    if MODEL_FILE.exists():
        m = load_model()
        print("Model geladen.")
        pred = get_prediction(m, sample_patient, org_chir=0, chemo=0)
        print(f"Voorspelde kans op vit_stat=1 (met chirurgie): {pred:.3f}")
    else:
        print("model.keras nog niet gevonden. Voer eerst model_longkanker.py uit.")