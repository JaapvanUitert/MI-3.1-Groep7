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

def encode_geslacht(geslacht):
    value = str(geslacht).strip().lower()

    if value == "man":
        return 1
    if value == "vrouw":
        return 2

def make_input(patient, leeftijd, geslacht, stadium, org_chir=None, chemo=None):
    values = dict(patient)

    if leeftijd is None or geslacht is None:
        raise ValueError("Leeftijd en geslacht moeten ingevuld zijn.")

    values["leeft"] = leeftijd
    values["gesl"] = encode_geslacht(geslacht)
    
    if org_chir is not None:
        values["org_chir"] = org_chir
    if chemo is not None:
        values["chemo"] = chemo
        
    numeric_values = []
    for feature in FEATURES:
        value = values.get(feature)
        if value is None:
            raise ValueError(
                f"Geen waarde gevonden voor feature '{feature}' "
                f"bij patiënt {values.get('display_name', values.get('id'))}."
            )

        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Feature '{feature}' moet numeriek zijn; gekregen: {value!r}."
            ) from error

        if not np.isfinite(number):
            raise ValueError(f"Feature '{feature}' bevat geen eindige waarde.")

        numeric_values.append(number)

    raw = np.array([numeric_values], dtype=np.float32)
    processed = np.asarray(preprocess(raw))
    if not np.all(np.isfinite(processed)):
        raise ValueError("De voorbewerkte modelinvoer bevat ongeldige waarden.")
    return processed

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

def get_prediction(model, patient, leeftijd, geslacht, stadium, org_chir, chemo):
    x = make_input(patient,leeftijd, geslacht, stadium, org_chir, chemo)
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
        pred = get_prediction(m, sample_patient, leeftijd = 55, geslacht= "man", stadium= "I", org_chir=0, chemo=0)
        print(f"Voorspelde kans op vit_stat=1 (met chirurgie): {pred:.3f}")
    else:
        print("model.keras nog niet gevonden. Voer eerst model_longkanker.py uit.")