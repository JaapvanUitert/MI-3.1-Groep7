from pathlib import Path
import numpy as np
import pandas as pd
import keras

RANDOM_SEED = 42
TARGET = "vit_stat"
FEATURES = ["leeft", "gesl", "tumsoort", "org_chir", "chemo"]

PATH = Path(__file__).resolve().parent
DATA_FILE = PATH / "iknl_long_syntheticdata.csv"
MODEL_FILE = PATH / "model.longkanker.keras"

if not DATA_FILE.exists():
    raise SystemExit(f"Bestand niet gevonden: {DATA_FILE}")

# data inlezen en opschonen
print("- Data inlezen -")
df = pd.read_csv(DATA_FILE, sep=";")

# kolomnamen opschonen (spaties verwijderen)
df.columns = df.columns.str.strip()

# Omzetten naar numerieke waarden
df["vit_stat_int"] = pd.to_numeric(
    df["vit_stat_int"], errors="coerce"
)

# Alleen patiënten met vit_stat_int >= 365 behouden
n_before = len(df)
df = df.loc[df["vit_stat_int"] >= 365].copy()

print(f"Verwijderde patiënten: {n_before - len(df)}")
print(f"Patiënten na filtering: {len(df)}")

print(f"Aantal patiënten in dataset: {len(df)}")
print(f"A priori risico op overlijden (vit_stat = 1): {df[TARGET].mean():.2%}")

# zorg ervoor dat de features numeriek zijn en verwijder missende waarden
data = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()

X = data[FEATURES].to_numpy(dtype="float32")
Y = data[TARGET].to_numpy(dtype="float32").reshape(-1, 1)

# train / test Split
rng = np.random.default_rng(RANDOM_SEED)
idx = rng.permutation(len(data))
n_train = int(0.8 * len(data))

train_idx, test_idx = idx[:n_train], idx[n_train:]
X_train, y_train = X[train_idx], Y[train_idx]
X_test, y_test = X[test_idx], Y[test_idx]

# neuraal netwerk bouwen
normalizer = keras.layers.Normalization(axis=-1, name="embedded_normalization")
normalizer.adapt(X_train)

model = keras.Sequential([
    keras.Input(shape=(len(FEATURES),), dtype="float32"),
    normalizer,
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid", name="survival_probability")
], name="longkanker_risico_model")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc")
    ]
)

print("\n- Model Samenvatting -")
model.summary()

# model trainen
print("\n- Model Trainen -")
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# evalueren op testdata
print("\n- Test Evaluatie -")
test_results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss:     {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.2%}")
print(f"Test AUC:      {test_results[2]:.4f}")

# model opslaan
model.save(str(MODEL_FILE))
print(f"\nModel succesvol opgeslagen als: {MODEL_FILE.resolve()}")