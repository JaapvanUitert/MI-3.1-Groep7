from pathlib import Path

import keras
import numpy as np
import sqlite3
import pandas as pd

# Volgorde van de acht inputs:
# num_pregnancies, glucose, blood_pressure_dias, skin_thickness,
# insulin, BMI, diab_pedigree, age

## met de data base inlezen

## variabelen in tabel dat voorspelt wilt worden
RANDOM_SEED = 42
TARGET = "diab_pedigree"

## database inlezen
path = Path(__file__).resolve().parent
DB_FILE = path / "ehr.db"
MODEL_FILE = path / "model.keras"

if not DB_FILE.exists():
    raise SystemExit("ehr.db was not found. Run python setup_database.py first.")
with sqlite3.connect(DB_FILE) as conn:mds = pd.read_sql_query("SELECT * FROM patient ORDER BY id", conn)

## database checken
print(f"First rows:\n {mds.head()}")
print("Rows and columns:", mds.shape)
print("Column names:", mds.columns)

mds.drop(columns='id', inplace=True)
mds.drop(columns='display_name', inplace=True)

print("Column names:", mds.columns)
print("Rows and columns:", mds.shape)

features = mds.columns.drop(TARGET)
data = mds[features].apply(pd.to_numeric, errors="coerce")

print(f"A priori kans op diabetes: \n {mds[TARGET].mean()}")

X = mds[features].to_numpy(dtype="float32")
Y = mds[TARGET].to_numpy(dtype="float32").reshape(-1, 1)

## train/test split 
idx = np.random.default_rng(RANDOM_SEED).permutation(len(data))
n_train = int(2 / 3 * len(mds))
train_idx, test_idx = idx[:n_train], idx[n_train:]
X_train, y_train = X[train_idx], Y[train_idx]
X_test, y_test = X[test_idx], Y[test_idx]


## bouw model 

keep = [i for i, feature in enumerate(features) if feature != TARGET]
selector_weights = np.eye(len(features), dtype="float32")[:, keep]

normalizer = keras.layers.Normalization(
        axis=-1, name="embedded_normalization"
    )

model = keras.Sequential([
        keras.Input(shape=(len(features),), dtype="float32"),
        keras.layers.Dense(
            len(keep),
            use_bias=False,
            trainable=False,
            kernel_initializer=keras.initializers.Constant(selector_weights),
            name="remove_target_from_inputs",
        ),
        normalizer,
        keras.layers.Dense(8, activation="sigmoid"),
        keras.layers.Dense(
            1,
            activation="sigmoid",
            name="bounded_pedigree_estimate",
        ),
    ], name="sequential_pedigree_demo")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=min(16, len(X_train)),
    verbose=0,
)

predicted = model.predict(X_test, verbose=1)
mean_absolute_error= float(np.mean(np.abs(predicted - y_test)))
rooted_mean_squared_error = float(np.sqrt(np.mean((predicted - y_test) ** 2)))
baseline_mean_absolute_error = float(np.mean(np.abs(y_test - y_train.mean())))
print(f"Test MAE: {mean_absolute_error:.4f}")
print(f"Test RMSE: {rooted_mean_squared_error:.4f}")
print(f"Training-mean baseline test MAE: {baseline_mean_absolute_error:.4f}")
print(pd.DataFrame({
    "actual_pedigree": y_test.ravel(),
    "predicted_pedigree": predicted.ravel(),
    }).to_string(index=False))

model_file = Path('model.keras')
model.save(str(model_file))

model.summary()
print(f"Opgeslagen: {model_file.resolve()}")



# ## zonder de database inlezen
# reference_mean = np.array(
#     [3.0, 120.0, 70.0, 25.0, 80.0, 32.0, 0.5, 33.0],
#     dtype="float32",
# )

# reference_std = np.array(
#     [3.0, 30.0, 15.0, 12.0, 80.0, 7.0, 0.3, 12.0],
#     dtype="float32",
# )

# # Acht ruwe invoerwaarden per patiënt.
# inputs = keras.Input(
#     shape=(8,),
#     dtype="float32",
#     name="patient_features",
# )

# # Normalisatie wordt in model.keras opgeslagen.
# normalized = keras.layers.Normalization(
#     axis=-1,
#     mean=reference_mean,
#     variance=reference_std ** 2,
#     name="embedded_normalization",
# )(inputs)

# # Eén demoscore tussen 0 en 1.
# score_layer = keras.layers.Dense(
#     units=1,
#     activation="sigmoid",
#     name="demo_score",
# )

# outputs = score_layer(normalized)

# model = keras.Model(
#     inputs=inputs,
#     outputs=outputs,
#     name="unvalidated_demo_model",
# )

# # Handgekozen gewichten: NIET geleerd uit trainingsdata.
# weights = np.array(
#     [0.20, 0.90, 0.20, 0.10, 0.05, 0.50, 0.25, 0.30],
#     dtype="float32",
# ).reshape(8, 1)

# bias = np.array([-0.60], dtype="float32")

# score_layer.set_weights([weights, bias])

# # Bij uitvoering: opslaan in de huidige werkmap.
# model_file = Path('model.keras')
# model.save(str(model_file))

# model.summary()
# print(f"Opgeslagen: {model_file.resolve()}")