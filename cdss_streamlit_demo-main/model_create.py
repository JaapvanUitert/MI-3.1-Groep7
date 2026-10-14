from pathlib import Path

import os
import keras
import numpy as np

# Volgorde van de acht inputs:
# num_pregnancies, glucose, blood_pressure_dias, skin_thickness,
# insulin, BMI, diab_pedigree, age

os.chdir(os.path.join(os.path.abspath(os.path.curdir),u'cdss_streamlit_demo-main'))

reference_mean = np.array(
    [3.0, 120.0, 70.0, 25.0, 80.0, 32.0, 0.5, 33.0],
    dtype="float32",
)

reference_std = np.array(
    [3.0, 30.0, 15.0, 12.0, 80.0, 7.0, 0.3, 12.0],
    dtype="float32",
)

# Acht ruwe invoerwaarden per patiënt.
inputs = keras.Input(
    shape=(8,),
    dtype="float32",
    name="patient_features",
)

# Normalisatie wordt in model.keras opgeslagen.
normalized = keras.layers.Normalization(
    axis=-1,
    mean=reference_mean,
    variance=reference_std ** 2,
    name="embedded_normalization",
)(inputs)

# Eén demoscore tussen 0 en 1.
score_layer = keras.layers.Dense(
    units=1,
    activation="sigmoid",
    name="demo_score",
)

outputs = score_layer(normalized)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="unvalidated_demo_model",
)

# Handgekozen gewichten: NIET geleerd uit trainingsdata.
weights = np.array(
    [0.20, 0.90, 0.20, 0.10, 0.05, 0.50, 0.25, 0.30],
    dtype="float32",
).reshape(8, 1)

bias = np.array([-0.60], dtype="float32")

score_layer.set_weights([weights, bias])

# Bij uitvoering: opslaan in de huidige werkmap.
model_file = Path('model.keras')
model.save(str(model_file))

model.summary()
print(f"Opgeslagen: {model_file.resolve()}")