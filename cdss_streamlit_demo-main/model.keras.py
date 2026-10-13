import random # willekeurige operaties
import numpy as np # wiskundige operaties
import pandas as pd # lezen van csv-bestanden
import matplotlib.pyplot as plt #plotten
import sqlite3 as sqlite3
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


## lees data
mds = pd.read_csv("iknl_long_syntheticdata.csv", sep=";", skipinitialspace=True, na_values=["\\N", ""])
mds.columns = mds.columns.str.strip()

## check data
print(f"koppen van de data: \n {mds.head()}")
print(f"Aantal rijen en kolommen: \n{mds.shape}")



## identifiers, overlevingstijd en constante kolom verwijderen
mds = mds.drop(columns=["key_nkr", "key_eid", "vit_stat_int", "topo"])
mds = mds.rename(columns={"vit_stat": "died"})

## kolommen met >25% missings weg, daarna rijen met missings weg
missing = mds.isna().mean()
mds = mds.drop(columns=missing[missing > 0.25].index)
mds = mds.dropna()


cat_cols = ["tumsoort", "diag_basis", "topo_sublok", "morf", "ct", "cn",
            "stadium", "cstadium", "uitgebr_chir_code", "later", "diffgrad",
            "her2_stat", "swk", "swk_uitslag", "chemo", "target", "rt",
            "imm", "multifoc"]
mds = pd.get_dummies(mds, columns=cat_cols, dtype=float)

## normaliseren
features = mds.columns.drop("died")
mds[features] = (mds[features] - mds[features].min()) / (mds[features].max() - mds[features].min())

print(mds.shape)
print("A priori kans op overlijden:", mds["died"].mean())




## train/test split (2/3 - 1/3)
np.random.seed(42)
idx = np.random.permutation(len(mds))
n_train = int(2/3 * len(mds))
train, test = mds.iloc[idx[:n_train]], mds.iloc[idx[n_train:]]

X_train, y_train = train.drop("died", axis=1), train["died"]
X_test,  y_test  = test.drop("died", axis=1),  test["died"]


## model trainen
nn = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000, random_state=42)
nn.fit(X_train, y_train)

## voorspellen
p = nn.predict_proba(X_test)[:, 1]
pred = (p > 0.5).astype(int)

## evalueren
print(confusion_matrix(y_test, pred))
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.3f}  |  Error rate: {1 - acc:.3f}")
