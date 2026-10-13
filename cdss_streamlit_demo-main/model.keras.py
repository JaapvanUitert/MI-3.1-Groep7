import random # willekeurige operaties
import numpy as np # wiskundige operaties
import pandas as pd # lezen van csv-bestanden
import matplotlib.pyplot as plt #plotten
import sqlite3 as sqlite3


## lees data
mds = pd.read_csv("iknl_long_syntheticdata.csv")

## check data
print(f"koppen van de data: \n {mds.head()}")
print(f"Aantal rijen en kolommen: \n{mds.shape}")
