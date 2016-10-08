from __future__ import print_function
import csv
import h5py
import numpy as np

def encode_direction(direction):
    return {
            "up": [1, 0, 0, 1],
            "straight": [0, 1, 0, 0],
            "left": [0, 0, 1, 0],
            "right": [0, 0, 0, 1]
            }[direction]

# Dataset de faces: 624 itens, cada um com 960 atributos (pixels).

f = h5py.File("faces.h5", "w")

# Preciso modificar isso para criar datasets de treino/validação.
f.create_dataset("data", (624, 960), dtype="i8")
f.create_dataset("label", (624, 4), dtype="f4")

# Abre o CSV, lê cada linha e popula f['data'] e f['label'].
with open("faces.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        f["data"][i] = row[0:-2]
        f["label"][i] = encode_direction(row[-1])
        i = i + 1
