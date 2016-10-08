from __future__ import print_function
import h5py
import numpy as np

# Dataset de faces: 624 itens, cada um com 960 atributos (pixels).

f = h5py.File("faces.h5", "w")
f.create_dataset("data", (), dtype="")