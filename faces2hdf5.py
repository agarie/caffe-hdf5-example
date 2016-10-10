from __future__ import print_function
import csv
import h5py
import numpy as np


def encode_direction(direction):
    return {"up": [1, 0, 0, 0],
            "straight": [0, 1, 0, 0],
            "left": [0, 0, 1, 0],
            "right": [0, 0, 0, 1]
            }[direction]


def data_and_labels():
    data = np.zeros((624, 960))
    labels = np.zeros((624, 4))
    with open("faces.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        next(reader)  # Header.
        for row in reader:
            data[i, :] = np.array([float(x) for x in row[0:-1]])
            labels[i, :] = encode_direction(row[-1])
            i = i + 1
    return data, labels


def save_dataset_with(filename, data, labels):
    f = h5py.File(filename, "w")
    f.create_dataset("data", data.shape, dtype="f8")
    f.create_dataset("label", labels.shape, dtype="f8")
    f["data"][:] = data.astype("f8")
    f["label"][:] = labels.astype("f8")
    f.close

data, labels = data_and_labels()
np.random.shuffle(data)
np.random.shuffle(labels)

save_dataset_with("faces-train.h5", data[0:261], labels[0:261])
save_dataset_with("faces-test.h5", data[261:], labels[261:])
