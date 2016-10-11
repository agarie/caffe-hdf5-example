from __future__ import print_function
import csv
import h5py
import numpy as np


def encode_pixel(pixel_value):
    return float(pixel_value) / 256.0

def encode_direction(direction):
    return {"up": 0,
            "straight": 1,
            "left": 2,
            "right": 3,
            }[direction]


def data_matrix_from_csv(csv_filename):
    X = np.zeros((624, 961))
    with open(csv_filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        next(reader)
        for row in reader:
            X[i, :] = [encode_pixel(x) for x in row[0:-1]] + [encode_direction(row[-1])]
            i = i + 1
    return X


def data_and_labels(data_matrix):
    np.random.shuffle(data_matrix)
    data = data_matrix[:, 0:-1]
    labels = data_matrix[:, -1]
    return data, labels


def save_dataset_with(filename, data, labels):
    f = h5py.File(filename, "w")
    f.create_dataset("data", data.shape, dtype="f8")
    f.create_dataset("label", labels.shape, dtype="i4")
    f["data"][:] = data.astype("f8")
    f["label"][:] = labels.astype("i4")
    f.close

np.random.seed(42)
data, labels = data_and_labels(data_matrix_from_csv("faces.csv"))
save_dataset_with("faces-train.h5", data[0:260], labels[0:260])
save_dataset_with("faces-test.h5", data[260:], labels[260:])
