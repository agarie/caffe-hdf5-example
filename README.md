# Using the HDF5Data layer in Caffe


~~~
layer {
  name: "faces"
  type: "HDF5Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  hdf5_data_param {
    source: "faces_train.txt"
    batch_size: 64
  }
}
~~~
