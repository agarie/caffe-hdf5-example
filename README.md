# Como usar a camada HDF5Data no Caffe

A camada `HDF5Data` é usada quando o dataset está no formato `h5`.

## Preparação do dataset

## Definição da rede

A rede é definida em um arquivo `network.prototxt` o qual é referenciado pelo `solver.prototxt`.

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

O arquivo `faces_train.txt` deve possuir o caminho (absoluto ou relativo) para o arquivo `h5` do dataset. Por exemplo:

~~~
./faces-train.h5
~~~

## Referências

1. https://en.wikipedia.org/wiki/Hierarchical_Data_Format
2. https://support.hdfgroup.org/HDF5/
