# Como usar a camada HDF5Data no Caffe

O formato HDF5 foi projetado para armazenar e organizar datasets grandes, tendo
suporte a dados multidimensionais e metadados.  A camada `HDF5Data` é usada no
Caffe para se trabalhar com esse formato e este repositório apresenta um
exemplo em que um arquivo CSV é convertido para HDF5 para ser usado no
treinamento de uma rede neural.

## Preparação do dataset

O dataset original são várias imagens PGM (grayscale) de rostos de pessoas
olhando para quatro direções (up, straight, left, right) em resolução 32x30.
Essas imagens foram convertidas em um arquivo CSV único (`faces.csv`), onde
cada pixel (32 * 30 = 960 no total) é guardado como uma coluna, além da direção
para a qual a pessoa está olhando.

O script `faces2hdf5.py` realiza a conversão de CSV para HDF5. O Caffe espera
que o arquivo possua um dataset com as classes (ou labels) e outro com os
valores de cada item. Neste caso, a função `save_dataset_with(filename, data, labels)`
é a responsável por de fato gerar os datasets:

~~~python
def save_dataset_with(filename, data, labels):
    f = h5py.File(filename, "w")
    f.create_dataset("data", data.shape, dtype="f8")
    f.create_dataset("label", labels.shape, dtype="i4")
    f["data"][:] = data.astype("f8")
    f["label"][:] = labels.astype("i4")
    f.close
~~~

## Definição da rede

A rede é definida em um arquivo `network.prototxt` o qual é referenciado pelo
`solver.prototxt`.

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

O arquivo `faces_train.txt` deve possuir o caminho (absoluto ou relativo) para
o arquivo `h5` do dataset. Por exemplo:

~~~
./faces-train.h5
~~~

A partir disso, o Caffe consegue recuperar dados e classes do arquivo HDF5 de
acordo com a política de treinamento definida no solver.

## Referências

1. https://en.wikipedia.org/wiki/Hierarchical_Data_Format
2. https://support.hdfgroup.org/HDF5/
