import struct
from PIL import Image
import numpy
from sklearn.model_selection import train_test_split
import scipy.misc
from tensorflow.python.lib.io import file_io

def load( test_split=0.2, dataset_location='../data/ETL8G/ETL8G_{:02d}', dict_location='../data/dict.csv' ):
    # variaveis auxiliares
    classes = 956
    img_rows, img_cols = 32, 32

    temp_img = []
    temp_labels = []

    new_number = 0
    dict = {}

    record_size = 8199  # this information was taken from the dataset definition
    num_records = 4780  # according to dataset definition

    X_train = numpy.zeros([classes * 160, img_rows, img_cols], dtype=numpy.float32)
    Y_train = numpy.zeros(classes * 160)

    # le os todos os samples de imagens e labels
    for dataset in range(1, 33):
        filename = dataset_location.format(dataset)
        with file_io.FileIO(filename, 'r') as file:
            for i in range(0, num_records):
                record_id = i
                file.seek(record_id * record_size)
                record_raw = file.read(record_size)
                record_raw = struct.unpack('>2H8sI4B4H2B30x8128s11x', record_raw)
                aux_image = Image.frombytes('F', (128, 127), record_raw[14], 'bit', 4)
                aux_image = aux_image.convert('L')
                record_image = Image.eval(aux_image, lambda x: 255 - x * 16)
                temp_img.append(record_image)
                temp_labels.append(record_raw[2])

    # redimensiona as imagens e transforma em uma matriz de valores de 0 a 255
    for index in range(0, len(temp_img)):
        X_train[index] = scipy.misc.imresize(temp_img[index], (img_rows, img_cols), mode='F')

    # normaliza os valores
    X_train = X_train.astype('float32')

    # converte todas as labels para valores numericos e guarda as correspondencias em um dicionario
    for i in range(0, len(temp_labels)):
        if dict.has_key(temp_labels[i]):
            Y_train[i] = dict[temp_labels[i]]
        else:
            dict[temp_labels[i]] = new_number
            Y_train[i] = new_number
            new_number += 1

    # salva a referencia das labels
    with file_io.FileIO(dict_location, 'w+') as file:
        file.write('KEY,VALUE\n')
        for (key, value) in dict.items():
            file.write('{},{}\n'.format(key,value))
        file.flush()
        file.close()

    return train_test_split(X_train, Y_train, test_size=test_split)
