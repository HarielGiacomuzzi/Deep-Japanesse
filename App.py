import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import load_data

def train(X_train, X_test, Y_train, Y_test,
          weights_location='./checkpoint/model.h5',
          json_location='./checkpoint/model.json',
          checkpoint_location='./checkpoint/weights.hdf5',
          log_dir='./logs'):
    classes = 956
    img_rows, img_cols = 32, 32

    # prepara os vetores para trabalhar com o backend em uso
    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # converte os vetores de labels em matrizes de binarias de classes
    Y_train = keras.utils.to_categorical(Y_train, classes)
    Y_test = keras.utils.to_categorical(Y_test, classes)

    # cria modificacoes aleatorias na rotacao e zoom das imagens
    datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.25)
    datagen.fit(X_train)

    # cria o modelo
    model = keras.models.Sequential()

    # define as camadas da rede neural
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    # cria o callback para fazer os logs do tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                histogram_freq=1,
                                batch_size=32,
                                write_graph=True,
                                write_grads=True,
                                write_images=True,
                                embeddings_freq=0,
                                embeddings_layer_names=None,
                                embeddings_metadata=None)

    # cria o callback para salvar os pesos da rede a cada epoch
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_location,
                                          monitor='val_acc',
                                          mode='max',
                                          verbose=1,
                                          save_best_only=True)

    # compila o modelo e inicia o treinamento
    model.summary()

    with open(json_location, 'w+') as f:
        f.write(model.to_json())
        print "model saved !"

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.save_weights(weights_location)

    model.fit(X_train, Y_train,
              batch_size=256,
              epochs=300,
              verbose=1,
              validation_data=(X_test, Y_test),
              callbacks=[tensorboard_callback, checkpoint_callback])

    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":

    (X_train, X_test, Y_train, Y_test) = load_data.load()

    train(X_train, X_test, Y_train, Y_test)
