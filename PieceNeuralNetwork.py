import numpy as np
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import numpy
from keras.models import load_model

def build_model(conv_size, conv_depth):
  board3d = layers.Input(shape=(14, 8, 8))

  # adding the convolutional layers
  x = board3d
  for _ in range(conv_depth):
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, 'relu')(x)
  x = layers.Dense(12, 'softmax')(x)

  return models.Model(inputs=board3d, outputs=x)

def build_model_residual(conv_size, conv_depth):
  board3d = layers.Input(shape=(19, 8, 8))

  # adding the convolutional layers
  x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_last')(board3d)
  for _ in range(conv_depth):
    previous = x
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, previous])
    x = layers.Activation('relu')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(1, 'sigmoid')(x)

  return models.Model(inputs=board3d, outputs=x)

def get_dataset():

  container0 = numpy.load('games/Lichess0.npz')
  container1 = numpy.load('games/Lichess1.npz')
  container2 = numpy.load('games/Lichess2.npz')
  container3 = numpy.load('games/Lichess3.npz')
  container4 = numpy.load('games/Lichess4.npz')
  container5 = numpy.load('games/Lichess5.npz')
  container6 = numpy.load('games/Lichess6.npz')
  container7 = numpy.load('games/Lichess7.npz')
  container8 = numpy.load('games/Lichess8.npz')
  container9 = numpy.load('games/Lichess9.npz')
  container10 = numpy.load('games/Lichess10.npz')
  container11 = numpy.load('games/Lichess11.npz')
  container12 = numpy.load('games/Lichess12.npz')
  container13 = numpy.load('games/Lichess13.npz')
  container14 = numpy.load('games/Lichess14.npz')
  container15 = numpy.load('games/Lichess15.npz')
  container16 = numpy.load('games/Lichess16.npz')
  container17 = numpy.load('games/Lichess17.npz')
  container18 = numpy.load('games/Lichess18.npz')
  container19 = numpy.load('games/Lichess19.npz')

  b = np.concatenate((container0['arr_0'], container1['arr_0'], container2['arr_0'], container3['arr_0'],
                      container4['arr_0'], container5['arr_0'], container6['arr_0'], container7['arr_0'], container8['arr_0'],
                      container9['arr_0']))
  v = np.concatenate((container0['arr_1'][:, 1].astype(int), container1['arr_1'][:, 1].astype(int), container2['arr_1'][:, 1].astype(int), container3['arr_1'][:, 1].astype(int),
                      container4['arr_1'][:, 1].astype(int), container5['arr_1'][:, 1].astype(int), container6['arr_1'][:, 1].astype(int), container7['arr_1'][:, 1].astype(int), container8['arr_1'][:, 1].astype(int),
                      container9['arr_1'][:, 1].astype(int)))
  return b, v

def train():
  x_train, y_train = get_dataset()
  print(x_train.shape)
  print(y_train.shape)

  model = build_model(32, 4)
  model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
  model.summary()
  model.fit(x_train, y_train,
            batch_size=2048,
            epochs=100,
            verbose=1,
            validation_split=0.1,
            callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                       callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])

  model.save('Piece')

def test():
  model = load_model('Piece')
  container0 = numpy.load('games/Lichess0.npz')
  lichess = container0['arr_0']
  for i in range (100):
    prediction = model.predict(lichess)
    print(prediction)

test()