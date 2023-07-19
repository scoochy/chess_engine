import numpy as np
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import numpy
from keras.models import load_model

def build_model(conv_size, conv_depth):
  board3d = layers.Input(shape=(12, 8, 8))

  # adding the convolutional layers
  x = board3d
  for _ in range(conv_depth):
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, 'relu')(x)
  x = layers.Dense(1, 'sigmoid')(x)

  return models.Model(inputs=board3d, outputs=x)

def build_model_residual(conv_size, conv_depth):
  board3d = layers.Input(shape=(12, 8, 8))

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
    container0 = numpy.load('games/EarlyPosition0.npz')
    container1 = numpy.load('games/EarlyPosition1.npz')
    container2 = numpy.load('games/EarlyPosition2.npz')
    container3 = numpy.load('games/EarlyPosition3.npz')
    container4 = numpy.load('games/EarlyPosition4.npz')
    container5 = numpy.load('games/EarlyPosition5.npz')
    container6 = numpy.load('games/EarlyPosition6.npz')
    container7 = numpy.load('games/EarlyPosition7.npz')
    container8 = numpy.load('games/EarlyPosition8.npz')
    container9 = numpy.load('games/EarlyPosition9.npz')
    container10 = numpy.load('games/EarlyPosition10.npz')
    container11 = numpy.load('games/EarlyPosition11.npz')
    container12 = numpy.load('games/EarlyPosition12.npz')
    container13 = numpy.load('games/EarlyPosition13.npz')
    container14 = numpy.load('games/EarlyPosition14.npz')
    container15 = numpy.load('games/EarlyPosition15.npz')
    container16 = numpy.load('games/EarlyPosition16.npz')
    container17 = numpy.load('games/EarlyPosition17.npz')
    container18 = numpy.load('games/EarlyPosition18.npz')
    container19 = numpy.load('games/EarlyPosition19.npz')
    container20 = numpy.load('games/EarlyPosition20.npz')
    container21 = numpy.load('games/EarlyPosition21.npz')
    container22 = numpy.load('games/EarlyPosition22.npz')
    container23 = numpy.load('games/EarlyPosition23.npz')
    container24 = numpy.load('games/EarlyPosition24.npz')
    container25 = numpy.load('games/EarlyPosition25.npz')
    container26 = numpy.load('games/EarlyPosition26.npz')
    container27 = numpy.load('games/EarlyPosition27.npz')
    container28 = numpy.load('games/EarlyPosition28.npz')
    container29 = numpy.load('games/EarlyPosition29.npz')
    container30 = numpy.load('games/EarlyPosition30.npz')
    container31 = numpy.load('games/Position31.npz')
    container32 = numpy.load('games/Position32.npz')
    container33 = numpy.load('games/Position33.npz')
    container34 = numpy.load('games/Position34.npz')
    container35 = numpy.load('games/Position35.npz')


    eval_container0 = numpy.load('games/EarlyPosition0Eval.npz')
    eval_container1 = numpy.load('games/EarlyPosition1Eval.npz')
    eval_container2 = numpy.load('games/EarlyPosition2Eval.npz')
    eval_container3 = numpy.load('games/EarlyPosition3Eval.npz')
    eval_container4 = numpy.load('games/EarlyPosition4Eval.npz')
    eval_container5 = numpy.load('games/EarlyPosition5Eval.npz')
    eval_container6 = numpy.load('games/EarlyPosition6Eval.npz')
    eval_container7 = numpy.load('games/EarlyPosition7Eval.npz')
    eval_container8 = numpy.load('games/EarlyPosition8Eval.npz')
    eval_container9 = numpy.load('games/EarlyPosition9Eval.npz')
    eval_container10 = numpy.load('games/EarlyPosition10Eval.npz')
    eval_container11 = numpy.load('games/EarlyPosition11Eval.npz')
    eval_container12 = numpy.load('games/EarlyPosition12Eval.npz')
    eval_container13 = numpy.load('games/EarlyPosition13Eval.npz')
    eval_container14 = numpy.load('games/EarlyPosition14Eval.npz')
    eval_container15 = numpy.load('games/EarlyPosition15Eval.npz')
    eval_container16 = numpy.load('games/EarlyPosition16Eval.npz')
    eval_container17 = numpy.load('games/EarlyPosition17Eval.npz')
    eval_container18 = numpy.load('games/EarlyPosition18Eval.npz')
    eval_container19 = numpy.load('games/EarlyPosition19Eval.npz')
    eval_container20 = numpy.load('games/EarlyPosition20Eval.npz')
    eval_container21 = numpy.load('games/EarlyPosition21Eval.npz')
    eval_container22 = numpy.load('games/EarlyPosition22Eval.npz')
    eval_container23 = numpy.load('games/EarlyPosition23Eval.npz')
    eval_container24 = numpy.load('games/EarlyPosition24Eval.npz')
    eval_container25 = numpy.load('games/EarlyPosition25Eval.npz')
    eval_container26 = numpy.load('games/EarlyPosition26Eval.npz')
    eval_container27 = numpy.load('games/EarlyPosition27Eval.npz')
    eval_container28 = numpy.load('games/EarlyPosition28Eval.npz')
    eval_container29 = numpy.load('games/EarlyPosition29Eval.npz')
    eval_container30 = numpy.load('games/EarlyPosition30Eval.npz')
    eval_container31 = numpy.load('games/Position31Eval.npz')
    eval_container32 = numpy.load('games/Position32Eval.npz')
    eval_container33 = numpy.load('games/Position33Eval.npz')
    eval_container34 = numpy.load('games/Position34Eval.npz')
    eval_container35 = numpy.load('games/Position35Eval.npz')

    eval_arr0 = eval_container0['arr_0']
    eval_arr1 = eval_container1['arr_0']
    eval_arr2 = eval_container2['arr_0']
    eval_arr3 = eval_container3['arr_0']
    eval_arr4 = eval_container4['arr_0']
    eval_arr5 = eval_container5['arr_0']
    eval_arr6 = eval_container6['arr_0']
    eval_arr7 = eval_container7['arr_0']
    eval_arr8 = eval_container8['arr_0']
    eval_arr9 = eval_container9['arr_0']
    eval_arr10 = eval_container10['arr_0']
    eval_arr11 = eval_container11['arr_0']
    eval_arr12 = eval_container12['arr_0']
    eval_arr13 = eval_container13['arr_0']
    eval_arr14 = eval_container14['arr_0']
    eval_arr15 = eval_container15['arr_0']
    eval_arr16 = eval_container16['arr_0']
    eval_arr17 = eval_container17['arr_0']
    eval_arr18 = eval_container18['arr_0']
    eval_arr19 = eval_container19['arr_0']
    eval_arr20 = eval_container20['arr_0']
    eval_arr21 = eval_container21['arr_0']
    eval_arr22 = eval_container22['arr_0']
    eval_arr23 = eval_container23['arr_0']
    eval_arr24 = eval_container24['arr_0']
    eval_arr25 = eval_container25['arr_0']
    eval_arr26 = eval_container26['arr_0']
    eval_arr27 = eval_container27['arr_0']
    eval_arr28 = eval_container28['arr_0']
    eval_arr29 = eval_container29['arr_0']
    eval_arr30 = eval_container30['arr_0']
    eval_arr31 = eval_container31['arr_0']
    eval_arr32 = eval_container32['arr_0']
    eval_arr33 = eval_container33['arr_0']
    eval_arr34 = eval_container34['arr_0']
    eval_arr35 = eval_container35['arr_0']


    eval_arr0 = numpy.asarray(eval_arr0 / abs(eval_arr0).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr1 = numpy.asarray(eval_arr1 / abs(eval_arr1).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr2 = numpy.asarray(eval_arr2 / abs(eval_arr2).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr3 = numpy.asarray(eval_arr3 / abs(eval_arr3).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr4 = numpy.asarray(eval_arr4 / abs(eval_arr4).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr5 = numpy.asarray(eval_arr5 / abs(eval_arr5).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr6 = numpy.asarray(eval_arr6 / abs(eval_arr6).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr7 = numpy.asarray(eval_arr7 / abs(eval_arr7).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr8 = numpy.asarray(eval_arr8 / abs(eval_arr8).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr9 = numpy.asarray(eval_arr9 / abs(eval_arr9).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr10 = numpy.asarray(eval_arr10 / abs(eval_arr10).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr11 = numpy.asarray(eval_arr11 / abs(eval_arr11).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr12 = numpy.asarray(eval_arr12 / abs(eval_arr12).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr13 = numpy.asarray(eval_arr13 / abs(eval_arr13).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr14 = numpy.asarray(eval_arr14 / abs(eval_arr14).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr15 = numpy.asarray(eval_arr15 / abs(eval_arr15).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr16 = numpy.asarray(eval_arr16 / abs(eval_arr16).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr17 = numpy.asarray(eval_arr17 / abs(eval_arr17).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr18 = numpy.asarray(eval_arr18 / abs(eval_arr18).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr19 = numpy.asarray(eval_arr19 / abs(eval_arr19).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr20 = numpy.asarray(eval_arr20 / abs(eval_arr20).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr21 = numpy.asarray(eval_arr21 / abs(eval_arr21).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr22 = numpy.asarray(eval_arr22 / abs(eval_arr22).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr23 = numpy.asarray(eval_arr23 / abs(eval_arr23).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr24 = numpy.asarray(eval_arr24 / abs(eval_arr24).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr25 = numpy.asarray(eval_arr25 / abs(eval_arr25).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr26 = numpy.asarray(eval_arr26 / abs(eval_arr26).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr27 = numpy.asarray(eval_arr27 / abs(eval_arr27).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr28 = numpy.asarray(eval_arr28 / abs(eval_arr28).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr29 = numpy.asarray(eval_arr29 / abs(eval_arr29).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr30 = numpy.asarray(eval_arr30 / abs(eval_arr30).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr31 = numpy.asarray(eval_arr31 / abs(eval_arr31).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr32 = numpy.asarray(eval_arr32 / abs(eval_arr32).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr33 = numpy.asarray(eval_arr33 / abs(eval_arr33).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr34 = numpy.asarray(eval_arr34 / abs(eval_arr34).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)
    eval_arr35 = numpy.asarray(eval_arr31 / abs(eval_arr35).max() / 2 + 0.5,
                               dtype=numpy.float32)  # normalization (0 - 1)

    b = np.concatenate((container0['arr_0'], container1['arr_0'], container2['arr_0'], container3['arr_0'], container4['arr_0'], container5['arr_0'], container6['arr_0'], container7['arr_0'], container8['arr_0'], container9['arr_0'], container10['arr_0'], container11['arr_0'],
                        container12['arr_0'], container13['arr_0'], container14['arr_0'], container15['arr_0'], container16['arr_0'], container17['arr_0'], container18['arr_0'],
                        container19['arr_0'], container20['arr_0'], container21['arr_0'], container22['arr_0'], container23['arr_0'], container24['arr_0'], container25['arr_0'], container26['arr_0'], container27['arr_0'], container28['arr_0'], container29['arr_0'], container30['arr_0']))
    v = np.concatenate((eval_arr0, eval_arr1, eval_arr2, eval_arr3, eval_arr4, eval_arr5, eval_arr6, eval_arr7, eval_arr8, eval_arr9, eval_arr10, eval_arr11, eval_arr12, eval_arr13, eval_arr14, eval_arr15, eval_arr16, eval_arr17, eval_arr18, eval_arr19, eval_arr20, eval_arr21, eval_arr22, eval_arr23, eval_arr24, eval_arr25, eval_arr26, eval_arr27, eval_arr28, eval_arr29, eval_arr30))
    return b , v

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

model.save('EarlyPosition')

