import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (InputLayer, Conv2D, Activation,
                                     Dropout, Dense, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow as tf
from joblib import dump
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

CSV_FILE_PATH = 'data/mel_spectrogram.csv'
MODEL_FILE_PATH = 'model/Epoch-{epoch:03d}_Val-{val_loss:.3f}.hdf5'
SCALER_FILE_PATH = 'model/scaler.joblib'
EPOCHS = 500
BATCH_SIZE = 64
EARLY_STOP = 100
learning_rate = 0.0001


def build_model(input_shape):
    # http://sejong.dcollection.net/public_resource/pdf/200000175071_20200923012042.pdf
    # Fast Convolutional Neural Network Structure Design for Face Recognition On Raspberry Pi (2. 2019, Beak)
    layers = Sequential()
    layers.add(InputLayer(input_shape=input_shape, dtype='float32', name='mel_spectrogram_input'))

    layers.add(Conv2D(filters=64, kernel_size=3, strides=2))
    layers.add(Activation(activation='relu'))
    layers.add(Dropout(rate=0.2))

    layers.add(Conv2D(filters=64, kernel_size=3, strides=1))
    layers.add(Activation(activation='relu'))
    layers.add(Dropout(rate=0.2))

    layers.add(Conv2D(filters=128, kernel_size=3, strides=2))
    layers.add(Activation(activation='relu'))
    layers.add(Dropout(rate=0.2))

    layers.add(Conv2D(filters=128, kernel_size=3, strides=1))
    layers.add(Activation(activation='relu'))
    layers.add(Dropout(rate=0.2))

    layers.add(Conv2D(filters=256, kernel_size=3, strides=1))
    layers.add(Activation(activation='relu'))
    layers.add(Dropout(rate=0.2))

    layers.add(Conv2D(filters=512, kernel_size=3, strides=2))
    layers.add(Activation(activation='relu'))
    layers.add(Dropout(rate=0.2))

    layers.add(Conv2D(filters=512, kernel_size=3, strides=1))
    layers.add(Activation(activation='relu'))

    layers.add(GlobalAveragePooling2D())
    layers.add(Dense(2, activation='sigmoid'))

    return layers


csv_read = pd.read_csv(CSV_FILE_PATH, header=None)
X_origin = csv_read.values[:, :-1].astype('float32')
y_origin = csv_read.values[:, -1].astype('int32')

X_origin, y_origin \
    = RandomOverSampler(random_state=777).fit_sample(X_origin, y_origin)

y_origin = np.eye(np.unique(y_origin, axis=0).shape[0])[y_origin].astype('float32')

scaler = StandardScaler()
X_origin = scaler.fit_transform(X_origin).reshape(-1, 99, 90)
dump(scaler, SCALER_FILE_PATH)

X_origin = np.expand_dims(X_origin, -1)

model = build_model(input_shape=(99, 90, 1))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

checkpoint = ModelCheckpoint(filepath=MODEL_FILE_PATH, monitor='val_loss', verbose=1,
                             save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOP)

history = model.fit(
    X_origin, y_origin,
    epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True,
    callbacks=[checkpoint, early_stopping]
)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
