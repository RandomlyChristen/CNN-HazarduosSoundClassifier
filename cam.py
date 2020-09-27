import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class Cam(object):
    def __init__(self, model, last_convact_idx, last_fc_idx):
        self._get_output = keras.backend.function([model.layers[0].input],
                                                  [model.layers[last_convact_idx].output,
                                                   model.layers[last_fc_idx].output])
        self._mid_shape = model.layers[last_convact_idx].output_shape
        self._class_weights = model.layers[last_fc_idx].get_weights()[0]

    def get_cam(self, X):
        [conv_outputs, predictions] = self._get_output(X)
        output = []
        for num, idx in enumerate(np.argmax(predictions, axis=1)):
            cam = tf.matmul(np.expand_dims(self._class_weights[:, idx], axis=0),
                            np.transpose(np.reshape(conv_outputs[num],
                                                    (self._mid_shape[1] * self._mid_shape[2], self._mid_shape[3]))))
            cam = tf.keras.backend.eval(cam).reshape(self._mid_shape[1], self._mid_shape[2])
            output.append(cam)

        return output
