import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


# https://keras.io/examples/vision/grad_cam/
class GradCam(object):
    def __init__(self, model, last_conv_layer_name, classifier_layer_names):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        self._last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        self._classifier_model = keras.Model(classifier_input, x)

    def get_gradcam(self, X):
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = self._last_conv_layer_model(X)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = self._classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        # heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap
