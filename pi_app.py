from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
from mel_spectrogram import mel_spectrogram
from grad_cam import GradCam
from six.moves import queue
import matplotlib.pyplot as plt
import pyaudio


SCALER_FILE_PATH = 'model/scaler.joblib'
MODEL_FILE_PATH = 'model/Epoch-261_Val-0.000.hdf5'

WATCH_CONV_ACT_LAYER_NAME = 'activation_6'
WATCH_CLASSIFIER_LAYER_NAMES = [
    'global_average_pooling2d',
    'dense'
]

DANGER_THRESHOLD = 0.6
MICS_SAM_RATE = 44100
MICS_CHANNELS = 2
MICS_DEVICE_ID = 0


class MicStream(object):
    def __init__(self, rate, frames_per_buffer, device_id, channels):
        self._rate = rate
        self._frames_per_buffer = frames_per_buffer
        self._buff = queue.Queue()
        self._device_id = device_id
        self._channels = channels
        self.closed = True
        super().__init__()

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._channels, rate=self._rate,
            input=True, frames_per_buffer=self._frames_per_buffer,
            input_device_index=self._device_id,
            stream_callback=self._fill_buffer
        )

        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._audio_stream.stop_stream()
        self._audio_stream.close()

        self.closed = False
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self, out_size):
        prev_chuck = np.empty((self._frames_per_buffer, 2), dtype=np.int16)

        while not self.closed:
            current_chunk = self._buff.get()
            if current_chunk is None:
                continue

            current_chunk = np.frombuffer(current_chunk, dtype=np.int16)\
                .reshape(self._frames_per_buffer, 2)
            appended = np.vstack([prev_chuck, current_chunk])

            if appended.shape[0] >= out_size:
                yield appended[:out_size]
                prev_chuck = appended[self._frames_per_buffer:]
            else:
                prev_chuck = appended


def vote_from_cam(data_0, data_1, _cam):
    argmax_x, argmax_y = np.unravel_index(_cam.argmax(), _cam.shape)
    x_block_size = data_0.shape[0] // cam.shape[0]
    y_block_size = data_0.shape[1] // cam.shape[1]

    data_0_vote, data_1_vote = 0, 0

    for i in range(x_block_size * argmax_x, x_block_size * argmax_x + 1):
        for j in range(y_block_size * argmax_y, y_block_size * argmax_y + 1):
            if data_0[i][j] > data_1[i][j]:
                data_0_vote += 1
            elif data_0[i][j] < data_1[i][j]:
                data_1_vote += 1

    return 0 if data_0_vote > data_1_vote else 1


if __name__ == '__main__':
    scaler: StandardScaler = load(SCALER_FILE_PATH)
    model = load_model(MODEL_FILE_PATH)
    cam_generator = GradCam(model, WATCH_CONV_ACT_LAYER_NAME, WATCH_CLASSIFIER_LAYER_NAMES)

    with MicStream(MICS_SAM_RATE, MICS_SAM_RATE//2) as stream:
        audio_generator = stream.generator(out_size=MICS_SAM_RATE)

        for data in audio_generator:
            left_data, right_data = data[:, 0], data[:, 1]

            left_mel = mel_spectrogram(left_data, MICS_SAM_RATE)
            right_mel = mel_spectrogram(right_data, MICS_SAM_RATE)

            X = scaler.transform(np.expand_dims(np.ravel(left_mel), 0))\
                .reshape(1, 99, 90, 1)

            pred = model.predict(X)

            if pred[0][0] > DANGER_THRESHOLD:
                cam = cam_generator.get_gradcam(X)

                vote_result = vote_from_cam(left_mel, right_mel, cam)
                if vote_result == 0:
                    print('LEFT DANGER')
                else:
                    print('RIGHT DANGER')
            else:
                print('SAFE')
