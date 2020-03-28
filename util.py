import numpy as np
import scipy.io.wavfile


def write_audio(filename, data, sample_rate=48000):
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)

    # Figure out the number of channels
    shape = data.shape
    if len(shape) > 3:
        raise ValueError("Data shape not understood. Not single or multi-channel.")
    if len(data.shape) > 1 and shape[0] < shape[1]:
        data = data.T

    if data.dtype == float:
        data = (data * (0.99 * 2.0 ** 15)).astype("int16")
    scipy.io.wavfile.write(filename, sample_rate, data)
