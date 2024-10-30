#  Copyright 2024 Hkxs
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the “Software”), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


import logging
from pathlib import Path
from typing import Tuple

import librosa
import librosa.feature
import numpy as np
import scipy

from lpc_vocoder.encode.pitch_estimation import pitch_estimator

logger = logging.getLogger(__name__)

# format
# frame size, sample rate, overlap, order, pitch, gain, data, pitch, gain, data
# if pitch == 0, then use noise as input

def preephasis(signal: np.ndarray) -> np.ndarray:
    return scipy.signal.lfilter([1, 0.9375], [1], signal)


def get_gain(frame: np.array, coefficients: np.array) -> float:
    rxx = librosa.autocorrelate(frame)
    return np.sqrt(np.dot(coefficients, rxx[:len(coefficients)]))


def _is_silence(signal: np.array) -> bool:
    """
    Check if the signal is silence, it uses -60dB as threshold, this is based on
    librosa.effects._signal_to_frame_nonsilent
    """
    rms = librosa.feature.rms(y=signal, frame_length=256, hop_length=256)
    power = librosa.core.amplitude_to_db(rms[..., 0, :], ref=np.max, top_db=None)
    is_silence = np.flatnonzero(power < -60)
    return is_silence.size > 0


def _process_frame(frame: np.array, sample_rate: float, order: int) -> Tuple[float, float, np.ndarray]:
    if _is_silence(frame):
        logger.debug("Silence found")
        lpc_coefficients = np.array([1] * 10)
        gain = 0
        pitch = 0
    else:
        pitch = pitch_estimator(frame, sample_rate)
        lpc_coefficients = librosa.lpc(frame, order=order)
        gain = get_gain(frame, lpc_coefficients)
    logger.debug(f"Pitch: {pitch}")
    logger.debug(f"Gain {gain}")
    return pitch, gain, lpc_coefficients


def encode_signal(filename: Path, window_size: int = None, overlap: int = 50, order: int = 10):
    sr = librosa.get_samplerate(str(filename))
    logger.debug(f"Sampling rate: {sr}")

    if not window_size:
        window_size = int(sr * 0.03)  # use a 30ms window

    overlap = window_size - int(overlap/100 * window_size)

    # window = librosa.filters.get_window('hamming', window_size)

    frames = librosa.stream(str(filename), block_length=1, frame_length=window_size, hop_length=overlap, mono=False)

    with open("file.txt", "w") as f:
        f.write(f"{256}, {sr}, {0}, {10}\n")
        for frame in frames:
            pitch, gain, a = _process_frame(frame, sr, 10)
            f.write(f"{pitch}, {gain}, {a.tobytes().hex()}\n")
