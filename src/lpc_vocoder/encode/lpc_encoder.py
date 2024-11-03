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

from lpc_vocoder.utils.pitch_estimation import pitch_estimator
from lpc_vocoder.utils.utils import get_frame_gain, is_silence

logger = logging.getLogger(__name__)


class LpcEncoder:
    def __init__(self, order: int = 10):
        self._frames = None
        self.sample_rate = None
        self.order = order
        self.frame_data = []
        self.window_size = None
        self.overlap = None

    def load_data(self, data: np.array, sample_rate: int, window_size: int, overlap: int = 50) -> None:
        self._get_window_data(window_size, overlap)
        self._frames = librosa.util.frame(data.astype(np.float64), frame_length=self.window_size, hop_length=self.overlap, axis=0)
        self.sample_rate = sample_rate

    def load_file(self, filename: Path, window_size: int, overlap: int = 50):
        self.sample_rate = librosa.get_samplerate(str(filename))
        self._get_window_data(window_size, overlap)
        self._frames = librosa.stream(str(filename), block_length=1, frame_length=self.window_size,
                                hop_length=self.overlap, mono=False, dtype=np.float64)
        self._frames = list(self._frames)

    def _get_window_data(self, window_size, overlap):
        self.window_size = window_size if window_size else int(self.sample_rate * 0.03)  # use a 30ms window
        self.overlap = self.window_size - int(overlap / 100 * self.window_size)  # calculate overlap in samples

    def encode_signal(self) -> None:
        # window = librosa.filters.get_window('hamming', window_size)
        for frame in self._frames:
            pitch, gain, coefficients = self._process_frame(frame)
            frame_data = {"pitch": pitch, "gain": gain, "coefficients": coefficients}
            self.frame_data.append(frame_data)

    def save_data(self, filename: Path) -> None:
        # format
        # frame size, sample rate, overlap, order, pitch, gain, data, pitch, gain, data
        with open(filename, "w") as f:
            f.write(f"{self.window_size}, {self.sample_rate}, {self.overlap}, {self.order}\n")
            for frame in self.frame_data:
                f.write(f"{frame['pitch']}, {frame['gain']}, {frame['coefficients'].tobytes().hex()}\n")

    def _process_frame(self, frame: np.array) -> Tuple[float, float, np.ndarray]:
        if is_silence(frame):
            logger.debug("Silence found")
            lpc_coefficients = np.array([1] * 10)
            gain = 0
            pitch = 0
        else:
            pitch = pitch_estimator(frame, self.sample_rate)
            # lpc_coefficients = librosa.lpc(frame, order=self.order)
            lpc_coefficients = self._calculate_lpc(frame)
            gain = get_frame_gain(frame, lpc_coefficients)
        logger.debug(f"Pitch: {pitch}")
        logger.debug(f"Gain {gain}")
        return pitch, gain, lpc_coefficients


    def _calculate_lpc(self, data):
        rxx = librosa.autocorrelate(data, max_size=self.order + 1)
        coeffs = scipy.linalg.solve_toeplitz((rxx[:-1], rxx[:-1]), rxx[1:])
        return np.concatenate(([1], -coeffs))
