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
from lpc_vocoder.utils.utils import get_frame_gain, is_silence, pre_emphasis
from lpc_vocoder.utils.dataclasses import EncodedFrame

logger = logging.getLogger(__name__)


class LpcEncoder:
    def __init__(self, order: int = 10):
        logger.debug(f"Encoding order: {order}")
        self._frames = None
        self.sample_rate = None
        self.order = order
        self.frame_data = []
        self.window_size = None
        self.overlap = None

    def load_data(self, data: np.array, sample_rate: int, window_size: int, overlap: int = 50) -> None:
        self._get_window_data(window_size, overlap)
        self._frames = librosa.util.frame(data.astype(np.float64), frame_length=self.window_size, hop_length=self._hop_size, axis=0)
        self.sample_rate = sample_rate
        self.frame_data = []

    def load_file(self, filename: Path, window_size: int = None, overlap: int = 50):
        self.sample_rate = librosa.get_samplerate(str(filename))
        logger.debug(f"Sample rate {self.sample_rate}")
        self._get_window_data(window_size, overlap)
        self._frames = librosa.stream(str(filename), block_length=1, frame_length=self.window_size,
                                hop_length=self._hop_size, mono=False, dtype=np.float64)
        self.frame_data = []

    def _get_window_data(self, window_size, overlap):
        self.overlap = overlap
        self.window_size = window_size if window_size else int(self.sample_rate * 0.03)  # use a 30ms window
        logger.debug(f"Using window size: {self.window_size}")
        self._hop_size = self.window_size - int(overlap / 100 * self.window_size)  # calculate overlap in samples
        logger.debug(f"Using Overlap: {self.overlap}% ({self._hop_size} samples)")

    def encode_signal(self) -> None:
        logger.debug("Encoding Signal")
        for frame in self._frames:
            frame_data= self._process_frame(frame)
            self.frame_data.append(frame_data)

    def save_data(self, filename: Path) -> None:
        # format
        # frame size, sample rate, overlap, order, pitch, gain, data, pitch, gain, data
        logger.debug(f"Saving data to '{filename}'")
        with open(filename, "w") as f:
            f.write(f"{self.window_size}, {self.sample_rate}, {self.overlap}, {self.order}\n")
            for frame in self.frame_data:
                f.write(str(frame))

    def _process_frame(self, frame: np.array) -> EncodedFrame:
        if is_silence(frame) or len(frame) < self.window_size:
            logger.debug("Silence found")
            lpc_coefficients = np.ones(self.order + 1)
            gain = 0
            pitch = 0
        else:
            window = librosa.filters.get_window('hamming', self.window_size)
            pitch = pitch_estimator(frame, self.sample_rate)
            logger.debug(f"Pitch: {pitch}")
            frame = pre_emphasis(window * frame)
            lpc_coefficients = self._calculate_lpc(frame)
            gain = get_frame_gain(frame, lpc_coefficients)
            logger.debug(f"Gain {gain}")

        return EncodedFrame(gain, pitch, lpc_coefficients)

    def _calculate_lpc(self, data):
        rxx = librosa.autocorrelate(data, max_size=self.order + 1)
        coefficients = scipy.linalg.solve_toeplitz((rxx[:-1], rxx[:-1]), rxx[1:])
        return np.concatenate(([1], -coefficients))
