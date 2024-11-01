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

from lpc_vocoder.utils.pitch_estimation import pitch_estimator
from lpc_vocoder.utils.utils import get_frame_gain, is_silence

logger = logging.getLogger(__name__)


class LpcEncoder:
    def __init__(self, filename: Path, window_size: int = None, overlap: int = 50, order: int = 10):
        self.filename = filename
        self.order = order
        self.frame_data = []
        self.sample_rate = librosa.get_samplerate(str(self.filename))
        self.window_size = window_size if window_size else int(self.sample_rate * 0.03)  # use a 30ms window
        self.overlap = window_size - int(overlap / 100 * window_size)
        logger.debug(f"Sampling rate: {self.sample_rate} Hz")
        logger.debug(f"Frame Size: {self.window_size} samples")
        logger.debug(f"Overlap: {self.overlap} samples")


    def encode_signal(self) -> None:
        # window = librosa.filters.get_window('hamming', window_size)
        frames = librosa.stream(str(self.filename), block_length=1, frame_length=self.window_size,
                                hop_length=self.overlap, mono=False, dtype=np.float64)

        for frame in frames:
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
            lpc_coefficients = librosa.lpc(frame, order=self.order)
            gain = get_frame_gain(frame, lpc_coefficients)
        logger.debug(f"Pitch: {pitch}")
        logger.debug(f"Gain {gain}")
        return pitch, gain, lpc_coefficients
