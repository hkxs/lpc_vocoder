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
import struct
from pathlib import Path
from typing import Generator

import librosa
import librosa.feature
import numpy as np
import numpy.typing as npt
import scipy
from lpc_vocoder.utils.dataclasses import EncodedFrame
from lpc_vocoder.utils.pitch_estimation import pitch_estimator
from lpc_vocoder.utils.utils import get_frame_gain
from lpc_vocoder.utils.utils import is_silence
from lpc_vocoder.utils.utils import pre_emphasis

logger = logging.getLogger(__name__)


class LpcEncoder:
    def __init__(self, order: int = 10):
        logger.debug(f"Encoding order: {order}")
        self._frames : npt.NDArray | Generator[np.ndarray, None, None] = np.array([0])
        self.sample_rate = 0
        self.order = order
        self.frame_data : list[EncodedFrame] = []
        self.window_size = 0
        self.overlap = 0

    def to_dict(self):
        signal_data = {
            "encoder_info": {
                "order": self.order,
                "window_size": self.window_size,
                "overlap": self.overlap,
                "sample_rate": self.sample_rate,
            },
            "frames": [frame.__dict__ for frame in self.frame_data],
        }
        return signal_data

    def load_data(self, data: npt.NDArray, sample_rate: int, window_size: int, overlap: int = 50) -> None:
        self._get_window_data(window_size, overlap)
        self._frames = librosa.util.frame(data.astype(np.float64), frame_length=self.window_size,
                                          hop_length=self._hop_size, axis=0)
        self.sample_rate = sample_rate
        self.frame_data = []

    def load_file(self, filename: Path, window_size: int = 0, overlap: int = 50):
        self.sample_rate = int(librosa.get_samplerate(str(filename)))
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
            frame_data = self._process_frame(frame)
            self.frame_data.append(frame_data)

    def save_data(self, filename: Path) -> None:
        # format
        # frame size, sample rate, overlap, order, pitch, gain, data, pitch, gain, data
        if not filename.suffix:
            filename = filename.with_suffix(".bin")
        logger.debug(f"Saving data to '{filename}'")
        data = bytearray()
        data.extend(struct.pack('i', self.window_size))
        data.extend(struct.pack('i', self.sample_rate))
        data.extend(struct.pack('i', self.overlap))
        data.extend(struct.pack('i', self.order))

        for frame in self.frame_data:
            data.extend(struct.pack('d', frame.gain))
            data.extend(struct.pack('d', frame.pitch))
            data.extend(frame.coefficients.tobytes())

        with open(filename, "wb") as f:
            f.write(data)

    def _process_frame(self, frame: npt.NDArray) -> EncodedFrame:
        if is_silence(frame) or len(frame) < self.window_size:
            logger.debug("Silence found")
            lpc_coefficients = np.ones(self.order + 1)
            gain = 0.0
            pitch = 0.0
        else:
            window = librosa.filters.get_window('hamming', self.window_size)
            pitch = pitch_estimator(frame, self.sample_rate)
            frame = pre_emphasis(window * frame)
            lpc_coefficients = self._calculate_lpc(frame)
            gain = get_frame_gain(frame, lpc_coefficients)

        return EncodedFrame(gain, pitch, lpc_coefficients)

    def _calculate_lpc(self, data):
        rxx = librosa.autocorrelate(data, max_size=self.order + 1)
        coefficients = scipy.linalg.solve_toeplitz((rxx[:-1], rxx[:-1]), rxx[1:])
        return np.concatenate(([1], -coefficients))
