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
    """
    A class for encoding a speech signal using linear predictive coding (LPC).

    Notes
    -----
    This is a companion of the class LpcDecoder from module
    lpc_vocoder.decoder.lpc_decoder

    Parameters
    ----------
    order : int, optional
        The order of the LPC filter (default is 10).

    Attributes
    ----------
    _frames : npt.NDArray or Generator
        An array or generator of audio frames.
    sample_rate : int
        The sample rate of the audio signal.
    order : int
        The order of the LPC filter.
    frame_data : list of EncodedFrame
        A list of encoded frames with gain, pitch, and LPC coefficients.
    window_size : int
        The size of each analysis window in samples.
    overlap : int
        The percentage overlap between adjacent windows.
    _hop_size : int
        The number of samples between adjacent windows.
    """

    def __init__(self, order: int = 10):
        """
        Create an Encoder instance and initialize the prediction order

        Parameters
        ----------
        order: int
            LPC predictor order, default=10
        """
        logger.debug(f"Encoding order: {order}")
        self._frames: npt.NDArray | Generator[np.ndarray, None, None] = np.array([0])
        self.sample_rate = 0
        self.order = order
        self.frame_data: list[EncodedFrame] = []
        self.window_size = 0
        self.overlap = 0

    def to_dict(self):
        """
        Convert the encoder information and frames to a dictionary.

        Returns
        -------
        dict
            A dictionary containing encoder metadata and encoded frames.
        """
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
        """
        Load raw audio data into the encoder.

        Parameters
        ----------
        data : npt.NDArray
            The audio signal to encode.
        sample_rate : int
            The sample rate of the audio.
        window_size : int
            The analysis window size in samples.
        overlap : int, optional
            The percentage overlap between adjacent frames (default is 50).
        """
        self._get_window_data(window_size, overlap)
        self._frames = librosa.util.frame(data.astype(np.float64), frame_length=self.window_size,
                                          hop_length=self._hop_size, axis=0)
        self.sample_rate = sample_rate
        self.frame_data = []

    def load_file(self, filename: Path, window_size: int = 0, overlap: int = 50):
        """
        Load audio data from a file.

        Parameters
        ----------
        filename : Path
            Path to the audio file to load.
        window_size : int, optional
            The analysis window size in samples (default is 0, which uses a 30 ms window).
        overlap : int, optional
            The percentage overlap between adjacent frames (default is 50).
        """
        self.sample_rate = int(librosa.get_samplerate(str(filename)))
        logger.debug(f"Sample rate {self.sample_rate}")
        self._get_window_data(window_size, overlap)
        self._frames = librosa.stream(str(filename), block_length=1, frame_length=self.window_size,
                                      hop_length=self._hop_size, mono=False, dtype=np.float64)
        self.frame_data = []

    def _get_window_data(self, window_size, overlap):
        """
        Calculate and set window size and overlap.

        Parameters
        ----------
        window_size : int
            The analysis window size in samples.
        overlap : int
            The percentage overlap between adjacent frames.
        """
        self.overlap = overlap
        self.window_size = window_size if window_size else int(self.sample_rate * 0.03)  # use a 30ms window
        logger.debug(f"Using window size: {self.window_size}")
        self._hop_size = self.window_size - int(overlap / 100 * self.window_size)  # calculate overlap in samples
        logger.debug(f"Using Overlap: {self.overlap}% ({self._hop_size} samples)")

    def encode_signal(self) -> None:
        """
        Encode the loaded audio signal into LPC frames.
        """
        logger.debug("Encoding Signal")
        for frame in self._frames:
            frame_data = self._process_frame(frame)
            self.frame_data.append(frame_data)

    def save_data(self, filename: Path) -> None:
        """
        Save encoded frames to a binary file.

        The binary file consist of a header with information of the signal and
        a list of frames, each of them will have the coefficients and any
        particular information of the frame:
        bin_data = header + [frame1, frame2, frame3, ...]

        The header of the binary file has the following format:
        header = frame size (int), sample rate (int), overlap (int), order (int)

        Each frame contains the following information:
        frame = pitch (float), gain (float), coefficients

        The coefficients are a np.array of order+1 elements (floats)

        Parameters
        ----------
        filename : Path
            Path to the file where data should be saved.
        """
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
        """
        Process a single frame to calculate LPC coefficients, gain, and pitch.

        Parameters
        ----------
        frame : npt.NDArray
            The audio frame to process.

        Returns
        -------
        EncodedFrame
            An EncodedFrame object with the processed data.
        """
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

    def _calculate_lpc(self, data: npt.NDArray) -> npt.NDArray:
        """
        Process a signal and calculate it's LPC coefficients

        Parameters
        ----------
        data : npt.NDArray
            The audio frame to process.

        Returns
        -------
        npt.NDArray
            LPC coefficients of order+1 (first element is always '1')
        """
        rxx = librosa.autocorrelate(data, max_size=self.order + 1)
        coefficients = scipy.linalg.solve_toeplitz((rxx[:-1], rxx[:-1]), rxx[1:])
        return np.concatenate(([1], -coefficients))
