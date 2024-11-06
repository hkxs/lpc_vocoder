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

from pathlib import Path
import logging

import numpy as np
import scipy

from lpc_vocoder.utils.utils import gen_excitation, de_emphasis, play_signal
from lpc_vocoder.utils.dataclasses import EncodedFrame


logger = logging.getLogger(__name__)


class LpcDecoder:
    def __init__(self):
        self.data = []
        self.sample_rate = None
        self.window_size = None
        self.overlap = None
        self.order = None
        self.frame_data = []
        self.signal = None

    def load_data(self, data: list[EncodedFrame], window_size: int, sample_rate: int,  overlap: int, order: int) -> None:
        """ Load data directly from the encoder """
        self.frame_data = data
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.order = order

    def load_data_file(self, filename: Path) -> None:
        with open(filename) as f:
            audio_data = f.readlines()
        self.window_size, self.sample_rate, self.overlap, self.order = map(int, audio_data[0].split(","))
        logger.debug(f"Encoding order: {self.order}")
        logger.debug(f"Sample rate: {self.sample_rate}")
        logger.debug(f"Using window size: {self.window_size}")

        for frame in audio_data[1:]:
            pitch, gain, lpc_coefficients = frame.split(",")
            logger.debug(f"Pitch: {pitch}")
            logger.debug(f"Gain {gain}")
            coefficients = np.frombuffer(bytes.fromhex(lpc_coefficients), dtype=np.float32)
            frame_data = EncodedFrame(float(gain), float(pitch), coefficients)
            self.frame_data.append(frame_data)

    def decode_signal(self) -> None:
        initial_conditions = np.zeros(self.order)
        hop_size = int(self.window_size * (1 - self.overlap/100))  # calculate overlap in samples
        total_length = len(self.frame_data) * hop_size + self.window_size
        output_signal = np.zeros(total_length)

        logger.debug(f"Using Overlap: {self.overlap}% ({hop_size} samples)")
        for index, frame in enumerate(self.frame_data):
            if not frame.gain:
                logger.debug("Adding Silence")
                reconstructed = np.zeros(self.window_size)  # just add silence
            else:
                excitation = gen_excitation(frame.pitch, self.window_size, self.sample_rate)
                reconstructed, initial_conditions = scipy.signal.lfilter([1.], frame.coefficients, frame.gain * excitation, zi=initial_conditions)
                reconstructed = de_emphasis(reconstructed)
            start_idx = index * hop_size
            end_idx = start_idx + self.window_size
            output_signal[start_idx:end_idx] += reconstructed
        self.signal = output_signal

    def save_audio(self, filename: Path) -> None:
        import soundfile as sf  # lazy loader, we only load it if we need it
        logger.debug(f"Creating file '{filename}'")
        sf.write(filename, self.signal, samplerate=self.sample_rate)

    def play_signal(self):
        logger.debug("Playing signal")
        play_signal(self.signal, sample_rate=self.sample_rate)
