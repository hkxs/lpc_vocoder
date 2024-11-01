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

import numpy as np
import scipy

from lpc_vocoder.utils.utils import gen_excitation


class LpcDecoder:
    def __init__(self, filename: Path):
        self.filename = filename
        self.data = []
        self.sample_rate = None
        self.window_size = None
        self.overlap = None
        self.order = None
        self.frame_data = []
        self._audio_frames = []
        self.signal = None

    def _get_signal_data(self):
        with open(self.filename) as f:
            audio_data = f.readlines()
        self.window_size, self.sample_rate, self.overlap, self.order = map(int, audio_data[0].split(","))

        for frame in audio_data[1:]:
            pitch, gain, lpc_coefficients = frame.split(",")
            frame_data = {"pictch": float(pitch), "gain": float(gain),
                          "coefficients": np.frombuffer(bytes.fromhex(lpc_coefficients), dtype=np.float32)}
            self.frame_data.append(frame_data)

    def decode_signal(self):
        self._audio_frames = []
        for frame in self.frame_data:
            if not frame["gain"]:
                reconstructed = np.array([0] * self.window_size)  # just add silence
            else:
                excitation = gen_excitation(frame["pitch"], self.window_size, self.sample_rate)
                reconstructed = scipy.signal.lfilter([frame["gain"]], frame["coefficients"], excitation)
        self._audio_frames.append(reconstructed)
        self.signal = np.concatenate(self._audio_frames)
