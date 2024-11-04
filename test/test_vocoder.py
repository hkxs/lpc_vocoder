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
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from lpc_vocoder.decode.lpc_decoder import LpcDecoder
from lpc_vocoder.encode.lpc_encoder import LpcEncoder
from lpc_vocoder.utils.utils import play_signal


def gen_sine_wave(frequency, sample_rate, length):
    samples = np.arange(length) / sample_rate
    return np.sin(2 * np.pi * frequency * samples)


class TestEncoder:

    sample_rate = 8000

    @pytest.fixture(scope="class")
    def encoder(self):
        return LpcEncoder()

    @pytest.fixture(scope="class")
    def sine_wave(self):
        return gen_sine_wave(440, self.sample_rate, 16000)

    @pytest.fixture(scope="class")
    def wav_file(self, sine_wave):
        signal = sine_wave
        wav_file = Path("test.wav")
        sf.write(wav_file, signal, samplerate=self.sample_rate)
        yield wav_file
        os.remove(wav_file)

    def test_load_data_from_file(self, encoder, wav_file):
        encoder.load_file(wav_file)
        assert encoder.order == 10
        assert encoder.sample_rate == self.sample_rate
        assert encoder.window_size == 240
        assert encoder.overlap == 50

        encoder.load_file(wav_file, window_size=256, overlap=70)
        assert encoder.order == 10
        assert encoder.sample_rate == self.sample_rate
        assert encoder.window_size == 256
        assert encoder.overlap == 70

    def test_load_data(self, encoder, sine_wave):
        encoder.load_data(sine_wave, self.sample_rate, 256)
        assert encoder.order == 10
        assert encoder.sample_rate == self.sample_rate
        assert encoder.window_size == 256
        assert encoder.overlap == 50

        encoder.load_data(sine_wave, self.sample_rate, 256, 70)
        assert encoder.order == 10
        assert encoder.sample_rate == self.sample_rate
        assert encoder.window_size == 256
        assert encoder.overlap == 70

    def test_encoding(self, encoder, wav_file):
        encoder.load_file(wav_file, window_size=256)
        assert not encoder.frame_data
        encoder.encode_signal()
        assert encoder.frame_data

        pitch = int(encoder.frame_data[0]["pitch"])
        for frame in encoder.frame_data:
            assert int(frame["pitch"]) == pitch
        assert pitch == 444

class TestDecoder:
    sample_rate = 8000

    @pytest.fixture(scope="class")
    def encoder(self):
        return LpcDecoder()
