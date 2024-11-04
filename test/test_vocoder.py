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
        wav_file = Path("test.wav")
        sf.write(wav_file, sine_wave, samplerate=self.sample_rate)
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
    order = 10
    overlap = 50
    window_size = 256

    @pytest.fixture(scope="class")
    def frame_data(self):
        coeffs = np.concatenate(([1], np.zeros(self.order)))
        return {"pitch": -1.0, "gain": 0.5, "coefficients": np.array(coeffs)}

    @pytest.fixture(scope="class")
    def encoded_file(self, frame_data):
        data_file = Path("test.txt")
        with open(data_file, "w") as f:
            f.write(f"{self.window_size}, {self.sample_rate}, {self.overlap}, {self.order}\n")
            f.write(f"{frame_data['pitch']}, {frame_data['gain']}, {frame_data['coefficients'].tobytes().hex()}\n")
            f.write(f"{frame_data['pitch']}, {frame_data['gain']}, {frame_data['coefficients'].tobytes().hex()}\n")
        yield data_file
        os.remove(data_file)

    @pytest.fixture(scope="class")
    def decoder(self):
        return LpcDecoder()

    def test_load_data(self, decoder, frame_data):
        decoder.load_data([frame_data], self.window_size, self.sample_rate, self.overlap, self.order)
        assert decoder.order == self.order
        assert decoder.window_size == self.window_size
        assert decoder.sample_rate == self.sample_rate
        assert decoder.overlap == self.overlap
        assert decoder.overlap == self.overlap

    def test_load_data_file(self, decoder, encoded_file):
        decoder.load_data_file(encoded_file)
        assert decoder.order == self.order
        assert decoder.window_size == self.window_size
        assert decoder.sample_rate == self.sample_rate
        assert decoder.overlap == self.overlap
        assert decoder.overlap == self.overlap

    def test_decoding(self, decoder, frame_data):
        decoder.load_data([frame_data], self.window_size, self.sample_rate, self.overlap, self.order)
        assert not decoder.signal
        decoder.decode_signal()
        assert decoder.signal.any()


class TestVocoder:
    sample_rate = 8000

    audio_path = Path() / "audios"

    def _process_audio(self, file, frame_size):
        encoder = LpcEncoder(order=40)
        decoder = LpcDecoder()
        encoder.load_file(file, frame_size)
        encoder.encode_signal()
        decoder.load_data(encoder.frame_data, encoder.window_size, encoder.sample_rate, encoder.overlap, encoder.order)
        decoder.decode_signal()
        play_signal(decoder.signal, decoder.sample_rate)

    def test_vocoder_1(self):
        self._process_audio(self.audio_path / "once_there_was.flac", 480)

    def test_vocoder_2(self):
        self._process_audio(self.audio_path / "the_boys.flac", 512)

    def test_vocoder_3(self):
        self._process_audio(self.audio_path / "what_do_you_mea_sir.flac", 512)

    def test_vocoder_4(self):
        self._process_audio(self.audio_path / "then_darkness.flac", 512)

    def test_vocoder_5(self):
        self._process_audio(self.audio_path / "sine_240hz.wav", 512)
