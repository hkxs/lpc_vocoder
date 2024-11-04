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

import numpy as np
import matplotlib.pyplot as plt

from lpc_vocoder.utils.pitch_estimation import pitch_estimator

class TestPitchDetector:

    sample_rate = 8000
    signal_size = 256

    def gen_sin_wave(self, frequency):
        samples = np.arange(self.signal_size) / self.sample_rate
        return np.sin(2 * np.pi * frequency * samples)

    def noise(self):
        return np.random.uniform(0, 1, self.signal_size)

    def test_sin_wave(self):
        for frequency in range(50, 600, 10):
            signal = self.gen_sin_wave(frequency)
            est_freq = pitch_estimator(signal, self.sample_rate)

            error_rate = abs((frequency - est_freq) / frequency)
            error_rate = float(100 * error_rate)
            assert error_rate <= 10  # check if we have less than 10% error

    def test_sin_wave_and_noise(self):
        for frequency in range(50, 600, 10):
            signal = self.gen_sin_wave(frequency) + self.noise()
            est_freq = pitch_estimator(signal, self.sample_rate)

            plt.plot(signal)
            plt.show()
            error_rate = abs((frequency - est_freq) / frequency)
            error_rate = float(100 * error_rate)
            assert error_rate <= 10  # check if we have less than 10% error
