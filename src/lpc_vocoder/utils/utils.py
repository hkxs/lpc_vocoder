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

import librosa
import librosa.feature
import numpy as np
import scipy


def preephasis(signal: np.ndarray) -> np.ndarray:
    return scipy.signal.lfilter([1, 0.9375], [1], signal)


def deephasis(signal: np.ndarray) -> np.ndarray:
    return scipy.signal.lfilter([1], [1, -0.9375], signal)


def gen_excitation(pitch: float, frame_size, sample_rate: int):
    if pitch == -1:
        excitation = np.random.rand(frame_size)
    else:
        period = int(sample_rate // int(pitch))
        excitation = scipy.signal.unit_impulse(frame_size, range(0, frame_size, period))
    return excitation

def get_frame_gain(frame: np.array, coefficients: np.array) -> float:
    rxx = librosa.autocorrelate(frame, max_size=len(coefficients))
    g_squared = rxx[0] - np.dot(coefficients[1:], rxx[1:])
    return np.sqrt(g_squared)


def is_silence(signal: np.array) -> bool:
    """
    Check if the signal is silence, it uses -60dB as threshold, this is based on
    librosa.effects._signal_to_frame_nonsilent
    """
    rms = librosa.feature.rms(y=signal, frame_length=256, hop_length=256)
    power = librosa.core.amplitude_to_db(rms[..., 0, :], ref=np.max, top_db=None)
    silence = np.flatnonzero(power < -60)
    return silence.size > 0

def play_signal(signal: np.array, sample_rate: int):
    import pyaudio  # lazy loader since we don't need it all the time

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
    stream.write(signal.astype('float32').tobytes())
    stream.close()
    p.terminate()
