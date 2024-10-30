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

import scipy
import pyaudio

import numpy as np

decoded = np.array([])
with open("../encode/file.txt") as f:
    audio_data = f.readlines()

audio_info = audio_data[0]
frame_size, sr, overlap, order = map(int, audio_info.split(","))
noise =  np.random.rand(frame_size)

for audio in audio_data[1:]:
    pitch, gain, a = audio.split(",")
    pitch = float(pitch)
    gain = float(gain)
    a = np.frombuffer(bytes.fromhex(a), dtype=np.float32)
    if pitch == -1:
        excitation = noise
    else:
        period = int(sr // int(pitch))
        excitation = scipy.signal.unit_impulse(frame_size, range(0, frame_size, period))
    reconstructed = scipy.signal.lfilter([gain], a, excitation)
    decoded = np.concatenate((decoded, reconstructed))

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(sr), output=True)
stream.write(decoded.astype('float32').tobytes())
stream.close()
p.terminate()
