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

import matplotlib.pyplot as plt

from lpc_vocoder.decode.lpc_decoder import LpcDecoder
from lpc_vocoder.encode.lpc_encoder import LpcEncoder
from lpc_vocoder.utils.utils import play_signal

def test_vocoder():
    sound_file = Path(__file__).parent / "audios" / "sine_24hz.wav"
    encoder = LpcEncoder(sound_file, 256, 0, 10)
    encoder.encode_signal()

    decoder = LpcDecoder()
    decoder.load_data(encoder.frame_data, encoder.window_size, encoder.sample_rate, encoder.overlap, encoder.order)
    decoder.decode_signal()
    plt.plot(decoder.signal[:512])
    plt.show()
    play_signal(decoder.signal, decoder.sample_rate)
