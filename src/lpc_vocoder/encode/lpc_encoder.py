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

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path


import librosa
import numpy as np

from lpc_vocoder.encode.pitch_estimation import pitch_estimator

logger = logging.getLogger(__name__)

# format
# frame size, sample rate, overlap, order, pitch, gain, data, pitch, gain, data
# if pitch == 0, then use noise as input

def get_gain(frame: np.array, coefficients: np.array) -> float:
    rxx = librosa.autocorrelate(frame)
    G = np.sqrt(rxx[0] - np.dot(coefficients[1:], rxx[1:len(coefficients)]))
    return G


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Encode a .wav signal using LPC"
    parser.add_argument("filename", type=Path, help="Name of the output file")
    # TODO add the following arguments, order, frame size, overlap percentage
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug messages")
    args = parser.parse_args()

    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO)

    return args

def main():
    args = parse_args()
    random.seed(datetime.now().timestamp())
    sr = librosa.get_samplerate(args.filename)
    logger.debug(f"Sampling rate: {sr}")

    frames = librosa.stream(args.filename,
                            block_length=1,
                            frame_length=256,
                            hop_length=256,
                            mono=False
                            )
    with open("file.txt", "w") as f:
        f.write(f"{256}, {sr}, {0}, {10}\n")

        for frame in frames:
            pitch = pitch_estimator(frame, sr)

            logger.debug(f"Pitch: {pitch}")
            a = librosa.lpc(frame, order=10)
            gain = get_gain(frame, a)
            logger.debug(f"Gain {gain}")
            # f.write(f"{pitch}, {gain}, {str(a).replace('\n', '')}\n")
            f.write(f"{pitch}, {gain}, {a.tobytes().hex()}\n")

if __name__ == '__main__':
    main()
