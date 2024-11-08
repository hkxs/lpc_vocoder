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
from pathlib import Path

from lpc_vocoder.decode.lpc_decoder import LpcDecoder

logger = logging.getLogger("lpc_vocoder")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Encode a .wav signal using LPC"
    parser.add_argument("encoded_file", type=Path, help="Name of the input file")
    parser.add_argument("audio_file", type=Path, help="Name of the output file")
    parser.add_argument("--play", action="store_true", help="Play the decoded signal")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug messages")
    args = parser.parse_args()

    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(asctime)s [%(levelname)s] %(message)s")

    return args


def main():
    args = parse_args()
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    logger.info(f"Decoding file '{args.audio_file.resolve()}'")

    decoder = LpcDecoder()
    decoder.load_data_file(args.encoded_file.resolve())
    decoder.decode_signal()
    decoder.save_audio(args.audio_file.resolve())
    if args.play:
        decoder.play_signal()


if __name__ == '__main__':
    main()
