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

from lpc_vocoder.encode.lpc_encoder import encode_signal

logger = logging.getLogger(__name__)


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
    logger.info(f"Encoding file '{args.filename}'")
    encode_signal(args.filename, window_size=256, overlap=0)


if __name__ == '__main__':
    main()
