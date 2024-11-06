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

from lpc_vocoder.encode.lpc_encoder import LpcEncoder

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Encode a .wav signal using LPC"
    parser.add_argument("audio_file", type=Path, help="Name of the input file")
    parser.add_argument("filename", type=Path, help="Name of the output file")
    parser.add_argument("--order",  default=10, action="store_true", help=f"Order of the LPC filter, default '{10}'")
    parser.add_argument("--frame_size", action="store_true", help="Frame Size, if not provided it will use a 30ms window based on the sample rate")
    parser.add_argument("--overlap", default=50, action="store_true", help=f"Overlap as percentage (0-100), default '{50}'")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug messages")
    args = parser.parse_args()

    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO)

    return args


def main():
    args = parse_args()
    logger.info(f"Encoding file '{args.audio_file}'")
    encoder = LpcEncoder(order=args.order)
    encoder.load_file(args.audio_file, window_size=args.frame_size, overlap=args.overlap)
    encoder.encode_signal()
    encoder.save_data(args.filename)

if __name__ == '__main__':
    main()
