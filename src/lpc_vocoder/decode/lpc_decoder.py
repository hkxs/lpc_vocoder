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

import logging
import struct
from pathlib import Path

import numpy as np
import scipy
from lpc_vocoder.utils.dataclasses import EncodedFrame
from lpc_vocoder.utils.utils import de_emphasis
from lpc_vocoder.utils.utils import gen_excitation
from lpc_vocoder.utils.utils import play_signal


logger = logging.getLogger(__name__)


class LpcDecoder:
    """
    A class for decoding LPC-encoded audio back to a waveform.

    Notes
    -----
    This is a companion of the class LpcEncoder from module
    lpc_vocoder.encoder.lpc_encoder

    Attributes
    ----------
    frame_data : list of EncodedFrame
        A list of frames to decode.
    sample_rate : int
        The sample rate of the audio.
    window_size : int
        The analysis window size in samples.
    overlap : int
        The percentage overlap between adjacent frames.
    order : int
        The order of the LPC filter.
    signal : npt.NDArray
        The reconstructed audio signal.
    """

    def __init__(self):
        """
        Creates a Decoder instance that ca be used to reconstruct the signal
        encoded by LpcEncoder.
        """
        self.data = []
        self.sample_rate = None
        self.window_size = None
        self.overlap = None
        self.order = None
        self.frame_data = []
        self.signal = None

    def load_data(self, data: dict) -> None:
        """
        Load LPC data from a dictionary containing encoded frames and metadata.

        Notes
        -----
        This data should come from LpcEncoder().to_dict() method

        Parameters
        ----------
        data : dict
            Dictionary containing encoded frames and metadata.
        """
        self.frame_data = [EncodedFrame(**frame) for frame in data["frames"]]
        self.window_size = data["encoder_info"]["window_size"]
        self.sample_rate = data["encoder_info"]["sample_rate"]
        self.overlap = data["encoder_info"]["overlap"]
        self.order = data["encoder_info"]["order"]

    def load_data_file(self, filename: Path) -> None:
        """
        Load LPC encoded data from a binary file.

        The binary file contains frame encoding parameters and metadata,
        including window size, sample rate, overlap, order, and frame data
        (gain, pitch, and coefficients).

        Notes
        -----
        This binary file should have been generated using
        LpcEncoder().save_data() method

        Parameters
        ----------
        filename : Path
            The path to the binary file to load.
        """
        with open(filename, "rb") as f:
            encoded_data = f.read()

        offset = 0
        self.window_size = struct.unpack_from("i", encoded_data, offset)[0]
        offset += struct.calcsize("i")
        self.sample_rate = struct.unpack_from("i", encoded_data, offset)[0]
        offset += struct.calcsize("i")
        self.overlap = struct.unpack_from("i", encoded_data, offset)[0]
        offset += struct.calcsize("i")
        self.order = struct.unpack_from("i", encoded_data, offset)[0]
        offset += struct.calcsize("i")

        logger.debug(f"Encoding order: {self.order}")
        logger.debug(f"Sample rate: {self.sample_rate}")
        logger.debug(f"Using window size: {self.window_size}")

        while offset < len(encoded_data):
            gain = struct.unpack_from('d', encoded_data, offset)[0]
            offset += struct.calcsize('d')

            pitch = struct.unpack_from('d', encoded_data, offset)[0]
            offset += struct.calcsize('d')

            coefficients = np.frombuffer(encoded_data, dtype=np.float64, count=self.order + 1, offset=offset)
            offset += (self.order + 1) * coefficients.itemsize

            frame_data = EncodedFrame(float(gain), float(pitch), coefficients)
            self.frame_data.append(frame_data)

    def decode_signal(self) -> None:
        """
        Decode the LPC-encoded signal using the loaded frame data and parameters.

        This method applies filtering and overlap-add synthesis to reconstruct the original signal
        from encoded frame data, gain, pitch, and coefficients.
        """
        initial_conditions = np.zeros(self.order)
        hop_size = int(self.window_size * (1 - self.overlap/100))  # calculate overlap in samples
        total_length = len(self.frame_data) * hop_size + self.window_size
        output_signal = np.zeros(total_length)

        logger.debug(f"Using Overlap: {self.overlap}% ({hop_size} samples)")
        for index, frame in enumerate(self.frame_data):
            if not frame.gain:
                logger.debug("Adding Silence")
                reconstructed = np.zeros(self.window_size)  # just add silence
            else:
                excitation = gen_excitation(frame.pitch, self.window_size, self.sample_rate)
                reconstructed, initial_conditions = scipy.signal.lfilter([1.], frame.coefficients,
                                                                         frame.gain * excitation, zi=initial_conditions)
                reconstructed = de_emphasis(reconstructed)
            start_idx = index * hop_size
            end_idx = start_idx + self.window_size
            output_signal[start_idx:end_idx] += reconstructed
        self.signal = output_signal

    def save_audio(self, filename: Path) -> None:
        """
        Save the decoded signal as an audio file in .wav format.

        Parameters
        ----------
        filename : Path
            The path to save the audio file.
        """
        import soundfile as sf  # lazy loader, we only load it if we need it
        logger.debug(f"Creating file '{filename}'")
        sf.write(filename, self.signal, samplerate=self.sample_rate)

    def play_signal(self) -> None:
        """
        Play the decoded audio signal through the default audio output.
        """
        logger.debug("Playing signal")
        play_signal(self.signal, sample_rate=self.sample_rate)
