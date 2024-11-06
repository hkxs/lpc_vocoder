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

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _period_estimator(rxx: npt.NDArray, sample_rate: float, bandwidth: tuple[float, float], umbral: float) -> float:
    min_sample = int(sample_rate // bandwidth[1])
    max_sample = int(sample_rate // bandwidth[0])

    logger.debug(f"{min_sample=}, {max_sample=}")
    amplitude = rxx[min_sample:max_sample].max()
    logger.debug(f"Max Energy: {float(rxx[0])}, Harmonic Energy: {float(amplitude)}, Ratio: {float(amplitude/rxx[0])}")

    index = rxx[min_sample:max_sample].argmax(axis=0)
    period = (index + min_sample) if amplitude >= (umbral * rxx[0]) else 0
    logger.debug(f"Harmonic Index: {int(period)}")
    return period


def pitch_estimator(signal: npt.NDArray, sample_rate: float) -> float:
    """
    Estimate the pitch of a signal

    Use autocorrelation to identify if a signal is periodic, then use the
    information of the max value within a predefined frequency range to estimate
     the pitch.

    This method assumes that the speach signal is within the following frequency
    range: 40-3500 Hz, but it's fundamental frequency (or pitch) should be
    between 40-600 Hz

    For non-periodic signals the output will be '-1', otherwise it will return
    the pitch in Hz

    TODO: look for a better algorithm

    Parameters
    ----------
    signal : npt.NDArray
        Signal to be analyzed
    sample_rate : int
        Sample rate of the signal

    Returns
    -------
    pitch : float
    """
    umbral = 0.30
    speech_bandwidth = (100, 600)
    max_period = (sample_rate // speech_bandwidth[1])

    rxx = np.correlate(signal, signal, mode='full')[len(signal) - 1:]

    period = _period_estimator(rxx, sample_rate, speech_bandwidth, umbral)
    if period <= max_period:
        # Our signal is so low on frequency that we need to update our interval,
        # we'll adjust the bandwidth and the umbral
        umbral = 0.1
        max_period = (sample_rate // speech_bandwidth[1])
        speech_bandwidth = (40, 100)
        period = _period_estimator(rxx, sample_rate, speech_bandwidth, umbral)
        if period < max_period:
            # if after the bandwidth adjustment we're still over the limit,
            # mark it as non-periodic
            period = 0
            logger.debug("Non-periodic signal")

    freq = sample_rate / period if period else -1
    logger.debug(f"Pitch: {freq}")
    return freq
