from scipy.signal import butter, filtfilt
import numpy as np

from python_propagate.states import OrbitalElements
from typing import Iterable


# Create a low-pass filter
def butter_lowpass_filter(data, cutoff_freq, fs, order=1):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="lowpass", analog=False)
    return filtfilt(b, a, data)  # Apply filter using zero-phase filtering


def fft_lowpass_filter(data, cutoff_freq, fs):
    freqs = np.fft.fftfreq(len(data), d=1 / fs)
    fft_vals = np.fft.fft(data)

    # Zero out frequencies beyond the cutoff
    fft_vals[np.abs(freqs) > cutoff_freq] = 0

    # Inverse FFT to get the filtered signal
    return np.real(np.fft.ifft(fft_vals))


def split_list(lst, n):
    return np.array_split(lst, n)


class ProperElements:

    def __init__(
        self, orbital_element_data: Iterable[OrbitalElements], duration, dt=30
    ):

        sma_data = np.array([oe.sma for oe in orbital_element_data])
        ecc_data = np.array([oe.ecc for oe in orbital_element_data])
        inc_data = np.array([oe.inc for oe in orbital_element_data])
        arg_data = np.array([oe.arg for oe in orbital_element_data])
        raan_data = np.array([oe.raan for oe in orbital_element_data])
        nu_data = np.array([oe.nu for oe in orbital_element_data])

        orbital_period = (
            2 * np.pi * np.sqrt(sma_data[0] ** 3 / orbital_element_data[0].mu)
        )
        # orbital_period = dt.total_seconds() * len(orbital_element_data)
        oe_data = (sma_data, ecc_data, inc_data, arg_data, raan_data, nu_data)
        fs = 1 / dt.total_seconds()
        segment = np.int16(duration.total_seconds() / orbital_period)
        cutoff_freq = 1 / orbital_period
        # fs = 1/(dt.total_seconds())

        self.mean_orbital_elements_data = [
            butter_lowpass_filter(data=oe, cutoff_freq=1 / orbital_period, fs=fs)
            for oe in oe_data
        ]
        self.mean_element_segments = [
            split_list(oe, segment) for oe in self.mean_orbital_elements_data
        ]

        oe_data = []
        for oe in self.mean_element_segments:
            segment_data = []
            for seg in oe:
                segment_data.extend(
                    fft_lowpass_filter(
                        data=seg, cutoff_freq=cutoff_freq / segment, fs=fs
                    )
                )
            oe_data.append(np.array(segment_data))

        self.proper_orbital_elements_data = oe_data

        pass
