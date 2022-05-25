"""Radar FFTs."""
from typing import Callable, Optional

import numpy as np


def range_fft(x: np.ndarray,
              window: Optional[Callable[[int], np.ndarray]] = None,
              remove_mean: bool = True) -> np.ndarray:
    """Apply a Range FFT.

    Args:
        x: Raw radar data of shape [..., n_samples].
        window: Window function for Range FFT. If None, no window is applied. Otherwise, the given window
            function is applied. (Default: None)
        remove_mean: If True, remove mean along samples dimension. (Default: True)

    Returns:
        Transformed range data of shape [..., n_range].

    Examples:
        Dummy data: 10 frames, 3 antennas, 64 chirps, 128 samples.

        >>> raw_data = np.random.rand(10,3,64,128)
        >>> range_data = range_fft(raw_data)

        Use window function. (Available in `scipy https://docs.scipy.org/doc/scipy/reference/signal.windows.html`)

        >>> from scipy import signal
        >>> range_data = range_fft(raw_data, signal.windows.blackman)

        If window function uses additional parameters you can wrap it with lambda.

        >>> range_window_func = lambda x: signal.windows.chebwin(x, at=100)
        >>> range_data = range_fft(raw_data, range_window_func)

    See Also:
        :func:`doppler_fft`: Apply a Doppler FFT.
    """
    n_samples = x.shape[-1]

    if remove_mean:
        x = x - x.mean(axis=-1, keepdims=True)

    if window is not None:
        w_array = window(n_samples)
        w_array /= w_array.sum()  # Normalize window
        x = x * w_array[(x.ndim - 1) * (None,) + (slice(None),)]
    x_range = np.fft.fft(x, axis=-1)  # Range FFT
    x_range = x_range[..., :n_samples // 2]  # Real data is symmetric

    return x_range


def doppler_fft(x: np.ndarray,
                window: Optional[Callable[[int], np.ndarray]] = None) -> np.ndarray:
    """Apply a Doppler FFT.

    Args:
        x: Range data of shape [..., n_chirps, n_range].
        window: Window function for Doppler FFT. If None, no window is applied. Otherwise, the given window
            function is applied.

    Returns:
        Transformed range doppler data of shape [..., n_doppler, n_range].

    Examples:
        Dummy data: 10 frames, 3 antennas, 64 chirps, 128 range samples.

        >>> range_data = np.random.rand(10,3,64,128)
        >>> doppler_data = doppler_fft(range_data)

        Use window function. (Available in `scipy https://docs.scipy.org/doc/scipy/reference/signal.windows.html`)

        >>> from scipy import signal
        >>> doppler_data = doppler_fft(range_data, signal.windows.blackman)

        If window function uses additional parameters you can wrap it with lambda.

        >>> doppler_window_func = lambda x: signal.windows.chebwin(x, at=100)
        >>> doppler_data  = doppler_fft(range_data, doppler_window_func)

    See Also:
        :func:`range_fft`: Apply a Range FFT.
    """
    n_chirps = x.shape[-2]

    if window is not None:
        w_array = window(n_chirps)
        w_array /= w_array.sum()  # Normalize window
        x = x * w_array[(x.ndim - 2) * (None,) + (slice(None), None)]
    x_rdi = np.fft.fft(x, axis=-2)  # Doppler FFT
    x_rdi = np.fft.fftshift(x_rdi, axes=-2)  # Swap spectrum

    return x_rdi


def range_doppler_fft(x: np.ndarray,
                      range_window: Optional[Callable[[
                          int], np.ndarray]] = None,
                      doppler_window: Optional[Callable[[
                          int], np.ndarray]] = None,
                      remove_mean: bool = True) -> np.ndarray:
    """
    Generate a Range Doppler Response.

    Args:
        x: Raw radar data of shape [..., n_chirps, n_samples].
        range_window: Window function for Range FFT. If None, no window is applied. Otherwise, the given window
            function is applied. (Default: None)
        doppler_window: Window function for Doppler FFT. If None, no window is applied. Otherwise, the given window
            function is applied. (Default: None)
        remove_mean: If True, remove mean along samples dimension. (Default: True)

    Returns:
        Transformed range data of shape [..., n_doppler, n_range].

    Examples:
        Dummy data: 10 frames, 3 antennas, 64 chirps, 128 samples.

        >>> raw_data = np.random.rand(10,3,64,128)
        >>> rdi = range_doppler_fft(raw_data)

        Use window function. (Available in `scipy https://docs.scipy.org/doc/scipy/reference/signal.windows.html`)

        >>> from scipy import signal
        >>> rdi = range_doppler_fft(raw_data, signal.windows.blackman, signal.windows.blackman)

        If window function uses additional parameters you can wrap it with lambda.

        >>> range_window_func = signal.windows.blackman
        >>> doppler_window_func = lambda x: signal.windows.chebwin(x, at=100)
        >>> rdi  = range_doppler_fft(raw_data, range_window_func, doppler_window_func)

    """

    x_range = range_fft(x, range_window, remove_mean)
    x_rdi = doppler_fft(x_range, doppler_window)

    return x_rdi
