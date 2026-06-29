"""Generic digital-signal-processing helpers (filters, STFT, envelopes)."""

from data_analysis.signal.core import (
    downsample_stride,
    downsample_blockmean,
    downsample_decimate,
    DOWNSAMPLE_METHODS,
    analyze_downsample_options,
    amplitude_spectrum,
    avg_amplitude_spectrum,
)

__all__ = [
    "downsample_stride",
    "downsample_blockmean",
    "downsample_decimate",
    "DOWNSAMPLE_METHODS",
    "analyze_downsample_options",
    "amplitude_spectrum",
    "avg_amplitude_spectrum",
]
