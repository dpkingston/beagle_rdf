# Copyright (c) 2026 Douglas P. Kingston III. MIT License — see LICENSE.
"""
Band-pass (or low-pass) FIR filter + integer decimation.

Maintains filter state across process() calls so consecutive buffers
produce a continuous output stream -- essential for correct sample-index
arithmetic in DeltaComputer.

Performance
-----------
Uses scipy.signal.upfirdn which natively decimates (only computes the
output samples that survive downsampling), avoiding the ~D× wasted work
of ``lfilter`` which computes every sample then strides.  Cross-buffer
continuity is maintained by prepending the last (num_taps-1) input
samples as a history prefix.

On macOS, an optional vDSP_desamp backend via Apple Accelerate provides
an additional ~10-40× speedup by exploiting hardware-vectorised
strided FIR convolution.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.signal import firwin, upfirdn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Apple Accelerate vDSP backend (macOS only)
# ---------------------------------------------------------------------------
_vDSP_desamp = None

def _try_load_vdsp():
    """Try to load vDSP_desamp from Apple Accelerate. Returns callable or None."""
    import ctypes
    import ctypes.util
    import sys
    if sys.platform != "darwin":
        return None
    try:
        path = ctypes.util.find_library("Accelerate")
        if not path:
            return None
        lib = ctypes.cdll.LoadLibrary(path)
        fn = lib.vDSP_desamp
        fn.restype = None
        fn.argtypes = [
            ctypes.c_void_p,    # A (input)
            ctypes.c_long,      # I (stride/decimation)
            ctypes.c_void_p,    # B (filter taps, reversed)
            ctypes.c_void_p,    # C (output)
            ctypes.c_ulong,     # N (output count)
            ctypes.c_ulong,     # P (filter length)
        ]
        return fn
    except Exception:
        return None

try:
    _vDSP_desamp = _try_load_vdsp()
    if _vDSP_desamp is not None:
        logger.debug("vDSP_desamp available — using Accelerate for decimation")
except Exception:
    pass


class Decimator:
    """
    Single-stage FIR low-pass filter + integer decimation.

    Parameters
    ----------
    decimation : int
        Downsample factor.  output_rate = input_rate / decimation.
    input_rate_hz : float
        Sample rate of the incoming IQ stream.
    cutoff_hz : float
        Low-pass filter cutoff frequency (-6 dB point).
        Must be < input_rate_hz / 2.
    num_taps : int
        FIR filter length (odd number preferred).  Longer -> sharper rolloff
        but more latency.  Default 127 gives ~80 dB stopband at 2× cutoff.
    """

    def __init__(
        self,
        decimation: int,
        input_rate_hz: float,
        cutoff_hz: float,
        num_taps: int = 127,
    ) -> None:
        if decimation < 1:
            raise ValueError(f"decimation must be >= 1, got {decimation}")
        if cutoff_hz <= 0 or cutoff_hz >= input_rate_hz / 2:
            raise ValueError(
                f"cutoff_hz must be in (0, {input_rate_hz / 2}), got {cutoff_hz}"
            )
        if num_taps < 1:
            raise ValueError(f"num_taps must be >= 1, got {num_taps}")

        self._decimation = decimation
        self._input_rate_hz = float(input_rate_hz)

        nyquist = input_rate_hz / 2.0
        taps = firwin(num_taps, cutoff_hz / nyquist, window="hamming")
        # float32 taps: half the memory bandwidth, better SIMD utilisation.
        self._taps = taps.astype(np.float32)
        self._ntaps = len(self._taps)

        # Reversed taps for vDSP_desamp (correlation form).
        self._taps_rev = self._taps[::-1].copy()

        # History buffer for cross-buffer FIR state.
        # Padded to a multiple of decimation so upfirdn's decimation
        # grid aligns with the global sample grid.
        self._pad_len = decimation * ((self._ntaps - 1 + decimation - 1) // decimation)
        self._skip = self._pad_len // decimation

        # Initialise history (real and imaginary, for split-path processing)
        self._hist_re = np.zeros(self._pad_len, dtype=np.float32)
        self._hist_im = np.zeros(self._pad_len, dtype=np.float32)

        # Pre-allocated extended buffers (resized on first call if needed)
        self._ext_re: np.ndarray | None = None
        self._ext_im: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def decimation(self) -> int:
        return self._decimation

    @property
    def input_rate_hz(self) -> float:
        return self._input_rate_hz

    @property
    def output_rate_hz(self) -> float:
        return self._input_rate_hz / self._decimation

    @property
    def group_delay_samples(self) -> int:
        """Group delay of the FIR filter in *output* samples."""
        return (self._ntaps - 1) // 2 // self._decimation

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, iq: np.ndarray) -> np.ndarray:
        """
        Filter and decimate a complex IQ buffer.

        Parameters
        ----------
        iq : np.ndarray, dtype complex64 or complex128
            Input samples at input_rate_hz.

        Returns
        -------
        np.ndarray, dtype complex64
            Decimated samples at output_rate_hz.
            Length = len(iq) // decimation  (any remainder is consumed by
            the filter but not emitted; state is preserved for the next call).
        """
        iq = np.asarray(iq, dtype=np.complex64)
        n = len(iq)
        if n == 0:
            return np.empty(0, dtype=np.complex64)

        n_out = n // self._decimation
        d = self._decimation
        pad = self._pad_len
        skip = self._skip
        ntaps = self._ntaps

        if _vDSP_desamp is not None:
            return self._process_vdsp(iq, n, n_out, d, pad, skip, ntaps)
        return self._process_upfirdn(iq, n, n_out, d, pad, skip, ntaps)

    def _process_vdsp(self, iq, n, n_out, d, pad, skip, ntaps):
        """Decimation via Apple vDSP_desamp (macOS only)."""
        # Ensure pre-allocated work buffers are the right size
        ext_len = pad + n
        if self._ext_re is None or len(self._ext_re) != ext_len:
            self._ext_re = np.empty(ext_len, dtype=np.float32)
            self._ext_im = np.empty(ext_len, dtype=np.float32)

        # Build extended input: [history | new data]
        self._ext_re[:pad] = self._hist_re
        self._ext_re[pad:] = iq.real
        self._ext_im[:pad] = self._hist_im
        self._ext_im[pad:] = iq.imag

        out_re = np.empty(n_out, dtype=np.float32)
        out_im = np.empty(n_out, dtype=np.float32)

        _vDSP_desamp(
            self._ext_re.ctypes.data, d,
            self._taps_rev.ctypes.data,
            out_re.ctypes.data, n_out, ntaps,
        )
        _vDSP_desamp(
            self._ext_im.ctypes.data, d,
            self._taps_rev.ctypes.data,
            out_im.ctypes.data, n_out, ntaps,
        )

        out = np.empty(n_out, dtype=np.complex64)
        out.real = out_re
        out.imag = out_im

        # Save history: last (ntaps-1) input samples, zero-padded to pad_len
        self._hist_re[:] = 0.0
        self._hist_re[-(ntaps - 1):] = iq.real[-(ntaps - 1):]
        self._hist_im[:] = 0.0
        self._hist_im[-(ntaps - 1):] = iq.imag[-(ntaps - 1):]

        return out

    def _process_upfirdn(self, iq, n, n_out, d, pad, skip, ntaps):
        """Decimation via scipy upfirdn (cross-platform)."""
        # Build extended input: [history | new data]
        ext_re = np.empty(pad + n, dtype=np.float32)
        ext_re[:pad] = self._hist_re
        ext_re[pad:] = iq.real

        ext_im = np.empty(pad + n, dtype=np.float32)
        ext_im[:pad] = self._hist_im
        ext_im[pad:] = iq.imag

        re_out = upfirdn(self._taps, ext_re, down=d)[skip:skip + n_out]
        im_out = upfirdn(self._taps, ext_im, down=d)[skip:skip + n_out]

        out = np.empty(n_out, dtype=np.complex64)
        out.real = re_out
        out.imag = im_out

        # Save history
        self._hist_re[:] = 0.0
        self._hist_re[-(ntaps - 1):] = iq.real[-(ntaps - 1):]
        self._hist_im[:] = 0.0
        self._hist_im[-(ntaps - 1):] = iq.imag[-(ntaps - 1):]

        return out

    def prime(self, iq: np.ndarray) -> None:
        """Feed IQ samples through the filter to warm up the history buffer.

        The output is discarded — this only updates the internal FIR state
        so that the next ``process()`` call starts with current filter history
        rather than stale data from a previous block.

        Typically called with the last ``pad_len`` raw samples of the settling
        period, which are on the target frequency with the PLL locked but
        are otherwise discarded for carrier detection.
        """
        if len(iq) == 0:
            return
        self.process(iq)  # updates history; output is discarded

    def prime_with_replica(self, iq: np.ndarray) -> None:
        """Prime the filter by replicating a short segment of usable data.

        Takes the first ``pad_len`` samples of the usable block and tiles
        them to create enough synthetic history for the FIR to reach steady
        state.  This ensures the filter output from sample zero matches what
        it would produce if the same signal had been present for a long time
        — eliminating the power ramp caused by mismatched history (e.g.
        settling-period data with different carrier state than the block).

        Parameters
        ----------
        iq : np.ndarray, complex64
            The full usable IQ block (pre-decimation, post-DC-removal).
            Only the first ``pad_len`` samples are used.
        """
        if len(iq) == 0:
            return
        segment = np.asarray(iq[:self._pad_len], dtype=np.complex64)
        if len(segment) == 0:
            return
        # Tile the segment to fill at least 2× pad_len so the filter
        # history is fully flushed with representative data.
        reps = max(2, (2 * self._pad_len + len(segment) - 1) // len(segment))
        tiled = np.tile(segment, reps)
        self.process(tiled)  # output discarded; history updated

    @property
    def prime_length(self) -> int:
        """Minimum number of raw IQ samples needed to fully prime the filter."""
        return self._pad_len

    def reset(self) -> None:
        """Reset filter state to zeros (use when starting a new stream)."""
        self._hist_re[:] = 0.0
        self._hist_im[:] = 0.0
