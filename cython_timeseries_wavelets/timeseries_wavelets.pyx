import numpy as np
cimport numpy as cnp 
from pywt import threshold
from pywt._extensions._dwt import dwt_single, idwt_single
from pywt._extensions._pywt import Wavelet, Modes
def wavelet_smoothing_historical(
    cnp.ndarray[double, ndim=1] signal,  
    str wavelet_name,                    
    int level,
    double threshold_value,
    str mode = 'symmetric'   
):
    """
    Perform wavelet smoothing without lookahead bias
    """

    cdef int t, n = signal.shape[0]
    cdef double[:] current_signal, cA, cD
    cdef list coeffs = [] # stores cA and cD
    wavelet = Wavelet(wavelet_name)

    cdef cnp.ndarray[double, ndim=1] smoothed_signal = np.zeros(n, dtype=np.float64)

    for t in range(1, n+1):
        current_signal = signal[:t]
        coeffs = []

        # Decompose the signal
        for _ in range(level):
            cA, cD = dwt_single(current_signal, wavelet, mode.encode())
            
            # Threshold the high frequency contribution
            cD = threshold(cD, threshold_value, mode='soft')

            coeffs.append((cA, cD))
            current_signal = cA

        # Reconstruct the signal
        current_signal = coeffs[-1][0] # -1 is the final level, 0 is cA
        for cA, cD in reversed(coeffs[:-1]):
            current_signal = idwt_single(cA, cD, wavelet.rec_lo, wavelet.rec_hi, mode.encode())

        smoothed_signal[t - 1] = current_signal[-1]
    return

def wavelet_smoothing_rolling(
    cnp.ndarray[double, ndim=1] signal,
    str wavelet_name,
    int level,
    double threshold_value,
    int window_size, 
    str mode = 'symmetric'
):
    """
    Perform wavelet smoothing over a rolling window without lookahead bias.
    """

    cdef int t, n = signal.shape[0]
    cdef double[:] current_signal, cA, cD
    cdef list coeffs = [] 
    wavelet = Wavelet(wavelet_name)

    cdef cnp.ndarray[double, ndim=1] smoothed_signal = np.zeros(n, dtype=np.float64)

    for t in range(1, n+1):
        if t<= window_size:
            current_signal = signal[:t]
        else:
            current_signal = signal[t - window_size:t]

        coeffs = []

        # Decomposition
        for _ in range(level):
            cA, cD = dwt_single(current_signal, wavelet, mode.encode())
            
            # Threshold the high frequency contribution
            cD = threshold(cD, threshold_value, mode='soft')

            coeffs.append((cA, cD))
            current_signal = cA

        # Reconstruct the signal
        current_signal = coeffs[-1][0] # -1 is the final level, 0 is cA
        for cA, cD in reversed(coeffs[:-1]):
            current_signal = idwt_single(cA, cD, wavelet.rec_lo, wavelet.rec_hi, mode.encode())

        smoothed_signal[t - 1] = current_signal[-1]

    return smoothed_signal