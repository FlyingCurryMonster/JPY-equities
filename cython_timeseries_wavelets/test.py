import numpy as np
from cython_timeseries_wavelets.timeseries_wavelets import ( # type: ignore
    wavelet_smoothing_historical,
    wavelet_smoothing_rolling
)


# Example signal
signal = np.random.randn(1000)

# Smoothing parameters
wavelet_name = 'db6'
level = 4
threshold_value = 0.5
window_size = 50

# Historical smoothing
smoothed_historical = wavelet_smoothing_historical(
    signal,
    wavelet_name=wavelet_name,
    level=level,
    threshold_value=threshold_value
)

# Rolling smoothing
smoothed_rolling = wavelet_smoothing_rolling(
    signal,
    wavelet_name=wavelet_name,
    level=level,
    threshold_value=threshold_value,
    window_size=window_size
)

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(signal, label="Original Signal")
plt.plot(smoothed_historical, label="Smoothed (Historical)", alpha=0.8)
plt.plot(smoothed_rolling, label="Smoothed (Rolling)", alpha=0.8)
plt.legend()
plt.show()
