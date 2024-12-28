import numpy as np
import time
from cython_timeseries_wavelets.timeseries_wavelets import ( # type: ignore
    wavelet_smoothing_historical,
    wavelet_smoothing_rolling
)


# Example signal
n = int(1e5)
signal = np.random.randn(n)

# Smoothing parameters
wavelet_name = 'db6'
level = 4
threshold_value = 0.5
window_size = 50

# Historical smoothing
t0 = time.time()

smoothed_historical = wavelet_smoothing_historical(
    signal,
    wavelet_name=wavelet_name,
    level=level,
    threshold_value=threshold_value
)
t1 = time.time()
print(f'Historical finished running in {t1-t0} with n={n}')
# Rolling smoothing

smoothed_rolling = wavelet_smoothing_rolling(
    signal,
    wavelet_name=wavelet_name,
    level=level,
    threshold_value=threshold_value,
    window_size=window_size
)
t2 = time.time()
print(f'Rolling finished running in {t2-t1} with n={n}')

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(signal, label="Original Signal")
plt.plot(smoothed_historical, label="Smoothed (Historical)", alpha=0.8)
plt.plot(smoothed_rolling, label="Smoothed (Rolling)", alpha=0.8)
plt.legend()
plt.show()
