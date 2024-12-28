import os
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

os.makedirs("cython_timeseries_wavelets", exist_ok=True)

setup(
    # name="cython_timeseries_wavelets",
    # package_dir={"": "."}, # specifies the root directory 
    ext_modules=cythonize("timeseries_wavelets.pyx", annotate=True),
    include_dirs=[np.get_include()]
)