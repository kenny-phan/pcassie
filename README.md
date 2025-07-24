# PCA Telluric Removal

Code based on M. Damiano, G. Micela, and G. Tinetti (2019). _[A Principal Component Analysis-based Method to Analyze High-resolution Spectroscopic Data on Exoplanets
](https://doi.org/10.3847/1538-4357/ab22b2)_

Originally developed for CRIRES+ data.

The code is structured as follows. All programs are .py files in the 'pipeline' folder. pipeline.py compiles the nessecary functions into one 'pipeline' function. The pipeline function optionally recalibrates the pixel-wavelength relation of the data spectrum to a telluric model from EsoSky (Noll et al. 2012, Jones et al. 2013), then will perform PCA subtraction on the data (from pca_subtraction.py) and cross-correlation with a user input simulated spectrum (from ccf.py). For the development of this codebase, I used MultiRex (Duque-Castaño et al. 2024), a straightforward implementation of TauRex (Al-Refaie et al. 2019). There are additional functions to inject the data with the simulated spectrum, create a S/N map, and perform a Welch's T-test in ccf.py.  
