import numpy as np
import astropy.units as u

from astropy.constants import c
from scipy.interpolate import interp1d

def doppler_shift(wave_arr, velocity):
    """velocity in m/s"""
    return wave_arr * (1 + (velocity * u.m / u.s )/ c)

def cross_correlation_function(x, y, d):
    """
    Compute the normalized cross-correlation coefficient (CCF) of two 1D arrays x and y at delay d.
    Positive d means y is shifted right (delayed).
    Returns the correlation coefficient at delay d.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    n = min(len(x), len(y))

    if d > 0:
        if d >= n:
            return 0.0
        x_seg = x[:n-d]
        y_seg = y[d:n]
    elif d < 0:
        if -d >= n:
            return 0.0
        x_seg = x[-d:n]
        y_seg = y[:n+d]
    else:
        x_seg = x[:n]
        y_seg = y[:n]

    if len(x_seg) == 0 or len(y_seg) == 0:
        return 0.0

    x_mean = np.mean(x_seg)
    y_mean = np.mean(y_seg)
    numerator = np.sum((x_seg - x_mean) * (y_seg - y_mean))
    denominator = np.sqrt(np.sum((x_seg - x_mean)**2)) * np.sqrt(np.sum((y_seg - y_mean)**2))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def compute_ccf_over_velocity_range(data_wave, data_flux, model_flux, velocity_range):
    """
    Compute CCF between observed data and Doppler-shifted model over a velocity grid.

    Parameters:
        data_wave (1D array): Wavelength grid of observed data
        data_flux (1D array): Observed flux
        model_flux (1D array): Model flux defined on same wave grid
        velocity_range (1D array): Velocities (m/s) to Doppler shift model to

    Returns:
        velocities (1D array), ccfs (1D array)
    """
    ccfs = []
    interp_model = interp1d(data_wave, model_flux, kind='linear', bounds_error=False, fill_value=np.nan)

    for v in velocity_range:
        shifted_wave = doppler_shift(data_wave, v)
        shifted_model_flux = interp_model(shifted_wave)

        # Mask NaNs introduced by shifting out of bounds
        valid = ~np.isnan(shifted_model_flux) & ~np.isnan(data_flux)
        if np.sum(valid) < 10:
            ccfs.append(0.0)
            continue

        cc = cross_correlation_function(data_flux[valid], shifted_model_flux[valid], d=0)
        ccfs.append(cc)

    return velocity_range, np.array(ccfs)