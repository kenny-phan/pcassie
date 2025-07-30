# Follows the steps of Damiano 2018, Sec. 5.3

"""
1. normalize each spectrum of each detector dividing by its median
2. take the mean spectrum of each detector
3. take all lines <0.8 & also in telluric template
4. fit each line to a gaussian, take the centroid
5. plot pixel position vs. wavelength
6. fit a polynomial to the centroid positions until no correlated residuals
7. find fit precision: deltaV (velocity) = std(residuals) * speed of light / central spectrum wavelength
8. interpolate all single spectra by a third order spline to derived wavelength grid (converting pixel to wavelength via the fit eqn)
"""

import numpy as np

from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import resample, find_peaks
from astropy.constants import c

from pcassie.utility_functions import split_divide_by_median

def gaussian(x, amp, mu, sigma, offset):
    """A simple gaussian function."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def fit_gaussian_to_peaks(x, y, peaks, window=5):
    """Fit peak absoption features to gaussians."""
    centroids = []
    for idx in peaks:
        left = max(0, idx - window)
        right = min(len(x), idx + window + 1)
        x_fit = x[left:right]
        y_fit = y[left:right]
        # Initial guesses
        amp_guess = np.min(y_fit) - np.median(y_fit)
        mu_guess = x[idx]
        sigma_guess = (x_fit[-1] - x_fit[0]) / 6 if len(x_fit) > 1 else 1
        offset_guess = np.median(y_fit)
        try:
            popt, _ = curve_fit(
                gaussian, x_fit, y_fit,
                p0=[amp_guess, mu_guess, sigma_guess, offset_guess]
            )
            centroids.append(popt[1])
        except Exception:
            centroids.append(mu_guess)
    return np.array(centroids)


def fit_segments_to_wavelengths(segment_centroids, tel_centroid_dict, telluric_wavelength, deg=3):
    """Fits individual detectors to a new wavelength grid."""
    fits = {}

    pixel_grid = np.arange(len(telluric_wavelength))
    interp_wave = interp1d(pixel_grid, telluric_wavelength, kind='linear', bounds_error=False, fill_value=np.nan)

    # Interpolator: wavelength to pixel index
    interp_idx = interp1d(telluric_wavelength, np.arange(len(telluric_wavelength)), 
                      kind='linear', bounds_error=False, fill_value=np.nan)

    tel_centroids_pixel = {seg_id: interp_idx(waves) for seg_id, waves in tel_centroid_dict.items()}

    for seg_id in segment_centroids:
        data_pix = segment_centroids[seg_id]
        tel_pix = tel_centroids_pixel[seg_id]

        # Skip if either segment is empty
        if len(data_pix) == 0 or len(tel_pix) == 0:
            continue

        # Interpolate wavelength at telluric centroids
        wave_vals = interp_wave(tel_pix)

        # Fit pixel → wavelength
        mask = ~np.isnan(data_pix) & ~np.isnan(wave_vals)
        p = Polynomial.fit(data_pix[mask], wave_vals[mask], deg=deg).convert()
        print(p)
        fits[seg_id] = p

    return fits


def precision(residuals, wave_arr):
    """A function to calculate the precision of the wavelength fit in terms of velocity. 
    Refernce https://discovery.ucl.ac.uk/id/eprint/10066066/7/Mario_Damiano_Thesis.pdf 
    pg. 117 for more info."""
    std = np.std(residuals)
    central_wave = np.median(wave_arr)
    return std * c / (central_wave)


def split_and_stack(arr, gaps):
    """Rearranges your spectrum into a 3d array with (detector, spectra, flux); 
    i.e. a shape of (n detector x n spectra x wavelength range)"""
    segments = []
    gaps = np.concatenate(([0], gaps))
    for i in range(len(gaps) - 1):
        segments.append(arr[gaps[i]:gaps[i + 1]].astype(float))  # cast to float
    segments.append(arr[gaps[-1]:].astype(float))
    maxlen = max(len(seg) for seg in segments)
    stacked = np.array([np.pad(seg, (0, maxlen - len(seg)), constant_values=np.nan) for seg in segments])
    return stacked


def calibrate_cr2res(data_wave, data_flux, telluric_wave, telluric_flux, gap_size_px=5, poly_order=5):
    """
    Runs a full calibration of your spectrum. 

    Args: 
    data_flux is an array of n spectra x wavelength array
    data_wave is a 1d wavelength array
    telluric_flux is a 1d telluric flux array
    telluric_wave is a 1d telluric wavelength array

    Ensure the wavelegnth units match :)
    """

    #1.1 mask data of NaNs
    pixels = np.arange(len(data_wave))
    valid_mask = ~np.isnan(data_flux).any(axis=0)  # Mask out nans

    valid_flux = data_flux[:, valid_mask]
    valid_pixels = pixels[valid_mask]
    valid_wave = data_wave[valid_mask]    

    print("flux, wavelength array shape", data_flux.shape, data_wave.shape)

    print("masked flux, pixel array shape", valid_flux.shape, valid_pixels.shape)

    # 1.2 divide each spectrum of each detector by its median
    normalized_flux_array = []
    gaps_arr = []  # Store gaps for each spectrum

    for ii in range(len(valid_flux)):
        orig_flux = valid_flux[ii, :]
        
        orig_norm_flux, gaps = split_divide_by_median(valid_wave, orig_flux, gap_size_px)
        
        normalized_flux_array.append(orig_norm_flux)
        gaps_arr.append(gaps)

    # 2 Take the mean spectrum of each detector
    mean_flux = np.nanmean(normalized_flux_array, axis=0)

    # 3 interpolate tellurics to ~data from the given wavelength range
    # Mask telluric arrays to this range
    data_wave_min = np.min(data_wave) 
    data_wave_max = np.max(data_wave)

    # Mask telluric arrays to this range
    mask = (telluric_wave >= data_wave_min) & (telluric_wave <= data_wave_max)
    telluric_wave_masked = telluric_wave[mask]
    telluric_flux_masked = telluric_flux[mask]

    # Resample telluric arrays to match the shape of the data arrays
    telluric_wave_resampled = resample(telluric_wave_masked, len(valid_pixels))
    telluric_flux_resampled = resample(telluric_flux_masked, len(valid_pixels))

    # 3.2 Isolate data peaks <0.8
    all_peaks, _ = find_peaks(-mean_flux)  # Find minima
    data_peaks = all_peaks[mean_flux[all_peaks] < 0.8]

    # 3.3 Isolate top-N deepest telluric peaks
    all_tel_peaks, _ = find_peaks(-telluric_flux_resampled)
    peak_fluxes = telluric_flux_resampled[all_tel_peaks]

    N = len(data_peaks)
    top_idx = np.argsort(peak_fluxes)[:N]              # deepest N
    tel_peaks_unsorted = all_tel_peaks[top_idx]

    # Sort by wavelength (assuming telluric_wavelength is defined)
    tel_wavelengths = telluric_wave_resampled[tel_peaks_unsorted]
    sorted_idx = np.argsort(tel_wavelengths)

    tel_peaks = tel_peaks_unsorted[sorted_idx]

    # 4 fit peaks to gaussians
    data_centroids = fit_gaussian_to_peaks(valid_pixels, mean_flux, data_peaks)
    tel_centroids = fit_gaussian_to_peaks(telluric_wave_resampled, telluric_flux_resampled, tel_peaks)

    # 5.1.1 Separate data_centroids by the gaps they fall into

    # data_centroids are pixel indices; use valid_pixels[data_centroids] if needed for mapping to pixel values
    # If you want to use the pixel values directly, use data_centroids as is

    sep_data_centroids = []
    gaps_with_ends = np.concatenate(([0], gaps, [valid_pixels[-1]]))  # prepend 0 for the first segment

    for i in range(len(gaps_with_ends) - 1):
        start = gaps_with_ends[i]
        end = gaps_with_ends[i + 1]
        # Select centroids that fall within this segment
        mask = (data_centroids >= start) & (data_centroids < end)
        sep_data_centroids.append(data_centroids[mask])

    # sep_data_centroids is a list of arrays, one per segment
    for idx, arr in enumerate(sep_data_centroids):
        print(f"Segment {idx}: {len(arr)} centroids")

    segment_centroids = {seg_id: arr for seg_id, arr in enumerate(sep_data_centroids)}

    # Get segment lengths from data_centroids
    segment_lengths = [len(arr) for arr in segment_centroids.values()]

    # Split telluric_centroids into matching segments
    split_tel_centroids = np.split(tel_centroids, np.cumsum(segment_lengths)[:-1])

    # Build dict with same keys as segment_centroids
    tel_centroid_dict = {i: seg for i, seg in enumerate(split_tel_centroids)}
        
    # Run the fit
    segment_fits = fit_segments_to_wavelengths(segment_centroids, tel_centroid_dict, telluric_wave_resampled, deg=poly_order)

    #6 get residuals
    residuals_dict = {}

    for seg_id, poly in segment_fits.items():
        pixels = np.atleast_1d(segment_centroids[seg_id])
        true_wavelengths = np.atleast_1d(tel_centroid_dict[seg_id])

        if len(pixels) != len(true_wavelengths):
            print(f"Skipping segment {seg_id}: mismatched lengths")
            continue

        # Evaluate fit
        fitted_wavelengths = poly(pixels)
        residuals = fitted_wavelengths - true_wavelengths
        residuals_dict[seg_id] = residuals

    #7. find fit precision

    segment_precision = {}

    for seg_id, residual in residuals_dict.items():
        
        pix = segment_centroids[seg_id]
        if len(pix) == 0:
            continue

        # Use full pixel span for this segment
        full_pix_range = np.arange(int(np.min(pix)), int(np.max(pix)) + 1)

        # Evaluate fitted poly at full pixel range
        fitted_waves = poly(full_pix_range)

        # Remove any nan pairs
        mask = ~np.isnan(fitted_waves) 

        # Compute precision
        prec = precision(residual, fitted_waves[mask])
        segment_precision[seg_id] = prec

    # 8 interpolate new wavelength grid
    sep_pixels = split_and_stack(valid_pixels, gaps_arr[0])
    data_wavelengths = np.full_like(mean_flux, np.nan, dtype=float)

    # Convert each segment's pixels to wavelength using fitted polynomials
    for seg_id, poly in segment_fits.items():
        pixels = sep_pixels[seg_id]
        pixels = pixels.astype(int)

        # Ensure we don't go out of bounds
        pixels = pixels[(pixels >= 0) & (pixels < len(mean_flux))]
        data_wavelengths[pixels] = poly(pixels)

    return data_wavelengths