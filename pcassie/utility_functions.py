import numpy as np

# Functions to support normalization of CRIRES+ data

def normalize_zero_one(spectrum):
    median = np.nanmedian(spectrum)
    std = np.nanstd(spectrum)

    clean = spectrum.copy()

    outliers = np.abs(clean - median) > 3 * std
    clean[outliers] = median

    zero_to_one = (clean - np.min(clean)) / (np.max(clean) - np.min(clean))

    return zero_to_one


def split_divide_by_median(wave, flux, m):
    """
    Splits the spectrum into sections based on gaps in the wavelength array,
    normalizes each section by dividing by its median, and returns the normalized flux,
    the gap indices, and the segment index pairs (start, end).
    """
    valid_wave = wave[~np.isnan(wave)]
    gap_threshold = m * np.median(np.diff(valid_wave))
    gaps = np.where(np.diff(wave) > gap_threshold)[0]

    section_edges = np.concatenate(([0], gaps + 1, [len(wave)]))

    norm_flux = np.full_like(flux, np.nan)
    segment_indices = []

    for i in range(len(section_edges) - 1):
        start, end = section_edges[i], section_edges[i + 1]
        segment_indices.append((start, end))

        section = flux[start:end]
        valid_mask = ~np.isnan(section)
        if not np.any(valid_mask):
            continue

        median = np.nanmedian(section[valid_mask])
        std = np.nanstd(section[valid_mask])

        section_clean = section.copy()
        outliers = np.abs(section_clean - median) > 3 * std
        section_clean[outliers & valid_mask] = median

        section_norm = section_clean / median
        norm_flux[start:end] = section_norm

    return norm_flux, segment_indices


def split_detectors(wave, flux, m=5):
    normalized_flux = []

    for ii in range(len(flux)):
        single_flux = flux[ii, :]
        single_wave = wave[ii, :]
        
        single_norm_flux, segment_indices = split_divide_by_median(single_wave, single_flux, m)
        
        normalized_flux.append(single_norm_flux)

    normalized_flux_array = np.array(normalized_flux)

    return normalized_flux_array, segment_indices 


def mask_gap_edges(wave, gaps, n):
    """
    Returns a boolean mask that is True for points to be masked:
    - The first n points after each gap
    - The last n points before each gap
    - Optionally, the first n and last n points of the array
    """
    mask = np.zeros_like(wave, dtype=bool)
    for g in gaps:
        # Mask last n points before the gap (ending at g)
        start = max(0, g - n + 1)
        mask[start:g+1] = True
        # Mask first n points after the gap (starting at g+1)
        end = min(len(wave), g + 1 + n)
        mask[g+1:end] = True
    # Optionally, mask the first n and last n points of the array
    mask[:n] = True
    mask[-n:] = True
    return mask


def debug_print(verbose, *args, **kwargs):
    """General function to allow for toggleable print statements."""
    if verbose:
        print(*args, **kwargs)
