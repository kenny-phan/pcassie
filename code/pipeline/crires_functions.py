import numpy as np

# Functions to support normalization of CRIRES+ data

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

def split_divide_by_median(wave, flux, m):
    """
    Splits the spectrum into sections based on gaps in the wavelength array,
    normalizes each section by dividing by its median, and returns the normalized flux.
    """
    gap_threshold = m * np.median(np.diff(wave))  # adjust multiplier as needed
    gaps = np.where(np.diff(wave) > gap_threshold)[0] 

    section_edges = np.concatenate(([0], gaps + 1, [len(wave)]))

    norm_flux = np.zeros_like(flux)
    for i in range(len(section_edges) - 1):
        start, end = section_edges[i], section_edges[i+1]
        section = flux[start:end]
        median = np.nanmedian(section)
        std = np.nanstd(section)
        # Mask values outside 3 std
        outlier_mask = np.abs(section - median) > 3 * std
        section_clean = section.copy()
        section_clean[outlier_mask] = median
        section_norm = section_clean / median
        norm_flux[start:end] = section_norm

    return norm_flux, gaps

def split_normalize(wave, flux, m):
    
    gap_threshold = m * np.median(np.diff(wave))  # adjust multiplier as needed
    gaps = np.where(np.diff(wave) > gap_threshold)[0] 

    section_edges = np.concatenate(([0], gaps + 1, [len(wave)]))

    norm_flux = np.zeros_like(flux)
    for i in range(len(section_edges) - 1):
        start, end = section_edges[i], section_edges[i+1]
        section = flux[start:end]
        section_norm = (section - np.nanmedian(section)) / np.nanstd(section)
        norm_flux[start:end] = section_norm

    return norm_flux, gaps

def preprocess_crires_data(flux, wave):
    """
    Preprocess CRIRES+ data by normalizing each section of the spectrum,
    masking edges around gaps, and applying sigma clipping.
    """
    filtered_flux = []
    filtered_wave = []

    for i in range(flux.shape[0]):
        orig_flux = flux[i]
        orig_wave = wave[i]  # <-- Fix: use the i-th wave array

        # 1. Normalize each section of the spectrum
        orig_norm_flux, gaps = split_normalize(orig_wave, orig_flux)
        #print(f"Spectrum {i}: norm_flux min={np.nanmin(orig_norm_flux)}, max={np.nanmax(orig_norm_flux)}, nan count={np.sum(np.isnan(orig_norm_flux))}")
        
        # 2. Mask edges around gaps
        mask = mask_gap_edges(orig_wave, gaps, 10)
        #print(f"Spectrum {i}: {np.sum(mask)} masked, {np.sum(~mask)} unmasked, total {len(mask)}")

        # 3. Mask arrays
        filtered_flux.append(orig_norm_flux[~mask])
        filtered_wave.append(orig_wave[~mask])

    return filtered_flux, filtered_wave