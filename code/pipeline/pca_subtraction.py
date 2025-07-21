import numpy as np
import pandas as pd

from pipeline.utility_functions import split_normalize

def convert_range_to_indices(wave, start, end):
    """Convert a wavelength range to indices."""
    start_index = np.searchsorted(wave, start)
    end_index = np.searchsorted(wave, end)
    return start_index, end_index

def preprocess(spectra):
    """ 
    Takes a spectral array (axis=0 wavelength bins, axis=1 individual spectra), 
    divides by median spectra, subtracts the median, 
    and divides each spectra by its own standard deviation. 
    """

    # Normalize by the median of this spectrum
    norm_flux = spectra / np.median(spectra)

    # Compute the median at each wavelength (column)
    median_flux = np.median(norm_flux, axis=0)

    # Subtract the median from each spectrum
    median_subtracted_flux = norm_flux - median_flux  # shape: (num_spectra, num_wavelengths)

    # Compute the standard deviation for each spectrum (row)
    row_std = np.std(median_subtracted_flux, axis=1, keepdims=True)  # shape: (num_spectra, 1)

    # Divide each row by its own standard deviation
    return median_subtracted_flux / row_std  # shape: (num_spectra, num_wavelengths)

def compute_eigenvalues_and_vectors(covariance_matrix):
    """Compute eigenvalues and eigenvectors of the covariance matrix."""

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

def explained_variance(eigenvalues):
    """Calculate the explained variance from eigenvalues."""

    total_variance = np.sum(eigenvalues)
    return eigenvalues / total_variance

def remove_components(data, eigenvectors, first_comps=0, last_comps=0):
    """
    Reconstruct data matrix with the first and/or last eigenvectors removed.

    Parameters:
        data (ndarray): Data matrix (observations × features).
        eigenvectors (ndarray): Eigenvectors (features × components).
        first_comps (int): Number of components to remove from the start.
        last_comps (int): Number of components to remove from the end.

    Returns:
        ndarray: Reconstructed data with selected components removed.
    """
    total_comps = eigenvectors.shape[1]
    
    # Compute the indices to keep
    start = first_comps
    end = total_comps - last_comps

    if start >= end:
        raise ValueError("Requested to remove all components — nothing left to reconstruct from.")

    projection_matrix = eigenvectors[:, start:end]
    projected_data = np.dot(data, projection_matrix)
    
    return np.dot(projected_data, projection_matrix.T)

def pca_subtraction(spectra, start_idx, end_idx, first_comps=0, last_comps=0, pre=False):
    """Runs PCA subtraction on the provided spectra within a specified wavelength range.
    Args:
        spectra (np.ndarray): 2D array of shape (num_spectra, num_wavelengths).
        wave (np.ndarray): 1D array of wavelengths corresponding to the spectra.
        start_wav (float): Start wavelength for PCA analysis.
        end_wav (float): End wavelength for PCA analysis.
        component_count (int): Number of principal components to remove."""

    if pre: 
        spectra = preprocess(spectra)
    else:
        spectra = spectra

    tdm_df = pd.DataFrame(spectra[:, start_idx:end_idx].T)
    wdm_df = pd.DataFrame(spectra[:, start_idx:end_idx])

    tdm_covariance = tdm_df.cov().values
    wdm_covariance = wdm_df.cov().values

    eval_tdm, evec_tdm  = compute_eigenvalues_and_vectors(tdm_covariance)
    eval_wdm, evec_wdm = compute_eigenvalues_and_vectors(wdm_covariance)

    #print(f"spectra shape: {spectra.shape}")

    # Remove components from TDM and WDM
    tdm_reconstructed = remove_components(spectra[:, start_idx:end_idx].T, evec_tdm, first_comps, last_comps)
    wdm_reconstructed = remove_components(spectra[:, start_idx:end_idx], evec_wdm, first_comps, last_comps)

    #print(f"shapes of tdm_reconstructed and wdm_reconstructed: {tdm_reconstructed.shape}, {wdm_reconstructed.shape}")

    return tdm_reconstructed, wdm_reconstructed

def run_pca_on_detector_segments(flux, wave, first_comps=0, last_comps=0, pre=False, gap_size_px=5):
    """
    For a given index in stacked_spectra_perstep.npz, split the spectrum into detector segments
    using split_normalize() gaps, run PCA subtraction on each, and concatenate the results.
    Returns concatenated tdm, wdm, and wavelength arrays.
    """

    _, gaps = split_normalize(wave, flux[0], gap_size_px)
    
    # Section edges: start, all gaps+1, end
    section_edges = np.concatenate(([0], gaps + 1, [len(wave)]))
    
    n_spectra, n_pixels = flux.shape
    wave = np.zeros((n_pixels))
    tdm_reconstructed = np.zeros((n_spectra, n_pixels))
    wdm_reconstructed = np.zeros((n_spectra, n_pixels))

    # For each detector segment
    for i in range(len(section_edges) - 1):
        start, end = section_edges[i], section_edges[i+1]
        print(f"Processing segment {i}: start={start}, end={end}, length={end - start}")
        seg_flux = flux[:, start:end]  # shape: (n_spectra, segment_length)
        seg_wave = wave[start:end]

        tdm, wdm = pca_subtraction(seg_flux, 0, seg_flux.shape[1], first_comps, last_comps, pre=pre)
        tdm_reconstructed[:, start:end] = tdm.T
        wdm_reconstructed[:, start:end] = wdm
        wave[start:end] = seg_wave

    return tdm_reconstructed, wdm_reconstructed, wave