import numpy as np
import glob
import os

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

def remove_components(data, eigenvectors, num_components):
    """Reconstruct data matrix without the first num_components eigenvectors."""

    projection_matrix = eigenvectors[:, num_components:-1]
    projected_data = np.dot(data, projection_matrix)

    return np.dot(projected_data, projection_matrix.T)

def run_pca_subtraction(spectra, wave, start_wav, end_wav, component_count):
    """Runs PCA subtraction on the provided spectra within a specified wavelength range.
    Args:
        spectra (np.ndarray): 2D array of shape (num_spectra, num_wavelengths).
        wave (np.ndarray): 1D array of wavelengths corresponding to the spectra.
        start_wav (float): Start wavelength for PCA analysis.
        end_wav (float): End wavelength for PCA analysis.
        component_count (int): Number of principal components to remove."""

    preprocessed_array = preprocess(spectra)

    start_idx, end_idx = convert_range_to_indices(wave, start_wav, end_wav)

    tdm_covariance = np.cov(preprocessed_array[:, start_idx:end_idx].T, rowvar=False)
    wdm_covariance = np.cov(preprocessed_array[:5, start_idx:end_idx], rowvar=False)

    eval_tdm, evec_tdm  = compute_eigenvalues_and_vectors(tdm_covariance)
    eval_wdm, evec_wdm = compute_eigenvalues_and_vectors(wdm_covariance)

    # Remove components from TDM and WDM
    tdm_reconstructed = remove_components(preprocessed_array[:, start_idx:end_idx].T, evec_tdm, component_count)
    wdm_reconstructed = remove_components(preprocessed_array[:, start_idx:end_idx], evec_wdm, component_count)

    return tdm_reconstructed, wdm_reconstructed