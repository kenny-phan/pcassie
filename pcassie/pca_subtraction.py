import numpy as np
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from utility_functions import debug_print

def convert_range_to_indices(wave, start, end):
    """Convert a wavelength range to indices."""
    start_index = np.searchsorted(wave, start)
    end_index = np.searchsorted(wave, end)
    return start_index, end_index


def preprocess(spectra):
    """
    Normalize by the median spectrum, subtract the median at each wavelength,
    and divide each spectrum by its own standard deviation.
    """
    norm_flux = spectra / np.median(spectra)  # global normalization
    median_flux = np.median(norm_flux, axis=0)
    median_subtracted = norm_flux - median_flux

    # Use norm for per-spectrum std estimation
    row_std = np.linalg.norm(median_subtracted, axis=1, keepdims=True) / np.sqrt(median_subtracted.shape[1])
    return median_subtracted / row_std


def compute_covariance_matrix(data):
    """Compute the covariance matrix using NumPy (faster than pandas)."""
    centered = data - np.mean(data, axis=0)
    return centered.T @ centered / (data.shape[0] - 1)


def compute_eigenvalues_and_vectors(cov_matrix):
    """Compute and sort eigenvalues/eigenvectors in descending order."""
    jax_cov_matrix = jnp.array(cov_matrix, dtype=jnp.float64)
    evals, evecs = jnp.linalg.eigh(jax_cov_matrix)
    idx = jnp.argsort(evals)[::-1]

    evals_sorted = np.array(evals[idx])
    evecs_sorted = np.array(evecs[:, idx])

    return evals_sorted, evecs_sorted


def explained_variance(eigenvalues):
    """Calculate explained variance ratio."""
    return eigenvalues / np.sum(eigenvalues)


def remove_components(data, eigenvectors, first_comps=0, last_comps=0, verbose=False):
    """Remove specified principal components from the data."""
    total_comps = eigenvectors.shape[1]
    start_comps = first_comps
    end_comps = total_comps - last_comps

    if start_comps >= end_comps:
        debug_print(verbose, f"total # of components: {total_comps}. removing {start_comps} from the beginning and {last_comps} from the end")
        raise ValueError("Requested to remove all components — nothing left to reconstruct from.")

    proj_matrix = eigenvectors[:, start_comps:end_comps]
    projected = data @ proj_matrix
    return projected @ proj_matrix.T


def pca_subtraction(spectra, start_idx, end_idx, first_comps=0, last_comps=0, pre=False, verbose=False):
    """
    Perform PCA subtraction in a wavelength slice from `start_idx` to `end_idx`.

    Args:
        spectra (np.ndarray): 2D array of shape (num_spectra, num_wavelengths).
        start_idx (int): Start index for PCA region.
        end_idx (int): End index for PCA region.
        first_comps (int): Components to remove from the beginning.
        last_comps (int): Components to remove from the end.
        pre (bool): Whether to apply preprocessing first.

    Returns:
        (tdm_result, wdm_result): PCA-subtracted arrays.
    """
    if pre:
        spectra = preprocess(spectra)

    spectra_slice = spectra[:, start_idx:end_idx]
    tdm = spectra_slice.T  # Transpose for TDM
    wdm = spectra_slice     # WDM as-is

    # Fast covariance
    tdm_cov = compute_covariance_matrix(tdm)
    wdm_cov = compute_covariance_matrix(wdm)

    _, evec_tdm = compute_eigenvalues_and_vectors(tdm_cov)
    _, evec_wdm = compute_eigenvalues_and_vectors(wdm_cov)

    debug_print(verbose, "tdm, wdm evec shapes:", evec_tdm.shape, evec_wdm.shape)

    # PCA removal
    tdm_clean = remove_components(tdm, evec_tdm, first_comps, last_comps)
    wdm_clean = remove_components(wdm, evec_wdm, first_comps, last_comps)

    return tdm_clean.T, wdm_clean
