import numpy as np
from numba import njit
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from pcassie.utility_functions import debug_print

def convert_range_to_indices(wave, start, end):
    """Convert a wavelength range to indices.
    
    Parameters
    ----------
    wave: array
        1d wavelength grid.
    start: float
        Starting wavelength value you want to crop to.
    end: float
        Ending wavelength value you want to crop to.

    Returns:
    --------
    int
        Index of the starting wavelength.
    int
        Index of the ending wavelength.
    """
    start_index = np.searchsorted(wave, start)
    end_index = np.searchsorted(wave, end)
    return start_index, end_index

# def forward_preprocess(spectra):
#     """
#     Returns:
#       preprocessed (n_rows, n_cols),
#       row_std (n_rows, 1),
#       col_med (1, n_cols)  # keep as 2D for easy broadcasting
#       global_med (scalar)
#     """
#     global_med = np.median(spectra)                     # scalar
#     norm_flux = spectra / global_med                    # shape (n_rows, n_cols)
#     col_med = np.median(norm_flux, axis=0, keepdims=True)  # shape (1, n_cols)
#     median_subtracted = norm_flux - col_med             # shape (n_rows, n_cols)

#     row_std = np.linalg.norm(median_subtracted, axis=1, keepdims=True) \
#               / np.sqrt(median_subtracted.shape[1])
#     row_std = np.where(row_std == 0.0, 1.0, row_std)    # avoid zeros

#     preprocessed = median_subtracted / row_std          # same shape (n_rows, n_cols)
#     return preprocessed, row_std, col_med, global_med


# def inverse_preprocess(preprocessed_slice, row_std, col_med, global_med, start_idx, end_idx):
#     """
#     Invert preprocessing for just the wavelength slice [start_idx:end_idx).
#     preprocessed_slice: shape (n_rows, n_slice_cols)
#     row_std: shape (n_rows, 1)
#     col_med: shape (1, n_cols_total)  <-- we'll index into the relevant slice
#     global_med: scalar
#     start_idx, end_idx: slice indices to pick the right columns from col_med
#     """
#     # 1) undo row normalization
#     median_subtracted = preprocessed_slice * row_std                     # broadcasts along columns

#     # 2) add back column median for that slice
#     col_med_slice = col_med[:, start_idx:end_idx]                        # shape (1, n_slice_cols)
#     norm_flux_slice = median_subtracted + col_med_slice                  # shapes broadcast

#     # 3) undo global normalization
#     reconstructed_slice = norm_flux_slice * global_med                   # shape (n_rows, n_slice_cols)
#     return reconstructed_slice

def preprocess(spectra):
    """
    Normalize by the median spectrum, subtract the median at each wavelength,
    and divide each spectrum by its own standard deviation.

    Parameters
    ----------
    spectra: array
        2d spectral flux grid.

    Returns
    -------
    array
        Median subtracted, standard deviation divided 2d spectral flux grid.
    """
    norm_flux = spectra / np.median(spectra)
    median_flux = np.median(norm_flux, axis=0)
    median_subtracted = norm_flux - median_flux

    row_std = np.linalg.norm(median_subtracted, axis=1, keepdims=True) \
              / np.sqrt(median_subtracted.shape[1])

    # avoid division by zero
    row_std = np.where(row_std == 0.0, 1.0, row_std)

    return median_subtracted / row_std, row_std

def compute_covariance_matrix(data):
    """Compute the covariance matrix using NumPy (faster than pandas).
    
    Parameters
    ----------
    data: array
        2d spectral array. Used after PCA analysis to the Time Domain or Wavelength Domain.
    
    Returns
    -------
    array
        Covariance matrix."""
    centered = data - np.mean(data, axis=0)
    return centered.T @ centered / (data.shape[0] - 1)


def compute_eigenvalues_and_vectors_jax(cov_matrix):
    """Compute and sort eigenvalues/eigenvectors in descending order.
    
    Parameters
    ----------
    cov_matrix: array
        Covariance matrix. Refer to pca_subtraction.compute_covariance_matrix.
        
    Returns
    -------
    array
        1d array of the eigenvalues in the order from highest to lowest (I think).
    array
        2d array of eigenvectors in the order of their corresponding eiganvalues."""
    jax_cov_matrix = jnp.array(cov_matrix, dtype=jnp.float64)
    evals, evecs = jnp.linalg.eigh(jax_cov_matrix)
    idx = jnp.argsort(evals)[::-1]

    evals_sorted = np.array(evals[idx])
    evecs_sorted = np.array(evecs[:, idx])

    return evals_sorted, evecs_sorted

@njit
def compute_eigenvalues_and_vectors_numba(cov_matrix):
    evals, evecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(evals)[::-1]   
    return evals, evecs, idx 


def explained_variance(eigenvalues):
    """Calculate explained variance ratio.
    
    Parameters
    ----------
    eigenvalues: array
        1d array of eigenvalues.
        
    Returns
    -------
    array
        Explained variance value for each eigenvalue."""
    return eigenvalues / np.sum(eigenvalues)


def remove_components(data, eigenvectors, first_comps=0, last_comps=0, verbose=False):
    """Remove specified principal components from the data.
    
    Parameters:
    data: array
        2d flux array.
    eigenvectors: array
        2d eigenvectors. Refer to ``pca_subtraction.compute_eigenvalues_and_vectors_jax``.
    first_comps: int, optional
        Index of first components (eigenvectors) to remove.
    last_comps: int, optional
        Index of last components (eigenvectors) to remove.
    verbose: boolean
        Refer to ``utility_functions.debug_print``.
        
    Returns
    -------
    array
        2d flux array after removing the ``first_comps`` and ``last_comps``."""
    total_comps = eigenvectors.shape[1]
    start_comps = first_comps
    end_comps = total_comps - last_comps

    if start_comps >= end_comps:
        debug_print(verbose, f"total # of components: {total_comps}. removing {start_comps} from the beginning and {last_comps} from the end")
        raise ValueError("Requested to remove all components — nothing left to reconstruct from.")

    proj_matrix = eigenvectors[:, start_comps:end_comps]
    projected = data @ proj_matrix
    return projected @ proj_matrix.T


def pca_subtraction(spectra, start_idx, end_idx, first_comps=0, last_comps=0, eighcalc='numba', pre=False, verbose=False):
    """
    Perform PCA subtraction in a wavelength slice from `start_idx` to `end_idx`.

    Parameters
    ----------
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
        spectra_pre, _ = preprocess(spectra)
    else:
        spectra_pre = spectra

    spectra_slice = spectra_pre[:, start_idx:end_idx]
    tdm = spectra_slice.T  # Transpose for TDM
    wdm = spectra_slice     # WDM as-is

    tdm_cov = compute_covariance_matrix(tdm) # spectra x spectra
    wdm_cov = compute_covariance_matrix(wdm) # wave x wave

    if eighcalc == 'jax':
        _, evec_tdm = compute_eigenvalues_and_vectors_jax(tdm_cov)
        _, evec_wdm = compute_eigenvalues_and_vectors_jax(wdm_cov)

    elif eighcalc == 'numba':
        _, evec_tdm, idx_tdm = compute_eigenvalues_and_vectors_numba(tdm_cov)
        evec_tdm = np.array(evec_tdm[:, idx_tdm])

        _, evec_wdm, idx_wdm  = compute_eigenvalues_and_vectors_numba(wdm_cov)
        evec_wdm = np.array(evec_wdm[:, idx_wdm])

    debug_print(verbose, "tdm, wdm evec shapes:", evec_tdm.shape, evec_wdm.shape)

    # PCA removal
    tdm_clean = remove_components(tdm, evec_tdm, first_comps, last_comps).T
    wdm_clean = remove_components(wdm, evec_wdm, first_comps, last_comps)
    # need to divide each column by std of the column

    return tdm_clean, wdm_clean