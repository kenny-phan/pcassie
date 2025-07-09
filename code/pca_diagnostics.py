import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pca_subtraction import *

def plot_spectral_square(spectra_array, wave, title=None):
    plt.figure(figsize=(10, 5))
    plt.imshow(spectra_array, aspect='auto', 
            extent=[wave[0], wave[-1], 0, spectra_array.shape[0]],
            origin='lower', cmap='viridis', vmin=np.percentile(spectra_array, 1), vmax=np.percentile(spectra_array, 99))
    plt.colorbar(label='Flux')
    plt.xlabel('Wavelength')
    plt.ylabel('Spectrum Index')
    if title:
        plt.title(title)
    else:
        plt.title('Spectral Square Plot')
    plt.show()

def plot_preprocess(flux, wave):
    plot_spectral_square(flux, wave, title="Base Spectra")

    # Normalize by the median of this spectrum
    norm_flux = flux / np.median(flux)

    plot_spectral_square(flux, wave, title="Normalized Spectra")

    # Compute the median at each wavelength (column)
    median_flux = np.median(flux, axis=0)

    # Subtract the median from each spectrum
    median_subtracted_flux = norm_flux - median_flux  # shape: (num_spectra, num_wavelengths)

    plot_spectral_square(median_subtracted_flux, wave, title="Median Subtracted Spectra")

    # Compute the standard deviation for each spectrum (row)
    row_std = np.std(median_subtracted_flux, axis=1, keepdims=True)  # shape: (num_spectra, 1)

    # Divide each row by its own standard deviation
    row_std_divided_flux = median_subtracted_flux / row_std  # shape: (num_spectra, num_wavelengths)

    plot_spectral_square(row_std_divided_flux, wave, title="Standard Deviation Divided Spectra")

    return row_std_divided_flux

def plot_covariance(tdm_covariance, wdm_covariance):
    plt.imshow(tdm_covariance, cmap='viridis', aspect='auto')
    plt.colorbar(label='Covariance')
    plt.title("TDM Covariance Matrix")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Wavelength Index")
    plt.show()

    plt.imshow(wdm_covariance, cmap='viridis', aspect='auto')
    plt.colorbar(label='Covariance')    
    plt.title("WDM Covariance Matrix")
    plt.xlabel("Spectrum Index")
    plt.ylabel("Spectrum Index")
    plt.show()

def plot_eigenvectors(eigenvectors, title=None):
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for i in range(5):
        axes[i].plot(eigenvectors[:, i], label=f'Eigenvector {i+1}')
        axes[i].set_ylabel('Value')
        axes[i].legend(loc='upper right')
        if i == 0 and title:
            axes[i].set_title(title)
    axes[-1].set_xlabel('Index')
    plt.tight_layout()
    plt.show()

def plot_explained_variance(eigenvalues, title=None):
    """Plot the explained variance from eigenvalues."""
    explained_var = explained_variance(eigenvalues)
    plt.figure(figsize=(10, 6))
    plt.plot(explained_var, marker='o', linestyle='-', color='b')
    plt.title('Explained Variance by Eigenvalues' if title is None else title)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Explained Variance')
    plt.yscale("log", base=10)
    plt.grid()
    plt.show()

def plot_reconstructed_spectra(original, reconstructed, wave, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(wave, original[0], label='Original Spectrum', alpha=0.5)
    plt.plot(wave, reconstructed[0], label='Reconstructed Spectrum', linestyle='--')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    if title:
        plt.title(title)
    else:
        plt.title('Original vs Reconstructed Spectrum')
    plt.legend()
    plt.show()

def plot_pca_subtraction(spectra, wave, start_wav, end_wav, component_count, preprocess=False):
    """Runs PCA subtraction and plots the results."""
    if preprocess:
        print("Preprocessing spectra...")
        spectra = plot_preprocess(spectra, wave)
    else:
        print("Skipping preprocessing...")

    start_idx, end_idx = convert_range_to_indices(wave, start_wav, end_wav)

    tdm_df = pd.DataFrame(spectra[:, start_idx:end_idx].T)
    wdm_df = pd.DataFrame(spectra[:, start_idx:end_idx])

    tdm_covariance = tdm_df.cov().values
    wdm_covariance = wdm_df.cov().values

    eval_tdm, evec_tdm  = compute_eigenvalues_and_vectors(tdm_covariance)
    eval_wdm, evec_wdm = compute_eigenvalues_and_vectors(wdm_covariance)

    # Remove components from TDM and WDM
    tdm_reconstructed = remove_components(spectra[:, start_idx:end_idx].T, evec_tdm, component_count)
    wdm_reconstructed = remove_components(spectra[:, start_idx:end_idx], evec_wdm, component_count)

    plot_covariance(tdm_covariance, wdm_covariance)

    plot_eigenvectors(evec_tdm, title="TDM Eigenvectors")
    plot_eigenvectors(evec_wdm, title="WDM Eigenvectors")

    plot_explained_variance(eval_tdm, title="TDM Explained Variance")
    plot_explained_variance(eval_wdm, title="WDM Explained Variance")

    plot_reconstructed_spectra(spectra[:, start_idx:end_idx], tdm_reconstructed.T, wave[start_idx:end_idx], title="TDM Reconstructed Spectrum")
    plot_reconstructed_spectra(spectra[:, start_idx:end_idx], wdm_reconstructed, wave[start_idx:end_idx], title="WDM Reconstructed Spectrum")

