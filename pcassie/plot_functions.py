import numpy as np
import matplotlib.pyplot as plt

from pcassie.pca_subtraction import *

plt.rcParams.update({'axes.linewidth' : 1.5,
                     'ytick.major.width' : 1.5,
                     'ytick.minor.width' : 1.5,
                     'xtick.major.width' : 1.5,
                     'xtick.minor.width' : 1.5,
                     'xtick.labelsize': 12, 
                     'ytick.labelsize': 12,
                     'axes.labelsize': 18,
                     'axes.labelpad' : 5,
                     'axes.titlesize' : 22,
                     'axes.titlepad' : 10,
                     'font.family': 'Serif'
                    })

def plot_spectral_square(spectra_array, wave, title=None, x_label=None, y_label=None, cbar_label=None):
    """Plots a 2d "spectra_array" grid against a 1d "wave" array. Valuable to show a grid od spectral observations, 
    2d CCF arrays, etc."""
    n_spec = spectra_array.shape[0]

    # Create 2D grid for wave and index
    wave_edges = np.concatenate([
        [wave[0] - (wave[1] - wave[0]) / 2],
        (wave[1:] + wave[:-1]) / 2,
        [wave[-1] + (wave[-1] - wave[-2]) / 2]
    ])
    idx = np.arange(n_spec + 1)

    plt.figure(figsize=(10, 5))
    mesh = plt.pcolormesh(wave_edges, idx, spectra_array, shading='auto',
                          cmap='viridis',
                          vmin=np.percentile(spectra_array, 1),
                          vmax=np.percentile(spectra_array, 99))
    plt.colorbar(mesh, label=cbar_label or 'Flux')
    plt.xlabel(x_label or 'Wavelength')
    plt.ylabel(y_label or 'Spectrum Index')
    plt.title(title or 'Spectral Square Plot')
    plt.tight_layout()
    plt.show()

def plot_preprocess(flux, wave):
    """Plots the spectral grid at various stages of normalizing, 
    median subtracting, and standard deviation dividing."""
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
    """Plots the covariance grid of the PCA analysis in the time and wavelegnth domains."""
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
    """Plots the first five eigenvectors."""
    _, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
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
    """Plots the original and post-PCA spectra."""
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

def plot_pca_subtraction(spectra, wave, start_wav, end_wav, first_comps=0, last_comps=0, preprocess=False):
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
    tdm_reconstructed = remove_components(spectra[:, start_idx:end_idx].T, evec_tdm, first_comps, last_comps)
    wdm_reconstructed = remove_components(spectra[:, start_idx:end_idx], evec_wdm, first_comps, last_comps)

    plot_covariance(tdm_covariance, wdm_covariance)

    plot_eigenvectors(evec_tdm, title="TDM Eigenvectors")
    plot_eigenvectors(evec_wdm, title="WDM Eigenvectors")

    plot_explained_variance(eval_tdm, title="TDM Explained Variance")
    plot_explained_variance(eval_wdm, title="WDM Explained Variance")

    plot_reconstructed_spectra(spectra[:, start_idx:end_idx], tdm_reconstructed.T, wave[start_idx:end_idx], title="TDM Reconstructed Spectrum")
    plot_reconstructed_spectra(spectra[:, start_idx:end_idx], wdm_reconstructed, wave[start_idx:end_idx], title="WDM Reconstructed Spectrum")


### CCF Plot Functions 

def plot_intransit_ccfs(planet_frame_vgrid, in_transit, mean_subtracted=False):
    """Plots the velocity vs. CCF value for all spectra taken at the time of transit, 
    as well as their co-added sum. mean_subtracted effectively toggles normalization."""
    plt.figure(figsize=(10, 6))

    if mean_subtracted:
        sum = np.zeros_like(planet_frame_vgrid)
        for i, ccf in enumerate(in_transit):
            ccf -= np.mean(ccf)  # Normalize each CCF by subtracting the mean
            sum += ccf
            plt.plot(np.array(planet_frame_vgrid) / 1000, ccf, label=f"Spectrum {i+1}")
        plt.plot(planet_frame_vgrid / 1000, sum, label="Mean Subtracted Sum", color='black', linewidth=2)
        plt.title("Mean-Subtracted In-transit CCFs")

    else: 
        sum = np.zeros_like(planet_frame_vgrid)
        for i, ccf in enumerate(in_transit):
            sum += ccf
            plt.plot(np.array(planet_frame_vgrid) / 1000, ccf, label=f"Spectrum {i+1}")
        plt.plot(planet_frame_vgrid / 1000, sum, label="Sum", color='black', linewidth=2)
        plt.title("In-transit CCFs")

    plt.xlabel(r"Velocity $[kms^{-1}]$")
    plt.ylabel("CCF co-added value")
    plt.legend(ncol=3, loc='lower right', fontsize='small')
    plt.grid()
    plt.show()

    return sum

## CCF TEST PLOT FUNCTIONS

def plot_welch_t_test(in_trail_vals, out_of_trail_vals, t_stat, p_value, bins=None): 
    """Plots a histogram of Welch's T-test values for the range of values associated 
    with the planet (in trail) vs. those outside (out trail)."""
    plt.figure(figsize=(10, 6))
    bins = bins or [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] 

    plt.hist(out_of_trail_vals, bins=bins, label='Out-of-trail', color='white', histtype='step', edgecolor='blue', density=True)
    plt.hist(in_trail_vals, bins=bins, label='In-trail', color='white', histtype='step', edgecolor='orange', density=True)

    plt.axvline(np.mean(in_trail_vals), color='orange', linestyle='--', label='In-trail mean')
    plt.axvline(np.mean(out_of_trail_vals), color='blue', linestyle='--', label='Out-of-trail mean')

    plt.title(f"Welch’s t-test\nT = {t_stat:.2f}, p = {p_value:.2e}")
    plt.xlabel("CCF Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
