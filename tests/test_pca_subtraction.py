import numpy as np

from pcassie.pca_subtraction import *

def test_convert_range_to_indices():
    wave = np.array([1, 2, 3, 4, 5])
    start, end = 2.5, 4.5
    start_idx, end_idx = convert_range_to_indices(wave, start, end)
    assert start_idx == 2
    assert end_idx == 4

# def test_preprocess():
#     # Use a NON-degenerate spectrum where no row becomes all zeros
#     spectra = np.array([
#         [1.0, 2.0, 4.0],
#         [2.0, 3.0, 5.0],
#         [3.0, 5.0, 8.0],
#     ])

#     processed = preprocess(spectra)

#     median = np.median(processed)

#     assert np.allclose(median, 0, atol=1e-6)
#     # assert np.allclose(row_std, 1, atol=1e-6)

def test_compute_covariance_matrix():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    cov_matrix = compute_covariance_matrix(data)
    assert cov_matrix.shape == (2, 2)
    assert np.allclose(cov_matrix, np.cov(data, rowvar=False))

def test_compute_eigenvalues_and_vectors_numba():
    cov = np.array([[2.0, 1.0], [1.0, 2.0]])

    evals, evecs, idx = compute_eigenvalues_and_vectors_numba(cov)
    
    # test correct evals
    expected_evals = np.array([1.0, 3.0]) #analytic solutions
    assert np.allclose(np.sort(evals), expected_evals)

    # check sorting is correct
    sorted_expected = expected_evals[::-1]  # check for descending order
    assert np.allclose(evals[idx], sorted_expected)

    # 3. Test orthonormality of eigenvectors: V^T V = I
    V = evecs
    I = V.T @ V
    assert np.allclose(I, np.eye(2))

def test_pca_removes_dominant_component():
    rng = np.random.default_rng(42)

    n_spectra = 10       # number of rows (observations)
    n_wavelengths = 50   # number of columns (features)

    # Create one dominant spatial pattern u (principal eigenvector)
    x = np.linspace(0, 2*np.pi, n_wavelengths)
    u = np.sin(x)           

    # Create different coefficients per spectrum
    coeffs = rng.normal(loc=0.0, scale=5.0, size=n_spectra)

    # Construct rank-1 signal
    signal = np.outer(coeffs, u)   # shape (n_spectra, n_wavelengths)

    # Add small gaussian noise so the PCA still finds the rank-1 signal
    noise_level = 1e-3
    noise = rng.normal(scale=noise_level, size=signal.shape)

    spectra = signal + noise

    # Perform PCA subtraction on the full wavelength range; remove first component
    start_idx = 0
    end_idx = n_wavelengths
    first_comps = 1
    last_comps = 0

    tdm_clean, wdm_clean = pca_subtraction(
        spectra,
        start_idx, end_idx,
        first_comps=first_comps,
        last_comps=last_comps,
        eighcalc='numba',   # or 'jax' if you prefer
        pre=False,
        verbose=False
    )

    # tdm_clean has same shape as input (n_spectra, n_wavelengths)
    assert tdm_clean.shape == spectra.shape
    assert wdm_clean.shape == spectra.shape
    # Check relative norm: cleaned data should be mostly noise now
    orig_norm = np.linalg.norm(spectra)
    cleaned_norm_tdm = np.linalg.norm(tdm_clean)
    cleaned_norm_wdm = np.linalg.norm(wdm_clean)

    # The cleaned_norm should be much smaller than orig_norm (signal removed).
    # Because we only used tiny noise, require e.g. cleaned_norm < 0.1 * orig_norm
    assert cleaned_norm_tdm < 0.1 * orig_norm
    assert cleaned_norm_wdm < 0.1 * orig_norm
    
def test_pca_noise_removal():
    np.random.seed(0)

    n_samples = 50
    n_features = 1000

    # ----- Construct synthetic data -----

    # Rank-1 signal (smooth, dominant structure)

    signal = (np.linspace(0, 1, n_samples))[:,None] * np.ones((1, n_features))
    # correlated_noise = 0.05 * (np.random.rand(n_samples, 1) @ np.random.rand(1, n_features))
    white_noise = 0.01 * np.random.normal(n_samples, n_features)

    # noise = correlated_noise + white_noise
    data = signal + white_noise

    # ----- Run PCA subtraction -----
    # Remove the LAST component (noise-dominated, in our construction)
    tdm_clean, wdm_clean = pca_subtraction(
        data,
        0,
        n_features,
        first_comps = 0,
        last_comps = n_samples - 1,
        pre=False,
        eighcalc="numba"
    )

    # ----- Compute norms for TDM -----
    # orig_tdm_noise_norm = np.linalg.norm((signal + noise) - signal)
    # cleaned_tdm_noise_norm = np.linalg.norm(tdm_clean - signal)

    # ----- Compute norms for WDM -----
    mean_squared_err_orig = np.mean(white_noise**2)
    mean_squared_err_wdm = np.mean((wdm_clean - signal)**2)
    
    # ----- Assertions -----
    assert mean_squared_err_wdm <= mean_squared_err_orig #check if removed any noise
 



