import pytest
import numpy as np
from pcassie.ccf import *

@pytest.mark.parametrize("velocity", [-100, 0, 100])
def test_ccf(velocity):
    # ----- Parameters -----
    n_detectors = 2
    n_spectra = 3
    n_wavelengths = 1000
    v_shift_range = np.linspace(-1000, 1000, 11)  # in m/s, coarse for test
    
    # ----- Wavelength grid -----
    base_wave = np.linspace(500, 600, n_wavelengths)  # nm

    # ----- Simulated signal -----
    sim_flux = np.sin(2 * np.pi * base_wave / 10)  # simple sinusoid
    sim_wave = base_wave.copy()

    # ----- PCA subtracted data -----
    all_pca = []
    all_wave = []

    # Create data with one spectrum shifted by +100 m/s
    shift_idx = 5  # middle of v_shift_range
    for det in range(n_detectors):
        spectra = []
        for spec in range(n_spectra):
            # For simplicity, one spectrum is the same as sim_flux, others are noise
            if spec == 0:
                # Shift sim_flux by +100 m/s
                shifted_wave = doppler_shift(sim_wave, velocity)  # Doppler shift approximation
                spectra.append(np.interp(base_wave, shifted_wave, sim_flux))
            else:
                spectra.append(np.random.randn(n_wavelengths) * 0.01)  # noise
        all_pca.append(np.array(spectra))
        all_wave.append(base_wave.copy())

    # ----- Run cross-correlation -----
    xcorr_result = ccf(all_pca, all_wave, v_shift_range, sim_wave, sim_flux, speed=True)

    # ----- Assertions -----
    assert xcorr_result.shape == (n_spectra, len(v_shift_range))  # correct shape
    # Peak should be at the known shift for the first spectrum
    peak_idx = np.argmax(xcorr_result[0])
    assert peak_idx == shift_idx
    # Other spectra (noise) should have small correlation
    assert np.all(np.abs(xcorr_result[1:]) < 0.2)


# --- Mock functions to isolate the test ---
def mock_compute_vbary_timeseries(ra, dec, mjd_obs, location):
    # Return a simple linear barycentric velocity series
    return np.linspace(0, 1000, len(mjd_obs))

def mock_doppler_correction(a, P_orb, i, t, T_not, v_sys, v_bary, Kp=None):
    # Return a deterministic "orbital + barycentric" velocity
    base = v_sys + v_bary
    if Kp is not None:
        # simple sinusoidal orbital motion
        base += Kp * np.sin(2 * np.pi * (t - T_not) / P_orb)
    return base

# Patch the imported functions in the module
import pcassie.ccf as ccf_module
ccf_module.compute_vbary_timeseries = mock_compute_vbary_timeseries
ccf_module.doppler_correction = mock_doppler_correction

# ---------------- TEST -----------------
def test_doppler_correct_ccf_basic():
    # ----- Inputs -----
    n_spectra = 3
    n_vel = 5
    summed_ccf = [np.ones(n_vel) for _ in range(n_spectra)]  # simple CC array
    v_shift_range = np.linspace(-10, 10, n_vel)
    mjd_obs = np.array([0, 1, 2])
    ra, dec = 0.0, 0.0
    location = (0.0, 0.0)
    a, P_orb, i, T_not = 1.0, 2.0, 90.0, 0.0
    v_sys = 5.0
    Kp = 1.0

    # ----- Run function -----
    cropped_ccf, common_v_grid = doppler_correct_ccf(
        summed_ccf,
        v_shift_range,
        mjd_obs,
        ra,
        dec,
        location,
        a, P_orb, i, T_not,
        v_sys,
        Kp=Kp,
        verbose=False
    )

    # ----- Assertions -----
    # 1. Shape of the output
    assert cropped_ccf.shape == (n_spectra, len(common_v_grid))
    
    # 2. Common velocity grid is as expected
    assert common_v_grid[0] == -50000
    assert common_v_grid[-1] == 50000
    
    # 3. Values are finite
    assert np.all(np.isfinite(cropped_ccf))
    
    # 4. Cropped CCFs are roughly positive (since input was all ones and doppler shifts just re-grid)
    assert np.all(cropped_ccf > 0)
