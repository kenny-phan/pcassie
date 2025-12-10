import pytest
import numpy as np

from astropy.coordinates import EarthLocation


def test_pipeline_basic():

    from pcassie.pipeline import pipeline
    from pcassie.ccf import doppler_shift

    np.random.seed(0)

    # ----- Synthetic parameters -----
    n_spectra = 5
    n_wavelengths = 500
    n_detectors = 1  # keep simple for test

    # Wavelength grid
    base_wave = np.linspace(500, 600, n_wavelengths)  # in nm

    # Simulated planetary signal: a simple sinusoid
    sim_flux = np.sin(2 * np.pi * base_wave / 10)
    sim_wave = base_wave.copy()

    shift = 10000.0

    # Observed spectra: signal + small noise
    flux = np.zeros((n_spectra, n_wavelengths))
    for i in range(n_spectra):
        # inject signal into the first spectrum
        if i == 0:
            shifted_wave = doppler_shift(sim_wave, shift)  # shift 1 km/s
            flux[i] = np.interp(base_wave, shifted_wave, sim_flux)
        else:
            flux[i] = np.random.randn(n_wavelengths) * 0.01  # noise

    # All spectra share the same wavelength grid
    wave = np.tile(base_wave, (n_spectra, 1))

    # Observation info
    mjd_obs = np.arange(n_spectra)
    ra = np.ones(n_spectra) * 100.0
    dec = np.ones(n_spectra) * 20.0
    location = EarthLocation.of_site("greenwich")  # simple placeholder

    # Planet parameters
    a = 0.05       # AU
    P_orb = 3.0    # days
    i = 90.0       # deg
    T_not = 0.0    # MJD
    v_sys = 0.0    # km/s
    transit_start_end = (0, 10)

    # PCA parameters
    gap_size = 5
    remove_segments = []
    first_components = 0
    last_components = 0

    kwargs = dict(
        wave=wave,
        flux=flux,
        mjd_obs=mjd_obs,
        ra=ra,
        dec=dec,
        location=location,
        a=a,
        P_orb=P_orb,
        i=i,
        T_not=T_not,
        v_sys=v_sys,
        transit_start_end=transit_start_end,
        gap_size=gap_size,
        remove_segments=remove_segments,
        first_components=first_components,
        last_components=last_components,
    )

    v_shift_range = np.linspace(-500000.0, 500000.0, 201)  # m/s, coarse

    # ----- Run pipeline -----
    outputs = pipeline(sim_wave, sim_flux, v_shift_range=v_shift_range, verbose=False, **kwargs)

    # ----- Check outputs -----
    all_tdm, all_wdm, all_wave, earth_ccf, planet_ccf, planet_vgrid, in_transit, Kp_range_ccf, sn_map_array, in_trail, out_trail, t_stat, p_value = outputs

    # TDM, WDM, wave lists
    assert isinstance(all_tdm, list)
    assert isinstance(all_wdm, list)
    assert isinstance(all_wave, list)
    assert all(isinstance(x, np.ndarray) for x in all_tdm)
    assert all(isinstance(x, np.ndarray) for x in all_wdm)
    assert all(isinstance(x, np.ndarray) for x in all_wave)

    # CCF arrays
    assert earth_ccf.shape == (n_spectra, len(v_shift_range))
    assert planet_ccf.shape[0] == n_spectra
    assert planet_vgrid.ndim == 1

    # S/N map
    assert sn_map_array.ndim == 2

    # In/out-trail stats
    assert in_trail.size > 0                    # in-trail region exists
    assert out_trail.size > in_trail.size       # out-trail much larger
    assert np.nanmax(Kp_range_ccf) in in_trail 
    assert np.isscalar(t_stat)
    assert np.isscalar(p_value)

    # Check that CCF for first spectrum peaks near injected signal
    peak_idx = np.argmax(earth_ccf[0])
    expected_idx = np.where(v_shift_range == shift)[0][0] # should check sign of this
    assert peak_idx == expected_idx
    assert abs(peak_idx - expected_idx) <= 1  # allow 1-bin tolerance


# ---------------------------------------------------------------------
# Mock helper functions
# ---------------------------------------------------------------------
def mock_pipeline(sim_wave, sim_flux, v_shift_range=None, verbose=False, **kwargs):
    """
    Return a minimal pipeline result with the correct structure:
    - results[8] -> sn_map_array
    - results[5] -> planet_vgrid
    """
    # Fake planet velocity grid
    planet_vgrid = np.linspace(-100, 100, 50)

    # Fake S/N map: magnitude increases as number of removed components increases
    fc = kwargs.get("first_components", 0)
    lc = kwargs.get("last_components", 0)
    sn_strength = fc + lc + 1

    sn_map = np.ones((10, planet_vgrid.size)) * sn_strength

    # Compose 13-element results list
    return [
        None, None, None, None, None,
        planet_vgrid,        # results[5]
        None, None,
        sn_map,              # results[8]
        None, None, 0, 1
    ]


def mock_find_max_sn(sn_map, vgrid, **kwargs):
    """Return simple S/N score for test."""
    return np.nanmax(sn_map)


# Patch the real functions in your module
import pcassie.pipeline as p
p.pipeline = mock_pipeline
p.find_max_sn_in_expected_range = mock_find_max_sn

def test_sample_components():
    from pcassie.pipeline import sample_components 
    from pcassie.ccf import doppler_shift
    np.random.seed(0)

    v_shift_range = np.linspace(-500000.0, 500000.0, 201)  # m/s, coarse

    # ----- Synthetic parameters -----
    n_spectra = 5
    n_wavelengths = 500
    n_detectors = 1  # keep simple for test

    # Wavelength grid
    base_wave = np.linspace(500, 600, n_wavelengths)  # in nm

    # Simulated planetary signal: a simple sinusoid
    sim_flux = np.sin(2 * np.pi * base_wave / 10)
    sim_wave = base_wave.copy()

    shift = 10000.0

    # Observed spectra: signal + small noise
    flux = np.zeros((n_spectra, n_wavelengths))
    for i in range(n_spectra):
        # inject signal into the first spectrum
        if i == 0:
            shifted_wave = doppler_shift(sim_wave, shift)  # shift 1 km/s
            flux[i] = np.interp(base_wave, shifted_wave, sim_flux)
        else:
            flux[i] = np.random.randn(n_wavelengths) * 0.01  # noise

    # All spectra share the same wavelength grid
    wave = np.tile(base_wave, (n_spectra, 1))

    # Observation info
    mjd_obs = np.arange(n_spectra)
    ra = np.ones(n_spectra) * 100.0
    dec = np.ones(n_spectra) * 20.0
    location = EarthLocation.of_site("greenwich")  # simple placeholder

    # Planet parameters
    a = 0.05       # AU
    P_orb = 3.0    # days
    i = 90.0       # deg
    T_not = 0.0    # MJD
    v_sys = 0.0    # km/s
    transit_start_end = (0, 10)

    # PCA parameters
    gap_size = 5
    remove_segments = []
    first_components = 0
    last_components = 0

    kwargs = dict(
        wave=wave,
        flux=flux,
        mjd_obs=mjd_obs,
        ra=ra,
        dec=dec,
        location=location,
        a=a,
        P_orb=P_orb,
        i=i,
        T_not=T_not,
        v_sys=v_sys,
        transit_start_end=transit_start_end,
        gap_size=gap_size,
        remove_segments=remove_segments,
        first_components=first_components,
        last_components=last_components,
    )

    start_components = 2
    stable_components = 1

    best_results, best_sn, best_comp = sample_components(
        start_components=start_components,
        stable_components=stable_components,
        sim_wave=sim_wave,
        sim_flux=sim_flux,
        v_shift_range=v_shift_range,
        sn_test=-40,
        sn_max=-100,
        sample_end=True,
        **kwargs
    )

    # sn increases with last_components, loop stops after first step
    assert best_sn >= -40
    assert best_comp <= 2
    # assert isinstance(best_results, list)

    best_results, best_sn, best_comp = sample_components(
            start_components=start_components,
            stable_components=stable_components,
            sim_wave=sim_wave,
            sim_flux=sim_flux,
            v_shift_range=v_shift_range,
            sn_test=-40,
            sn_max=-100,
            sample_end=False,
            **kwargs
        )
    
    assert best_sn >= -40
    assert best_comp <= 2