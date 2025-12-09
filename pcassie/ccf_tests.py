import numpy as np

from scipy.stats import ttest_ind
from scipy.interpolate import interp1d

from pcassie.ccf import *

def inject_simulated_signal(sim_wave, sim_flux,  
                            R_p, R_star, multiple=1, verbose=False, **kwargs):
    """
    Inject a simulated signal into the observed flux array. Both flux and sim_flux must be normalized.

    Parameters
    ----------
    sim_wave: array
        Refer to pipeline.pipeline.
    sim_flux: array
        Refer to pipeline.pipeline.
    R_p: float
        Radius of the exoplanet. Units must be the same as R_star.
    R_star: float
        Radius of the host star. Units must be tha same as R_p.
    multiple: int
        Scalar multiple of the injection. E.g. if you want the injected signal to be 0.5x the scale of the 
        expected planet signal, set ``multiple=0.5``.
    verbose: boolean
        Refer to utility_functions.debug_print.
    **kwargs
        Refer to pipeline.pipeline.

    Returns
    -------
    array
        new flux array with injected signal
    """
    wave, flux, mjd_obs, ra, dec, location = kwargs['wave'][0], kwargs['flux'], kwargs['mjd_obs'], kwargs['ra'], kwargs['dec'], kwargs['location']
    a, P_orb, i, T_not, v_sys = kwargs['a'], kwargs['P_orb'], kwargs['i'], kwargs['T_not'], kwargs['v_sys']

    factor = (R_p / R_star) * multiple
    v_bary = compute_vbary_timeseries(ra, dec, mjd_obs, location)
    correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs, T_not=T_not, v_sys=v_sys, v_bary=v_bary)
    debug_print(verbose, f"correction: {correction}")
    sim_on_obs_grid = interp1d(sim_wave, sim_flux, bounds_error=False, fill_value=0)
    spectra_grid = np.zeros_like(flux)

    sim_shifts = []
    for j in range(len(mjd_obs)):
        shifted_wave = doppler_shift(wave, correction[j])  # ← shift wavelengths, not flux
        shifted_sim = sim_on_obs_grid(shifted_wave) * factor
        spectra_grid[j, :] = flux[j, :] + shifted_sim
        sim_shifts.append(shifted_sim)
    return spectra_grid

def sn_map(
    planet_frame_ccf, planet_frame_vgrid,
    Kp_range=np.linspace(50_000, 150_000, 101), **kwargs
):
    """
    SNR map computation for the CCF.

    Parameters
    ----------
    planet_frame_ccf: array
        2d CCF array in the planet rest frame. Refer to pipeline.pipeline.
    planet_frame_vgrid: array
        1d velocity grid (km/s). Refer to pipeline.pipeline.
    Kp_range: array
        Range of velocities in m/s to sample radial velocity amplitude Kp. Like ``v_shift_range`` and ``planet_frame_vgrid``, ideal to have in steps of 1 km/s.
    **kwargs
        Refer to pipeline.pipeline.

    Returns
    -------
    array
        2d grid of CCF values over the sampled radial velocity amplitude (Kp) vs. velocity shift space.
    array
        2d S/N grid made by dividing the Kp vs. velocity shift grid by its the standard deviation of the out trail values.
    """
    mjd_obs, ra, dec, location = kwargs['mjd_obs'], kwargs['ra'], kwargs['dec'], kwargs['location']
    a, P_orb, i, T_not, v_sys, transit_start_end = kwargs['a'], kwargs['P_orb'], kwargs['i'], kwargs['T_not'], kwargs['v_sys'], kwargs['transit_start_end']

    n_Kp = len(Kp_range)
    n_v = len(planet_frame_vgrid)

    # Preallocate result array instead of appending
    Kp_range_ccf = np.zeros((n_Kp, n_v), dtype=np.float32)
    
    for idx, Kp in enumerate(Kp_range):
        try:
            # Doppler correct
            this_cropped_ccf, _ = doppler_correct_ccf(
                planet_frame_ccf, planet_frame_vgrid, mjd_obs,
                ra, dec, location, a, P_orb, i, T_not, v_sys, Kp=Kp
            )

            if np.isnan(this_cropped_ccf).any():
                continue  # Skip Kp values with NaNs

            # Remove out-of-transit
            ccf_in_transit = remove_out_of_transit(
                transit_start_end=transit_start_end,
                grid=this_cropped_ccf,
                mjd_obs=mjd_obs
            )

            # Sum across time axis (axis=0)
            Kp_range_ccf[idx] = np.sum(ccf_in_transit, axis=0)

        except Exception as e:
            print(f"Skipping Kp = {Kp:.1f} due to error: {e}")
            continue

    # Mask near-planet velocities (±15 km/s = 15000 m/s)
    exclude_planet_mask = (np.abs(planet_frame_vgrid) < 15000)

    # Standard deviation of CCF away from planet signal
    outside_std = np.std(Kp_range_ccf[:, exclude_planet_mask], axis=1)

    # Avoid division by zero
    outside_std[outside_std == 0] = np.nan

    # Compute S/N map
    sn_map_array = Kp_range_ccf / outside_std[:, np.newaxis]

    return Kp_range_ccf, sn_map_array


def welch_t_test(Kp_range_ccf, zoom_radius=15):
    """Performs Welch's T-test to compare values in the CCF grid
    that may be associated with the planet with those that are not.
    
    Parameters
    ----------
    Kp_range_ccf: array
        2d Kp grid from ccf_tests.sn_map. Refer to ccf_tests.sn_map.
    zoom_radius: int, optional
        Pixel space radius to build the in trail box around. Here, 1 pixel = 1 km/s (if you have your units right).

    Returns
    -------
    array
        2d Kp vs. velocity shift CCF grid for in trail values.
    array
        2d Kp vs. velocity shift CCF grid for out trail values.
    int
        Welch's T-test statistic comparing in trail and out trail CCF values.
    int
        Welch's p value comparing in trail and out trail CCF values.
    """
    # Define zoom radius in pixels (typically km/s)
    Kp_range_ccf = np.array(Kp_range_ccf)  # Ensure it's a NumPy array

    # Step 1: Find max index in S/N map
    max_val = np.nanmax(Kp_range_ccf)
    max_idx = np.argwhere(Kp_range_ccf == max_val)[0]
    max_row, max_col = max_idx

    # Step 2: Clip the box to array boundaries
    n_rows, n_cols = Kp_range_ccf.shape
    min_row = max(0, max_row - zoom_radius)
    max_row_clip = min(n_rows, max_row + zoom_radius + 1)  # +1 because slicing is exclusive
    min_col = max(0, max_col - zoom_radius)
    max_col_clip = min(n_cols, max_col + zoom_radius + 1)

    # Step 3: Extract in-trail values
    in_trail_vals = Kp_range_ccf[min_row:max_row_clip, min_col:max_col_clip].ravel()

    # Step 4: Create out-of-trail values by masking in-trail region
    masked_array = Kp_range_ccf.copy()
    masked_array[min_row:max_row_clip, min_col:max_col_clip] = np.nan
    out_of_trail_vals = masked_array[~np.isnan(masked_array)]

    # Step 5: Welch’s t-test
    t_stat, p_value = ttest_ind(in_trail_vals, out_of_trail_vals, equal_var=False)

    return in_trail_vals, out_of_trail_vals, t_stat, p_value


def find_max_sn_in_expected_range(sn_array, v_grid, offset=75, zoom_radius=15, **kwargs):
    """Finds the maximum S/N value in the range of values expected of the planet.
    
    Parameters
    ----------
    
    sn_array: array
        2d S/N map from ccf_tests.sn_map.
    v_grid: array
        1d velocity range corresponding to ``sn_array``.
    offset: int, optional
        Valocity offset of the Kp range. E.g. if you were to sample a CCF from 75 km/s < Kp < 175 km/s, set ``offset=75``.
    zoom_radius: int
        Refer to ccf_test.welch_t_test.
    **kwargs
        Refer to pipeline.pipeline.

    Returns
    -------
    int
        Maximum S/N value within the ``zoom_radius`` of the expected Kp and velocity shift values (expected v shift value in 
        planet frame is 0).
    """

    a, P_orb, i = kwargs['a'], kwargs['P_orb'], kwargs['i']
    Kp = rv_amplitude(a * 1.495979e11, P_orb * 24 * 3600, np.radians(i)) / 1000
    #print(Kp)

    row_idx = int(Kp) - offset
    col_idx = np.argwhere(v_grid == 0)[0][0]

    #print(row_idx, col_idx)

    n_rows, n_cols = sn_array.shape
    min_row = max(0, row_idx - zoom_radius)
    max_row_clip = min(n_rows, row_idx + zoom_radius + 1)  # +1 because slicing is exclusive
    min_col = max(0, col_idx - zoom_radius)
    max_col_clip = min(n_cols, col_idx + zoom_radius + 1)
    #print(min_row, max_row_clip, min_col, max_col_clip)

    expected_range = sn_array[min_row:max_row_clip, min_col:max_col_clip]

    return np.max(expected_range)  