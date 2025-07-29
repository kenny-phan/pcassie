import numpy as np

from scipy.stats import ttest_ind
from scipy.interpolate import interp1d

from ccf import *

def inject_simulated_signal(wave, flux, sim_wave, sim_flux, 
                            mjd_obs, ra, dec, location, 
                            a, P_orb, i, T_not, v_sys, R_p, R_star, multiple=1, verbose=False):
    """
    Inject a simulated signal into the observed flux array. Both flux and sim_flux must be normalized.
    """
    factor = ((R_p / R_star)**2) * multiple
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
    cropped_ccf_array, v_shift_range, mjd_obs, ra, dec, location,
    a, P_orb, i, T_not, v_sys, transit_start_end,
    Kp_range=np.linspace(50_000, 150_000, 101)
):
    """
    Optimized SNR map computation for the CCF.
    """
    n_Kp = len(Kp_range)
    n_v = len(v_shift_range)

    # Preallocate result array instead of appending
    Kp_range_ccf = np.zeros((n_Kp, n_v), dtype=np.float32)
    
    for idx, Kp in enumerate(Kp_range):
        try:
            # Doppler correct
            this_cropped_ccf, _ = doppler_correct_ccf(
                cropped_ccf_array, v_shift_range, mjd_obs,
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
    exclude_planet_mask = (np.abs(v_shift_range) < 15000)

    # Standard deviation of CCF away from planet signal
    outside_std = np.std(Kp_range_ccf[:, exclude_planet_mask], axis=1)

    # Avoid division by zero
    outside_std[outside_std == 0] = np.nan

    # Compute S/N map
    sn_map_array = Kp_range_ccf / outside_std[:, np.newaxis]

    return Kp_range_ccf, sn_map_array


def welch_t_test(Kp_range_ccf, zoom_radius=15):
    """Performs Welch's T-test to compare values in the CCF grid
    that may be associated with the planet with those that are not."""
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


def find_max_sn_in_expected_range(sn_array, v_grid, a, P_orb, i, offset=75, zoom_radius=15):
    """Finds the maximum S/N value in the range of values expected of the planet."""
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