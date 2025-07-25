import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pipeline.ccf import doppler_correct_ccf, remove_out_of_transit

def sn_map(cropped_ccf_array, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, transit_start_end, Kp_range=np.linspace(50_000, 150_000, 101)):
    """
    Compute SNR map for the CCF.
    """
    Kp_range_ccf = []   
    
    try: 
        for Kp in Kp_range:                         
            this_cropped_ccf, _ = doppler_correct_ccf(cropped_ccf_array, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, Kp=Kp)
            if np.any(np.isnan(this_cropped_ccf)):
                print(f"Kp = {Kp}, NaNs found in corrected CCF")

            Kp_range_ccf_in_transit = remove_out_of_transit(
                transit_start_end=transit_start_end,
                grid=this_cropped_ccf,
                mjd_obs=mjd_obs)
            
            Kp_range_ccf.append(np.sum(np.array(Kp_range_ccf_in_transit), axis=0))

    except Exception as e:
        print(f"Error processing Kp = {Kp}: {e} Try decreasing the Kp range.")
        return None 
    
    Kp_range_ccf = np.array(Kp_range_ccf)

    exclude_planet_mask = (np.abs(v_shift_range) < 15000)  # Exclude velocities within 15 km/s of the planet's velocity
    outside_std = np.std(Kp_range_ccf[:, exclude_planet_mask], axis=1)
    sn_map_array = Kp_range_ccf / outside_std[:, np.newaxis]

    return Kp_range_ccf, sn_map_array

from scipy.stats import ttest_ind

def welch_t_test(Kp_range_ccf, zoom_radius=15):
    Kp_range_ccf = Kp_range_ccf

    # Define zoom radius in pixels (not km/s here)
    zoom_radius = 15
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

def plot_welch_t_test(in_trail_vals, out_of_trail_vals, t_stat, p_value, bins=None): 
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

    