import numpy as np
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
    sn_map = Kp_range_ccf / outside_std[:, np.newaxis]

    return sn_map