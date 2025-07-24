import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from numba import njit
from scipy.interpolate import interp1d


@njit
def doppler_shift(wave_arr, velocity):
    C = 299792458.0  # Speed of light in m/s

    """
    Doppler shift the wavelength array by a velocity (in m/s).
    
    Parameters:
        wave_arr (1D array): Wavelengths (in any units, e.g., nm or µm)
        velocity (float): Velocity in m/s

    Returns:
        shifted_waves (1D array): Doppler-shifted wavelengths
    """
    return wave_arr * (1 + velocity / C)

def ccf(all_pca, all_wave, v_shift_range, sim_wave, sim_flux):
    sort_idx = np.argsort(sim_wave)
    sorted_wave = sim_wave[sort_idx]
    sorted_flux = sim_flux[sort_idx]

    stacked_segment_xcorr = []
    
    for detector_spectra, detector_wavs in zip(all_pca, all_wave):
        detector_xcorr = []

        for single_data_spectra in detector_spectra:
            norm_xcorr_arr = []

            for v_shift in v_shift_range:
                # Doppler shift the template wavelength
                shifted_wave = doppler_shift(sorted_wave, v_shift)

                # Interpolate shifted flux onto segment wavelength grid
                interp_func = interp1d(shifted_wave, sorted_flux, bounds_error=False, fill_value=0.0)
                shifted_flux = interp_func(detector_wavs)

                shifted_flux = (shifted_flux - np.mean(shifted_flux)) #/ np.std(shifted_flux)
                single_data_spectra = (single_data_spectra - np.mean(single_data_spectra)) #/ np.std(single_data_spectra)

                # Cross-correlate (dot product)
                xcorr = np.dot(shifted_flux, single_data_spectra)
                norm_xcorr = xcorr / np.sqrt(np.sum(shifted_flux**2) * np.sum(single_data_spectra**2))
                norm_xcorr_arr.append(norm_xcorr)

            norm_xcorr_arr = np.array(norm_xcorr_arr)
            detector_xcorr.append(norm_xcorr_arr)

        detector_xcorr = np.array(detector_xcorr)
        stacked_segment_xcorr.append(detector_xcorr)

    return np.sum(np.array(stacked_segment_xcorr), axis=0)

# doppler shift correction functions

def orbital_phase(t, T_not, P_orb):
    return (t - T_not)/P_orb

def orbit_velocity(a, P_orb):
    return 2 * np.pi * a / P_orb 

def rv_amplitude(a, P_orb, i):
    v_orb = orbit_velocity(a, P_orb)
    return v_orb * np.sin(i)

def doppler_correction(a, P_orb, i, t, T_not, v_sys, v_bary, Kp=None):
    """
    a in au, P_orb in days, i in degrees
    t in MJD
    T_not in MJD (mid-transit time)
    v_sys in km/s
    v_bary in km/s
    Kp in m/s, if None, will compute from a, P_orb, i
    """
    a = a * 1.495979e11  # Convert au to meters
    P_orb = P_orb * 24 * 3600  # Convert days to seconds
    i = np.radians(i)  # Convert degrees to radians
    t = t * 24 * 3600  # Convert MJD to seconds
    T_not = T_not * 24 * 3600  # Convert MJD to seconds
    v_bary = v_bary * 1000  # Convert km/s to m/s   
    v_sys = v_sys * 1000  # Convert km/s to m/s

    if Kp is None:
        Kp = rv_amplitude(a, P_orb, i)
    phi = orbital_phase(t, T_not, P_orb)

    #print(f"Kp: {Kp} m/s", "orbital phase:", phi)

    return (Kp * np.sin(2*np.pi*phi)) + v_sys + v_bary

def compute_vbary_timeseries(ra_deg, dec_deg, times_utc, location):
    """
    Compute v_bary(t) for a target at (ra, dec) and a time array.
    
    Parameters:
        ra_deg (float): RA in degrees
        dec_deg (float): Dec in degrees
        times_utc (array-like): List or array of UTC times (ISO strings or float MJD)
        location (EarthLocation): Astropy EarthLocation (observatory)

    Returns:
        np.ndarray: Barycentric velocities (km/s) for each time
    """
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    times = Time(times_utc, format='mjd', scale='utc', location=location)

    barycorr = target.radial_velocity_correction(obstime=times)
    return barycorr.to(u.km/u.s).value

def doppler_correct_ccf(summed_ccf, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, Kp=None):
    v_bary_timeseries = compute_vbary_timeseries(ra, dec, mjd_obs, location)
    #print("v_bary_timeseries: ", v_bary_timeseries)
    all_doppler_corrects = []

    for jj in range(len(mjd_obs)):

        if Kp is None:
            correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs[jj], T_not=T_not, v_sys=v_sys, v_bary=v_bary_timeseries[jj])

        else:
            correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs[jj], T_not=T_not, v_sys=v_sys, v_bary=v_bary_timeseries[jj], Kp=Kp)

        all_doppler_corrects.append(correction)

    # check for nans in doppler correction
    # print("Doppler correction contains NaNs: ", np.any(np.isnan(all_doppler_corrects)))
    # print("Doppler corrections:", all_doppler_corrects)
    new_vel_grids = []

    for kk in range(len(summed_ccf)):
        new_vel_grid = v_shift_range + all_doppler_corrects[kk]
        #print("start, end of new_vel_grid: ", new_vel_grid[0], new_vel_grid[-1])
        new_vel_grids.append(new_vel_grid)

    min_v, max_v = -50000, 50000
    common_v_grid = np.linspace(min_v, max_v, 101)  # Common velocity grid for cropping

    #check for nans in new_vel_grids
    #print("New velocity grids contain NaNs: ", np.any([np.any(np.isnan(v)) for v in new_vel_grids]))

    cropped_ccf = []

    for i in range(len(new_vel_grids)):
        v = new_vel_grids[i]
        ccf = summed_ccf[i]
            
        common_mask = (v >= min_v) & (v <= max_v)
        interp_ccf = np.interp(common_v_grid, v[common_mask], ccf[common_mask])
        cropped_ccf.append(interp_ccf)

    #check for nans in cropped_ccf
    #print("Cropped CCF contains NaNs: ", np.any(np.isnan(cropped_ccf)))

    return np.array(cropped_ccf), common_v_grid

def remove_out_of_transit(transit_start_end, grid, mjd_obs):

    # remove spectra outside of transit start and end
    transit_start, transit_end = transit_start_end
    transit_mask = (mjd_obs >= transit_start) & (mjd_obs <= transit_end)
    filtered_grid = [grid[i] for i in range(grid.shape[0]) if transit_mask[i]]

    return filtered_grid

def run_ccf_on_detector_segments(all_wave, 
                                 all_pca, v_shift_range, segment_indices, sim_wave, 
                                 sim_flux, mjd_obs, ra, dec, location, 
                                 a, P_orb, i, T_not, v_sys, transit_start_end, remove_segments=None): #sim_wave in um for now

    if remove_segments is None:
        remove_segments = []

    # Filter segments
    keep_indices = [i for i in range(len(segment_indices)) if i not in remove_segments]

    filtered_all_pca = [all_pca[i] for i in keep_indices]
    filtered_all_wave = [all_wave[i] for i in keep_indices]

    earth_frame_ccf = ccf(filtered_all_pca, filtered_all_wave, v_shift_range, sim_wave, sim_flux)

    # check for NaNs in earth_frame_ccf
    #print("Earth frame CCF contains NaNs: ", np.any(np.isnan(earth_frame_ccf)))

    planet_frame_ccf, planet_frame_vgrid = doppler_correct_ccf(earth_frame_ccf, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys)

    in_transit = remove_out_of_transit(
    transit_start_end, planet_frame_ccf, mjd_obs)

    return earth_frame_ccf, planet_frame_ccf, planet_frame_vgrid, in_transit

def inject_simulated_signal(wave, flux, sim_wave, sim_flux, 
                            mjd_obs, ra, dec, location, 
                            a, P_orb, i, T_not, v_sys):
    """
    Inject a simulated signal into the observed flux array.
    """
    v_bary = compute_vbary_timeseries(ra, dec, mjd_obs, location)
    correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs, T_not=T_not, v_sys=v_sys, v_bary=v_bary)
    sim_on_obs_grid = interp1d(sim_wave, sim_flux, bounds_error=False, fill_value=0)(wave * 0.001)  # Convert wave to microns
    spectra_grid = np.zeros_like(flux)
    for i in range(len(mjd_obs)):
        shifted_sim = doppler_shift(sim_on_obs_grid, correction[i])
        spectra_grid[i, :] = flux[i, :] - shifted_sim
    return spectra_grid

def kp_vel_grid(cropped_ccf_array, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, transit_start_end, Kp_range=np.linspace(50_000, 150_000, 101)):
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

    return np.array(Kp_range_ccf)