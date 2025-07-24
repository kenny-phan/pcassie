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

                shifted_flux = (shifted_flux - np.mean(shifted_flux)) / np.std(shifted_flux)
                single_data_spectra = (single_data_spectra - np.mean(single_data_spectra)) / np.std(single_data_spectra)


                # Cross-correlate (dot product)
                xcorr = np.correlate(shifted_flux, single_data_spectra)
                norm_xcorr = xcorr / np.sqrt(np.sum(shifted_flux**2) * np.sum(single_data_spectra**2))
                norm_xcorr_arr.append(norm_xcorr[0])

            norm_xcorr_arr = np.array(norm_xcorr_arr)
            detector_xcorr.append(norm_xcorr_arr)

        detector_xcorr = np.array(detector_xcorr)
        stacked_segment_xcorr.append(detector_xcorr)

    return np.array(stacked_segment_xcorr)

# doppler shift correction functions

def orbital_phase(t, T_not, P_orb):
    return (t - T_not)/P_orb

def orbit_velocity(a, P_orb):
    return 2 * np.pi * a / P_orb 

def rv_amplitude(a, P_orb, i):
    v_orb = orbit_velocity(a, P_orb)
    return v_orb * np.sin(i)

def doppler_correction(a, P_orb, i, t, T_not, v_sys, v_bary, Kp=None):
    if Kp is None:
        Kp = rv_amplitude(a, P_orb, i)
    phi = orbital_phase(t, T_not, P_orb)

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
    v_bary_timeseries = []
    all_doppler_corrects = []

    for jj in range(len(mjd_obs)):
        v_bary = compute_vbary_timeseries(ra[jj], dec[jj], mjd_obs[jj], location)
        v_bary_timeseries.append(v_bary)

        if Kp is None:
            correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs[jj], T_not=T_not, v_sys=v_sys, v_bary=v_bary)

        else:
            correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs[jj], T_not=T_not, v_sys=v_sys, v_bary=v_bary, Kp=Kp)

        all_doppler_corrects.append(correction)

    new_vel_grids = []

    for kk in range(len(summed_ccf)):
        new_vel_grid = v_shift_range + all_doppler_corrects[kk]
        new_vel_grids.append(new_vel_grid)

    min_v, max_v = -50000, 50000
    common_v_grid = np.linspace(min_v, max_v, 100)  # Common velocity grid for cropping

    cropped_ccf = []
    cropped_v = []

    for i in range(len(new_vel_grids)):
        v = new_vel_grids[i]
        ccf = summed_ccf[i]
            
        # Define interpolator over current velocity grid
        interp_func = interp1d(v, ccf, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_ccf = interp_func(common_v_grid)  # now length 100

        cropped_ccf.append(interp_ccf)
        cropped_v.append(common_v_grid)

    return np.array(cropped_ccf), np.array(cropped_v)

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

    summed_ccf = ccf(filtered_all_pca, filtered_all_wave, v_shift_range, sim_wave, sim_flux)

    cropped_ccf_array, cropped_v_grid = doppler_correct_ccf(summed_ccf, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys)

    in_transit = remove_out_of_transit(
    transit_start_end, np.sum(cropped_ccf_array, axis=0), mjd_obs)

    return cropped_ccf_array, cropped_v_grid, in_transit

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

def sn_map(summed_ccf, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, transit_start_end):
    """
    Compute SNR map for the CCF.
    """
    Kp_range = np.arange(0, 250_000, 1000)  # in m/s
    Kp_range_ccf = []   

    for Kp in Kp_range:                         
        this_cropped_ccf, _ = doppler_correct_ccf(summed_ccf, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, Kp=Kp)

        Kp_range_ccf_in_transit = remove_out_of_transit(
            transit_start_end=transit_start_end,
            grid=this_cropped_ccf,
            mjd_obs=mjd_obs)
        
        Kp_range_ccf.append(Kp_range_ccf_in_transit)

    return np.array(Kp_range_ccf)
        