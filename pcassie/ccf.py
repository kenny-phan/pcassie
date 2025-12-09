import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from numba import njit
from scipy.interpolate import interp1d

from pcassie.utility_functions import debug_print

@njit
def doppler_shift(wave_arr, velocity):
    """
    Doppler shift the wavelength array by a velocity (in m/s).
    
    Parameters
    ----------
    wave_arr: array
        1d wavelengths array (in any units, e.g., nm or µm)
    velocity: float
        Velocity in m/s

    Returns
    -------
    array 
        Doppler-shifted wavelengths
    """
    C = 299792458.0  # Speed of light in m/s
    return wave_arr * (1 + velocity / C)


def ccf(all_pca, all_wave, v_shift_range, sim_wave, sim_flux, speed=True, verbose=False):
    """Loops through all detectors and spectra to 
    compute the cross-correlation function of each spectrum 
    with the simulated spectrum.
    
    Parameters
    ----------
    all_pca: list
        PCA removed spectra with dimensions (n detectors x n spectra x wavelength grid).
    all_wave: list
        Wavelength arrays (n detectors x wavelength grid) for the corresponding PCA subtracted spectra. 
    v_shift_range: array
        1d array of velocity shifts to sample. Should be structured such that 
        the array has a step of 1 km/s, e.g. np.linspace(-100_000, 100_000, 201) with units of meters.
    sim_wave: array
        Simulated wave array.
    sim_flux: array
        Simulated flux array. 
    speed: boolean
        If True, uses numpy for interpolation (much faster). If False, uses scipy's interp1d (more accurate). 
        I reccomend using numpy for PCA sampling and scipy for science data.
        
    Returns
    -------
    array
        2d cross-correlation function array (n spectra x v_shift)"""
    sort_idx = np.argsort(sim_wave)
    sorted_wave = sim_wave[sort_idx]
    sorted_flux = sim_flux[sort_idx]

    stacked_segment_xcorr = []
    
    debug_print(verbose, "sorted wave, flux")
    for detector_spectra, detector_wavs in zip(all_pca, all_wave):
        detector_xcorr = []

        debug_print(verbose, f"cycling through {len(detector_spectra)} spectra")
        for single_data_spectra in detector_spectra:
            norm_xcorr_arr = []
            debug_print(verbose, f"single_data_spectra: {len(single_data_spectra)}")
            for v_shift in v_shift_range:
                # Doppler shift the template wavelength
                shifted_wave = doppler_shift(sorted_wave, v_shift)

                # Interpolate shifted flux onto segment wavelength grid
                if speed:   
                    # np.interp is faster but less accurate for v < 0; good for sampling the PCA space
                    shifted_flux = np.interp(detector_wavs, shifted_wave, sorted_flux)

                else:
                    # scipy's interp1d is slowe but more accurate for v < 0; good for getting science values 
                    interp_func = interp1d(shifted_wave, sorted_flux, bounds_error=False, fill_value=0.0)
                    shifted_flux = interp_func(detector_wavs)

                shifted_flux = (shifted_flux - np.mean(shifted_flux)) #/ np.std(shifted_flux)
                single_data_spectra = (single_data_spectra - np.mean(single_data_spectra)) #/ np.std(single_data_spectra)

                # Cross-correlate (dot product)
                xcorr = np.dot(shifted_flux, single_data_spectra)
                denom = np.sqrt(np.sum(shifted_flux**2) * np.sum(single_data_spectra**2))
                if denom == 0:
                    norm_xcorr = 0
                else:
                    norm_xcorr = xcorr / denom

                norm_xcorr_arr.append(norm_xcorr)

            norm_xcorr_arr = np.array(norm_xcorr_arr)
            debug_print(verbose, f"norm_xcorr_arr shape: {norm_xcorr_arr.shape}")
            detector_xcorr.append(norm_xcorr_arr)

        detector_xcorr = np.array(detector_xcorr)
        debug_print(verbose, f"detector_xcorr shape: {detector_xcorr.shape}")
        stacked_segment_xcorr.append(detector_xcorr)

    return np.sum(np.array(stacked_segment_xcorr), axis=0)


# doppler shift correction functions
def orbital_phase(t, T_not, P_orb):
    """Calculates the orbital phase as 
    phi(t) = (t - T_not) / P_orb
    where phi(t) is phase as a function of time, t is time, 
    T_not is the mid-transit time, and P_orb is the orbital period.
    Please ensure units match."""
    return (t - T_not)/P_orb


def orbit_velocity(a, P_orb):
    """Calculates orbital velocity as
    v_orb = 2*pi*a / P_orb
    where a is the semi-major axis and 
    P_orb is the orbital period.
    Please ensure units match."""
    return 2 * np.pi * a / P_orb 


def rv_amplitude(a, P_orb, i):
    """Calculates the radial velocity amplitude as:
    Kp = v_orb * sin(i)
    where v_orb is the orbital velocity, i is the inclination,
    a is the semi-major axis, and P_orb is the orbital period. 
    See orbit_velocity for the calculation of v_orb.
    Please ensure units match."""
    v_orb = orbit_velocity(a, P_orb)
    return v_orb * np.sin(i)


def doppler_correction(a, P_orb, i, t, T_not, v_sys, v_bary, Kp=None, verbose=False):
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

    debug_print(verbose, f"Kp: {Kp} m/s, orbital phase: {phi}")

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


def doppler_correct_ccf(summed_ccf, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, Kp=None, verbose=False):
    """Corrects the full cross-correlation array for the Doppler shift according to
    Vp = Kp*sin[2*pi*phi(t)] + v_sys + v_bary
    where Vp is the velocity correction, v_sys is the systems radial velocity, and 
    v_bary is the barcentric correction. Refer to rv_amplitude and orbital_phase 
    for the definitions of Kp and phi(t)"""
    v_bary_timeseries = compute_vbary_timeseries(ra, dec, mjd_obs, location)
    debug_print(verbose, f"v_bary_timeseries: {v_bary_timeseries}")
    all_doppler_corrects = []

    for jj in range(len(mjd_obs)):

        if Kp is None:
            correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs[jj], T_not=T_not, v_sys=v_sys, v_bary=v_bary_timeseries[jj])

        else:
            correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs[jj], T_not=T_not, v_sys=v_sys, v_bary=v_bary_timeseries[jj], Kp=Kp)

        all_doppler_corrects.append(correction)

    # check for nans in doppler correction
    debug_print(verbose, f"Doppler correction contains NaNs: {np.any(np.isnan(all_doppler_corrects))}")
    debug_print(verbose, f"Doppler corrections: {all_doppler_corrects}")

    new_vel_grids = []

    for kk in range(len(summed_ccf)):
        new_vel_grid = v_shift_range + all_doppler_corrects[kk]
        #debug_print(verbose, f"start, end of new_vel_grid: {new_vel_grid[0]}, {new_vel_grid[-1]}")
        new_vel_grids.append(new_vel_grid)

    min_v, max_v = -50000, 50000
    common_v_grid = np.linspace(min_v, max_v, 101)  # Common velocity grid for cropping

    #check for nans in new_vel_grids
    debug_print(verbose, f"New velocity grids contain NaNs: {np.any([np.any(np.isnan(v)) for v in new_vel_grids])}")

    cropped_ccf = []

    for i in range(len(new_vel_grids)):
        v = new_vel_grids[i]
        ccf = summed_ccf[i]
            
        common_mask = (v >= min_v) & (v <= max_v)
        debug_print(verbose, f"common velocity grid shape: {common_v_grid.shape}, v shape: {v.shape}, ccf shape: {ccf.shape}, common_mask sum: {np.sum(common_mask)}")
        interp_ccf = np.interp(common_v_grid, v[common_mask], ccf[common_mask])
        cropped_ccf.append(interp_ccf)

    #check for nans in cropped_ccf
    debug_print(verbose, f"Cropped CCF contains NaNs: {np.any(np.isnan(cropped_ccf))}")

    return np.array(cropped_ccf), common_v_grid 


def remove_out_of_transit(transit_start_end, grid, mjd_obs):
    """remove spectra outside of ingress (transit start) and egress (trasnit end)."""
    transit_start, transit_end = transit_start_end
    transit_mask = (mjd_obs >= transit_start) & (mjd_obs <= transit_end)
    filtered_grid = [grid[i] for i in range(grid.shape[0]) if transit_mask[i]]

    return filtered_grid


def run_ccf_on_detector_segments(all_wave, 
                                 all_pca, v_shift_range, segment_indices, sim_wave, 
                                 sim_flux, mjd_obs, ra, dec, location, 
                                 a, P_orb, i, T_not, v_sys, transit_start_end, verbose=False): #sim_wave in um for now
    """Full pipeline to runn cross-correlation analysis on your full dataset (n detectors x n spectra x wavelength range)"""
    
    earth_frame_ccf = ccf(all_pca, all_wave, v_shift_range, sim_wave, sim_flux, verbose=verbose)

    # check for NaNs in earth_frame_ccf
    debug_print(verbose, f"Earth frame CCF contains NaNs: {np.any(np.isnan(earth_frame_ccf))}")
    debug_print(verbose, f"Earth frame CCF shape: {earth_frame_ccf.shape}")
    planet_frame_ccf, planet_frame_vgrid = doppler_correct_ccf(earth_frame_ccf, v_shift_range, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, verbose=verbose)

    in_transit = remove_out_of_transit(
    transit_start_end, planet_frame_ccf, mjd_obs)

    return earth_frame_ccf, planet_frame_ccf, planet_frame_vgrid, in_transit