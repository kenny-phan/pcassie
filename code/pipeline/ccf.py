import numpy as np
import astropy.units as u

from astropy.constants import c
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from scipy.interpolate import interp1d

def doppler_shift(wave_arr, velocity):
    """velocity in m/s"""
    return wave_arr * (1 + (velocity * u.m / u.s )/ c)

def cross_correlation_function(x, y, d):
    """
    Compute the normalized cross-correlation coefficient (CCF) of two 1D arrays x and y at delay d.
    Positive d means y is shifted right (delayed).
    Returns the correlation coefficient at delay d.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    n = min(len(x), len(y))

    if d > 0:
        if d >= n:
            return 0.0
        x_seg = x[:n-d]
        y_seg = y[d:n]
    elif d < 0:
        if -d >= n:
            return 0.0
        x_seg = x[-d:n]
        y_seg = y[:n+d]
    else:
        x_seg = x[:n]
        y_seg = y[:n]

    if len(x_seg) == 0 or len(y_seg) == 0:
        return 0.0

    x_mean = np.mean(x_seg)
    y_mean = np.mean(y_seg)
    numerator = np.sum((x_seg - x_mean) * (y_seg - y_mean))
    denominator = np.sqrt(np.sum((x_seg - x_mean)**2)) * np.sqrt(np.sum((y_seg - y_mean)**2))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def compute_ccf_over_velocity_range(data_wave, data_flux, model_flux, velocity_range):
    """
    Compute CCF between observed data and Doppler-shifted model over a velocity grid.

    Parameters:
        data_wave (1D array): Wavelength grid of observed data
        data_flux (1D array): Observed flux
        model_flux (1D array): Model flux defined on same wave grid
        velocity_range (1D array): Velocities (m/s) to Doppler shift model to

    Returns:
        velocities (1D array), ccfs (1D array)
    """
    ccfs = []
    interp_model = interp1d(data_wave, model_flux, kind='linear', bounds_error=False, fill_value=np.nan)

    for v in velocity_range:
        shifted_wave = doppler_shift(data_wave, v)
        shifted_model_flux = interp_model(shifted_wave)

        # Mask NaNs introduced by shifting out of bounds
        valid = ~np.isnan(shifted_model_flux) & ~np.isnan(data_flux)
        if np.sum(valid) < 10:
            ccfs.append(0.0)
            continue

        cc = cross_correlation_function(data_flux[valid], shifted_model_flux[valid], d=0)
        ccfs.append(cc)

    return velocity_range, np.array(ccfs)

def sum_ccf_matrices(data_wave, normalized_flux_array, pca_subtracted, segment_indices, sim_wave, sim_flux):
    total_vel_grid = []
    total_ccf_vals = []

    for i in range(len(pca_subtracted)):
        start, end = segment_indices[i]
        wave_i = data_wave[start:end]
        flux_i = normalized_flux_array[:, start:end]
        pca_i = pca_subtracted[i]
        nanmask = ~np.isnan(wave_i) & ~np.isnan(flux_i[0])

        sort_idx = np.argsort(sim_wave)
        sim_interp = interp1d(sim_wave[sort_idx], sim_flux[sort_idx], bounds_error=False, fill_value=0)
        sim_on_obs_grid = sim_interp(wave_i[nanmask] * 0.001)

        # Velocity range from -100 km/s to 100 km/s in 1 km/s steps
        velocities = np.arange(-100_000, 100_001, 1000)  # in m/s

        # Run the CCF
        vel_grid_2d = []
        ccf_vals_2d = []

        for ii in range(flux_i.shape[0]):
            vel_grid, ccf_vals = compute_ccf_over_velocity_range(wave_i[nanmask], pca_i[ii], -sim_on_obs_grid, velocities)
            vel_grid_2d.append(vel_grid)
            ccf_vals_2d.append(ccf_vals)
        
        total_vel_grid.append(vel_grid_2d)
        total_ccf_vals.append(ccf_vals_2d)
    
    summed_ccf = np.sum(np.array(total_ccf_vals), axis=0)
    consistent_vel_grid = np.array(vel_grid_2d)[0]
    return summed_ccf, consistent_vel_grid


# doppler shift correction functions

def orbital_phase(t, T_not, P_orb):
    return (t - T_not)/P_orb

def orbit_velocity(a, P_orb):
    return 2 * np.pi * a / P_orb 

def rv_amplitude(a, P_orb, i):
    v_orb = orbit_velocity(a, P_orb)
    return v_orb * np.sin(i)

def doppler_correction(a, P_orb, i, t, T_not, v_sys, v_bary):
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

def doppler_correct_ccf(summed_ccf, consistent_vel_grid, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys):
    v_bary_timeseries = []
    all_doppler_corrects = []

    for jj in range(len(mjd_obs)):
        v_bary = compute_vbary_timeseries(ra[jj], dec[jj], mjd_obs[jj], location)
        v_bary_timeseries.append(v_bary)

        correction = doppler_correction(a=a, P_orb=P_orb, i=i, t=mjd_obs[jj], T_not=T_not, v_sys=v_sys, v_bary=v_bary)
        all_doppler_corrects.append(correction)

    new_vel_grids = []

    for kk in range(len(summed_ccf)):
        new_vel_grid = consistent_vel_grid + all_doppler_corrects[kk]
        new_vel_grids.append(new_vel_grid)

    min_v, max_v = -50000, 50000

    cropped_ccf = []
    cropped_v = []

    for i in range(len(new_vel_grids)):
        v = new_vel_grids[i]
        ccf = summed_ccf[i]
            
        mask = (v >= min_v) & (v <= max_v)
        
        if np.sum(mask) == 0:
            continue  # skip if nothing in range
        
        cropped_ccf.append(ccf[mask])
        cropped_v.append(v[mask])

    return np.array(cropped_ccf), np.array(cropped_v)

def run_ccf_on_detector_segments(wave_1d, normalized_flux_array, 
                                 all_pca, segment_indices, sim_wave, 
                                 sim_flux, mjd_obs, ra, dec, location, 
                                 a, P_orb, i, T_not, v_sys, remove_segments=None): #sim_wave in um for now

    if remove_segments is None:
        remove_segments = []

    # Filter segments
    keep_indices = [i for i in range(len(segment_indices)) if i not in remove_segments]

    filtered_segment_indices = [segment_indices[i] for i in keep_indices]
    filtered_all_pca = [all_pca[i] for i in keep_indices]

    summed_ccf, consistent_vel_grid = sum_ccf_matrices(wave_1d, normalized_flux_array, filtered_all_pca, filtered_segment_indices, sim_wave, sim_flux)
    cropped_ccf_array, cropped_v_grid = doppler_correct_ccf(summed_ccf, consistent_vel_grid, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys)
    return cropped_ccf_array, cropped_v_grid