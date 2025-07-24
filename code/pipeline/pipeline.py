import numpy as np

from pipeline.utility_functions import split_detectors
#from pipeline.calibration import calibrate_cr2res
from pipeline.pca_subtraction import pca_subtraction
#from pipeline.pca_diagnostics import plot_spectral_square
from pipeline.ccf import run_ccf_on_detector_segments, sn_map

def pipeline(wave, flux, sim_wave, sim_flux, mjd_obs, ra, dec, location, 
                                 a, P_orb, i, T_not, v_sys, v_shift_range=np.arange(-100_000, 100_000, 1000), transit_start_end=None, gap_size=5, remove_segments=[], plot=False):

    normalized_flux_array, segment_indices = split_detectors(wave, flux, m=gap_size)

    jax_tdm, jax_wdm, all_wave = [], [], []

    for ii in range(len(segment_indices)):
        start, end = segment_indices[ii]
        #print("start, end: ", start, end)
        wave_i = wave[0, start:end]
        flux_i = normalized_flux_array[:, start:end]
        nanmask = ~np.isnan(wave_i) & ~np.isnan(flux_i[0])
        #print(flux_i[:, nanmask].shape)
        tdm_concat, wdm_concat = pca_subtraction(flux_i[:, nanmask], 0, np.sum(nanmask), first_comps=10, last_comps=10, pre=True)
        
        jax_tdm.append(tdm_concat)
        jax_wdm.append(wdm_concat)
        all_wave.append(wave_i[nanmask])

    all_tdm = [np.array(x) for x in jax_tdm]
    all_wdm = [np.array(x) for x in jax_wdm]

    cropped_ccf_array, cropped_v_grid, in_transit = run_ccf_on_detector_segments(all_wave, 
                                 all_tdm, v_shift_range, segment_indices, sim_wave, 
                                 sim_flux, mjd_obs, ra, dec, location, 
                                 a, P_orb, i, T_not, v_sys, transit_start_end, remove_segments=None)
    
    #kp_sn = sn_map(summed_ccf, consistent_vel_grid, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, transit_start_end)
    
    return all_tdm, all_wdm, all_wave, cropped_ccf_array, cropped_v_grid, in_transit#, kp_sn

    