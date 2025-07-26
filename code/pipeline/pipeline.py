import numpy as np

from pipeline.utility_functions import split_detectors
#from pipeline.calibration import calibrate_cr2res
from pipeline.pca_subtraction import pca_subtraction
#from pipeline.pca_diagnostics import plot_spectral_square
from pipeline.ccf import run_ccf_on_detector_segments
from pipeline.ccf_tests import sn_map, welch_t_test, find_max_sn_in_expected_range

def pipeline(wave, flux, sim_wave, sim_flux, mjd_obs, ra, dec, location, 
                                 a, P_orb, i, T_not, v_sys, v_shift_range=np.linspace(-100_000, 100_000, 201), 
                                 transit_start_end=None, gap_size=5, remove_segments=[], 
                                 first_components=5, last_components=5):

    print("Running pipeline...")
    print("Normalizing flux array...")
    normalized_flux_array, segment_indices = split_detectors(wave, flux, m=gap_size)

    if remove_segments is None:
        remove_segments = []

    # Filter segments
    keep_indices = [i for i in range(len(segment_indices)) if i not in remove_segments]
    print(f"Retaining detector indices {keep_indices}")

    print("Running PCA subtraction on detector segments...")
    jax_tdm, jax_wdm, all_wave = [], [], []

    for keep_index in keep_indices:
        start, end = segment_indices[keep_index]
        #print("start, end: ", start, end)
        wave_i = wave[0, start:end]
        flux_i = normalized_flux_array[:, start:end]
        nanmask = ~np.isnan(wave_i) & ~np.isnan(flux_i[0])
        #print(flux_i[:, nanmask].shape)
        tdm_concat, wdm_concat = pca_subtraction(flux_i[:, nanmask], 0, np.sum(nanmask), first_comps=first_components, last_comps=last_components, pre=True)
        
        jax_tdm.append(tdm_concat)
        jax_wdm.append(wdm_concat)
        all_wave.append(wave_i[nanmask])

    all_tdm = [np.array(x) for x in jax_tdm]
    all_wdm = [np.array(x) for x in jax_wdm]

    print("Running CCF on detector segments...")
    earth_frame_ccf, planet_frame_ccf, planet_frame_vgrid, in_transit = run_ccf_on_detector_segments(all_wave, 
                                 all_tdm, v_shift_range, keep_indices, sim_wave, 
                                 sim_flux, mjd_obs, ra, dec, location, 
                                 a, P_orb, i, T_not, v_sys, transit_start_end)
    
    print("Making the S/N map...")
    Kp_range_ccf, sn_map_array = sn_map(planet_frame_ccf, planet_frame_vgrid, mjd_obs, ra, dec, location, a, P_orb, i, T_not, v_sys, transit_start_end) 

    print("Performing Welch's t-test...")
    in_trail_vals, out_of_trail_vals, t_stat, p_value = welch_t_test(Kp_range_ccf)   
    
    print("Pipeline completed successfully.")
    return all_tdm, all_wdm, all_wave, earth_frame_ccf, planet_frame_ccf, planet_frame_vgrid, in_transit, Kp_range_ccf, sn_map_array, in_trail_vals, out_of_trail_vals, t_stat, p_value


def sample_full_pca_components(first_components, end_components, wave, flux, sim_wave, 
        sim_flux, mjd_obs, ra, dec, location,
        a, P_orb, i, T_not, v_sys, transit_start_end,
        remove_segments=[], sn_test=-50, sn_max=-100):

    first_best_results, first_sn_max, first_best_components = sample_components(
        first_components, end_components, wave, flux, sim_wave, 
        sim_flux, mjd_obs, ra, dec, location,
        a, P_orb, i, T_not, v_sys, transit_start_end=transit_start_end,
        remove_segments=remove_segments, sn_test=sn_test, sn_max=sn_max, sample_end=False)

    best_results, sn_max, last_best_components = sample_components(
        end_components, first_best_components - 1, wave, flux, sim_wave, 
        sim_flux, mjd_obs, ra, dec, location,
        a, P_orb, i, T_not, v_sys, transit_start_end=transit_start_end,
        remove_segments=remove_segments, sn_test=first_sn_max, sn_max=sn_max, sample_end=True, results=first_best_results)

    print(f"Best fc = {first_best_components - 1}, best lc = {last_best_components - 1}, S/N = {sn_max}")

    return best_results, first_best_components - 1, last_best_components - 1, sn_max
    

def sample_components(start_components, stable_components, wave, flux, sim_wave, 
                      sim_flux, mjd_obs, ra, dec, paranal,
                        a, P_orb, i, T_not, v_sys,
                        transit_start_end,
                        remove_segments=[], sn_test=-50, sn_max=-100, sample_end=False, results=None):

    while sn_test >= sn_max:

        best_results = results
        sn_max = sn_test
        best_components = start_components

        # Run pipeline
        if sample_end: 
            print("sampling from the end. new sn_max =", sn_test, "fc = ", stable_components, "lc =", start_components)

            results = pipeline(
                wave, flux, sim_wave, sim_flux, mjd_obs, ra, dec, paranal,
                a, P_orb, i, T_not, v_sys, transit_start_end=transit_start_end, remove_segments=remove_segments,
                first_components=stable_components, last_components=start_components
            )

        else: 
            print("sampling from the start. new sn_max =", sn_test, "fc = ", start_components, "lc =", stable_components)

            results = pipeline(
                wave, flux, sim_wave, sim_flux, mjd_obs, ra, dec, paranal,
                a, P_orb, i, T_not, v_sys, transit_start_end=transit_start_end, remove_segments=remove_segments,
                first_components=start_components, last_components=stable_components 
            )

        # Compute S/N
        sn_test = find_max_sn_in_expected_range(results[8], results[5] / 1000, a, P_orb, i)
        print("sn_test =", sn_test)

        start_components += 1

    return best_results, sn_max, best_components