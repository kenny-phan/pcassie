import numpy as np

from pcassie.utility_functions import split_detectors, debug_print
from pcassie.pca_subtraction import pca_subtraction
from pcassie.ccf import run_ccf_on_detector_segments
from pcassie.ccf_tests import sn_map, welch_t_test, find_max_sn_in_expected_range

def pipeline(sim_wave, sim_flux, v_shift_range=np.linspace(-100_000, 100_000, 201), Kp_range=np.linspace(50_000, 150_000, 101), verbose=True, eighcalc='numba', **kwargs):
    """Runs principal component analysis and cross-correlation with simulated signal. Additionally outputs
    a signal to noise map and Welch's T-test values. It is more legibple to run ``results = pipeline(...)`` 
    and handle the outputs as indices of ``results``.
     
    Parameters
    ----------
    sim_wave: array
        1d wavelength grid for the simulated specrum. Must be the same shape as kwargs['wave'].
    sim_flux: array
        1d flux values for the simulated spectrum. Must be the same shape as kwargs['wave'].
    v_shift_range: array
        1d range of velocity values to sample in the cross-correlation function. Ideally, this is 
        in units of meters/second and orgamized with a 1 km/s step, e.g. the default ``np.linspace(-100_000, 100_000, 201)``.
        At this stage, a 1 km/s step is assumed to be equivalent to 1 pixel in the CCF grid.
    verbose: boolean
        Boolean statement to toggle print statements throughout the pipeline.
    **kwargs: 
        For ease of use, ``pipeline`` takes keyword arguments for the many inputs it requires. This is typically neater organized as:
        ``obs_args = {'wave', 'flux', 'mjd_obs', 'ra', 'dec' 'location'}``, ``planet_args = {'a', 'P_orb', 'i', 'T_not', 'v_sys'}``,
        ``params = {'transit_start_end', 'gap_size', 'remove_segments', 'first_components', 'last_components'}`` and called as: 
        ``pipeline(sim_wave, sim_flux, **obs_args, **planet_args, **params)``. Please refer to the tutorial for an example use. 
            wave: array
                2d wavelength array of dimensions n sepectra x wavegrid. wavegrid must be consistent across all spectrum, i.e. this is 
                effectively a 1d array repeated for the number of spectra. This will be changed to only input a 1d array in later versions.
            flux: array
                2d flux array of dimensions n spectra x wavegrid. n spectra should be sorted with respect to time, i.e. ``flux[0]`` is the spectra 
                observed at ``mjd_obs[0]``.
            mjd_obs: list or array
                List or 1d array of the MJD-OBS value in the specta's ``.fits`` files. In my implementation, I use the mean of MJD-START and MJD-END, 
                but so long as ``mjd_obs`` is properly ordered (e.g. first observation to last observation) the use of MJD-START, MJD-END, or the mean is 
                largely irrelevant. However, it is necessary to have units of MJD (modified julian date).
            ra: list or array
                List or 1d array of the RA value in the spectra's ``.fits`` files. Should be ordered in accordance with ``mjd_obs``, e.g. ``mjd_obs[0]``
                corresponds to ``ra[0]`` and so on.
            dec: list or array
                List or 1d array of the DEC value in the spectra's ``.fits`` files. Should be ordered in accordance with ``mjd_obs``, e.g. ``mjd_obs[0]``
                corresponds to ``dec[0]`` and so on.
            location: EarthLocation
                ``astropy`` EarthLocation of the observation facility. Refer to the tutorial and ``astropy`` documentation for more information. The key 
                is that this value fullfills the ``location`` argument in ``astropy.time.Time``.
            a: float
                Semi-major axis of the exoplanet in units of AU. This can be found with NASA's Exoplanet Archive.
            P_orb: float
                Orbital period of the exoplanet in units of days. This can be found with NASA's Exoplanet Archive.
            i: float
                The inclination of the exoplanet in units of degrees. This can be found with NASA's Exoplanet Archive.
            T_not: float
                A mid-transit time of the exoplanet in units of modified julian days (MJD). It is not neccesary that 
                this date correponds to the specific transit in your data (I think). TBH I will have to double check 
                with some tests, and this value will likely be unnecessary in later iterations due to the ``transit_start_end``
                input. 
            v_sys: float
                The radial velocity of the exoplanetary system in units of km/s. This can be found with NASA's Exoplanet Archive.
            transit_start_end: tuple
                The start and end times of the transit in your data in units of modified julian date (MJD). Takes the form (start, end).
            gap_size: int
                Minimum number of NaNs between detector segments in your flux array to qualify as a separate detector. For CRIRES+, a good 
                value is ``gap_size=5``. However, test yourself to ensure all detectors are identified. You can do this by comparing the number of 
                detected detectors to those shown in, e.g. the ESO exposure time calculator https://www.eso.org/observing/etc/.
            remove_segments: array of ints
                Indices of the detectors that you do not want to include in the analysis. This is typically because some detectors are saturated 
                with telluric noise. E.g., in the CRIRES+ analysis, I did not want detectors 1, 2, 3, 4, 6, and 20; therefore ``remove_segments=[0, 1, 2, 3, 5, 19]``.
            first_components: int
                Number of components to remove starting from the first component found in PCA. The first components correspond to correlated signal, primarily the 
                stellar and telluric spectrum. It helps for this value to immediately disregard the first few components for this reason. 3-5 have 
                been good values in the past. If this value is too low, the pipeline could get trapped in a local maximum in the signal to noise (S/N), which 
                would lead to a suboptimal decomposition.
            last_components: int
                Number of components to remove starting from the last component found in PCA. The last components correspond to uncorrelated signal, e.g. instrumental 
                noise. It is less necessary to immediately disregard components here, particulary because the planetary signal sits right at the edge of the noise. 
                In the past, values from 0-3 have worked OK for me.
    Returns
    -------
    list
        Time-domain PCA reduction. Preferred over the wavelength domain. Takes the form of a list with the number of unremoved detectors as its length. Each index of the
        list contains a 2d array of shape n spectra x wavelength grid. The wavelength grids are often not consistent between detectors.
    list
        Wavelength-domain PCA reduction. Takes the form of a list with the number of unremoved detectors as its length. Each index of the list contains a 2d array of shape n spectra x wavelength grid. 
        The wavelength grids are often not consistent between detectors.
    list
        List of 1d wavelength grids for each unremoved detector.
    array
        2d CCF array (n spectra x velocity grid defined in ``v_shift_range``) in the rest frame of the Earth.
    array
        2d CCF array (n spectra x ``planet_frame_vgrid``) doppler corrected into the rest frame of the exoplanet. 
    array
        1d array of the velocity range for the exoplanetary CCF grid. Same format as ``v_shift_range``, however covers a smaller range.
    array
        2d CCF array (n in-transit spectra x ``planet_frame_vgrid``) of only in-transit spectra in the exoplanetary rest frame.
    array
        2d CCF array sampling values in a radial velocity amplitude array (``Kp_range`` x ``planet_frame_vgrid``). Refer to ccf_tests.sn_map.
    array
        2d signal to noise (S/N) map array (``Kp_range`` x ``planet_frame_vgrid``). Refer to ccf_tests.sn_map.
    array
        In-trail values of the S/N map. Refer to ccf_tests.welch_t_test.
    array
        Out-trail values of the S/N map. Refer to ccf_tests.welch_t_test.
    array
        T-statistic of the S/N map. Refer to ccf_tests.welch_t_test.
    array
        p value of the S/N map. Refer to ccf_tests.welch_t_test."""
    wave, flux, mjd_obs, ra, dec, location = kwargs['wave'], kwargs['flux'], kwargs['mjd_obs'], kwargs['ra'], kwargs['dec'], kwargs['location']
    a, P_orb, i, T_not, v_sys, transit_start_end = kwargs['a'], kwargs['P_orb'], kwargs['i'], kwargs['T_not'], kwargs['v_sys'], kwargs['transit_start_end']
    gap_size, remove_segments, first_components, last_components = kwargs['gap_size'], kwargs['remove_segments'], kwargs['first_components'], kwargs['last_components']

    debug_print(verbose, "Running pipeline...")
    debug_print(verbose, "Normalizing flux array...")
    normalized_flux_array, segment_indices = split_detectors(wave, flux, m=gap_size)

    if remove_segments is None:
        remove_segments = []

    # Filter segments
    keep_indices = [i for i in range(len(segment_indices)) if i not in remove_segments]
    debug_print(verbose, f"Retaining detector indices {keep_indices}")

    debug_print(verbose, "Running PCA subtraction on detector segments...")
    all_tdm, all_wdm, all_wave = [], [], []

    for keep_index in keep_indices:
        start, end = segment_indices[keep_index]
        #print("start, end: ", start, end)
        wave_i = wave[0, start:end]
        flux_i = normalized_flux_array[:, start:end]
        nanmask = ~np.isnan(wave_i) & ~np.isnan(flux_i[0])
        #print(flux_i[:, nanmask].shape)
        tdm_concat, wdm_concat = pca_subtraction(flux_i[:, nanmask], 0, np.sum(nanmask), first_comps=first_components, last_comps=last_components, pre=True, eighcalc=eighcalc)
        
        all_tdm.append(tdm_concat)
        all_wdm.append(wdm_concat)
        all_wave.append(wave_i[nanmask])    

    debug_print(verbose, "length of all_tdm: ", len(all_tdm))
    debug_print(verbose, "length of all_wdm: ", len(all_wdm))
    debug_print(verbose, "length of all_wave: ", len(all_wave))
    
    all_tdm = [np.array(x) for x in all_tdm]
    all_wdm = [np.array(x) for x in all_wdm]

    debug_print(verbose, "Running CCF on detector segments...")
    earth_frame_ccf, planet_frame_ccf, planet_frame_vgrid, in_transit = run_ccf_on_detector_segments(all_wave, 
                                 all_tdm, v_shift_range, keep_indices, sim_wave, 
                                 sim_flux, mjd_obs, ra, dec, location, 
                                 a, P_orb, i, T_not, v_sys, transit_start_end, verbose=verbose)
    
    debug_print(verbose, "Making the S/N map...")
    Kp_range_ccf, sn_map_array = sn_map(planet_frame_ccf, planet_frame_vgrid, Kp_range=Kp_range, **kwargs) 

    debug_print(verbose, "Performing Welch's t-test...")
    in_trail_vals, out_of_trail_vals, t_stat, p_value = welch_t_test(Kp_range_ccf)   
    
    debug_print(verbose, "Pipeline completed successfully.")
    return all_tdm, all_wdm, all_wave, earth_frame_ccf, planet_frame_ccf, planet_frame_vgrid, in_transit, Kp_range_ccf, sn_map_array, in_trail_vals, out_of_trail_vals, t_stat, p_value

def sample_components(start_components, stable_components, sim_wave, 
                      sim_flux, v_shift_range=np.linspace(-100_000, 100_000, 201), sn_test=-50, sn_max=-100, sample_end=False, results=None, verbose=True, **kwargs):
    """Samples through a range of the PCA component space to maximize S/N.
    
    Parameters
    ----------
    start_components: int
        Components to immediately remove from start or end. 
    stable_components: int
        Components of either the start or end that remain consistent. E.g. if the optimal first components have 
        already been found, ``stable_components`` is used to maintain the ``first_components`` value.
    sim_wave: array
        Refer to pipeline.pipeline.
    sim_flux: array
        Refer to pipeline.pipline. 
    sn_test: int
        Refer to pipeline.sample_full_pca_components.
    sn_max: int
        Refer to pipeline.sample_full_pca_components.
    sample_end: boolean
        True if optimizing for the best ``last_components``, False if sampling for the est ``first_components``.
    results: list
        Returned list of values from pipeline.pipeline.
    verbose: boolean
        Refer to untility_functions.debug_print.
    **kwargs
        Refer to pipeline.pipeline.
        
    Returns
    -------
    list
        List of optimized results from pipeline.pipeline."""
    # a, P_orb, i = kwargs['a'], kwargs['P_orb'], kwargs['i']

    while sn_test >= sn_max:

        best_results = results
        sn_max = sn_test
        best_components = start_components

        # Run pipeline
        if sample_end: 
            debug_print(verbose, f"sampling from the end. new sn_max = {sn_test} fc = {stable_components} lc = {start_components}")
            
            kwargs['first_components'] = stable_components 
            kwargs['last_components'] = start_components

            results = pipeline(
                sim_wave, sim_flux, v_shift_range=v_shift_range, verbose=verbose, **kwargs
            )

        else: 
            debug_print(verbose, f"sampling from the start. new sn_max = {sn_test} fc = {start_components} lc = {stable_components}")

            kwargs['first_components'] = start_components
            kwargs['last_components'] = stable_components

            results = pipeline(
                sim_wave, sim_flux, v_shift_range=v_shift_range, verbose=verbose, **kwargs
            )

        # Compute S/N
        sn_test = find_max_sn_in_expected_range(results[8], results[5] / 1000, **kwargs)
        debug_print(verbose, "sn_test =", sn_test)

        start_components += 1

    return best_results, sn_max, best_components    


def sample_full_pca_components(sim_wave, 
        sim_flux, v_shift_range=np.linspace(-100_000, 100_000, 201), sn_test=-50, sn_max=-100, verbose=True, **kwargs):
    """Loops pipeline() through the component space of the principal 
    component analysis, progressively removing the first components 
     (associated with the stellar spectrum and tellurics) until S/N in 
     the range of the exoplanetary parameters (+- 15 km/s from Kp=Kp, velocity in planet frame = 0)
    is maximized, then doing the same to the end components 
    (associated with uncorrelated/instrumental noise).
    
    Parameters
    ----------
    sim_wave: array
        Refer to pipeline.pipeline.
    sim_flux: array 
        Refer to pipeline.pipeline.
    sn_test: int
        Starting value for the S/N test value. Must be greater than ``sn_max`` but still start quite low.
    sn_max: int
        Starting value for the S/N max value. Must be less than ``sn_test``.
    verbose: boolean
        Refer to pipeline.pipeline
    **kwargs
        Refer to pipeline.pipeline.

    Returns
    -------
    list
        List of the returned values in pipeline.pipeline for the optimal range of PCA components.
    int
        Optimal value for ``first_components``.
    int
        Optimal value for ``last_components``.
    float
        Maximum S/N value in planetary parameter range. 
    """
    first_components, last_components = kwargs['first_components'], kwargs['last_components']

    first_best_results, first_sn_max, first_best_components = sample_components(
        first_components, last_components, sim_wave, 
        sim_flux, v_shift_range=v_shift_range, sn_test=sn_test, sn_max=sn_max, sample_end=False, verbose=verbose, **kwargs)

    best_results, sn_max, last_best_components = sample_components(
        last_components, first_best_components - 1, sim_wave, 
        sim_flux, v_shift_range=v_shift_range, sn_test=first_sn_max, sn_max=sn_max, sample_end=True, results=first_best_results, verbose=verbose, **kwargs)

    debug_print(verbose, f"Best fc = {first_best_components - 1}, best lc = {last_best_components - 1}, S/N = {sn_max}")

    return best_results, first_best_components - 1, last_best_components - 1, sn_max
    
