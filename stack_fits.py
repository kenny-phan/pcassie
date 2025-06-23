import numpy as np
from astropy.io import fits

def extract_spectrum_from_fits(filepath, hdu_range=range(1, 21)):
    wave_all, flux_all, err_all, orig_flux_all = [], [], [], []
    try:
        with fits.open(filepath) as hdul:
            for hdu_index in hdu_range:
                data = hdul[hdu_index].data
                if data is not None:
                    wave_all.append(data['WAVE'])
                    flux_all.append(data['FLUX'])
                    err_all.append(data['ERR'])
                    orig_flux_all.append(data['ORIG_FLUX'])
        return {
            'wave': np.concatenate(wave_all),
            'flux': np.concatenate(flux_all),
            'err': np.concatenate(err_all),
            'orig_flux': np.concatenate(orig_flux_all)
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

import os
from datetime import datetime

def is_valid_dir(path):
    # Return False if any folder in the path starts with '_'
    return not any(part.startswith('_') for part in path.split(os.sep))

def get_date_obs(filepath):
    try:
        with fits.open(filepath) as hdul:
            date_obs_str = hdul[0].header.get('DATE-OBS')
            if date_obs_str:
                return datetime.fromisoformat(date_obs_str)
    except Exception as e:
        print(f"Error reading DATE-OBS from {filepath}: {e}")
    return None

def extract_spectrum_from_fits(filepath, hdu_range=range(1, 21)):
    wave_all, flux_all, err_all, orig_flux_all = [], [], [], []
    try:
        with fits.open(filepath) as hdul:
            for hdu_index in hdu_range:
                data = hdul[hdu_index].data
                if data is not None:
                    wave_all.append(data['WAVE'])
                    flux_all.append(data['FLUX'])
                    err_all.append(data['ERR'])
                    orig_flux_all.append(data['ORIG_FLUX'])
        return {
            'wave': np.concatenate(wave_all),
            'flux': np.concatenate(flux_all),
            'err': np.concatenate(err_all),
            'orig_flux': np.concatenate(orig_flux_all)
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def find_sci_fits(root_dir):
    fits_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if not is_valid_dir(dirpath):
            continue
        for filename in filenames:
            if filename.startswith("SCI") and filename.endswith(".fits"):
                full_path = os.path.join(dirpath, filename)
                date_obs = get_date_obs(full_path)
                if date_obs:
                    fits_files.append((date_obs, full_path))
                else:
                    print(f"No valid DATE-OBS in {full_path}")
    # Sort by date_obs
    return sorted(fits_files, key=lambda x: x[0])

from scipy.interpolate import interp1d

def interpolate_spectrum_to_wave(spec, wave_ref):
    interp_flux = interp1d(spec['wave'], spec['flux'], kind='linear', bounds_error=False, fill_value=np.nan)
    interp_err = interp1d(spec['wave'], spec['err'], kind='linear', bounds_error=False, fill_value=np.nan)
    interp_orig_flux = interp1d(spec['wave'], spec['orig_flux'], kind='linear', bounds_error=False, fill_value=np.nan)

    return {
        'wave': wave_ref,
        'flux': interp_flux(wave_ref),
        'err': interp_err(wave_ref),
        'orig_flux': interp_orig_flux(wave_ref),
    }

def stack_all_spectra(root_dir):
    fits_file_list = find_sci_fits(root_dir)
    wave_ref = None
    flux_stack, err_stack, orig_flux_stack = [], [], []

    for date, filepath in fits_file_list:
        spec = extract_spectrum_from_fits(filepath)
        if spec:
            if wave_ref is None:
                wave_ref = spec['wave']
            # Interpolate to wave_ref grid
            spec_interp = interpolate_spectrum_to_wave(spec, wave_ref)
            flux_stack.append(spec_interp['flux'])
            err_stack.append(spec_interp['err'])
            orig_flux_stack.append(spec_interp['orig_flux'])

    if not flux_stack:
        print("No spectra loaded.")
        return None

    return {
        'wave': wave_ref,
        'flux': np.stack(flux_stack, axis=0),
        'err': np.stack(err_stack, axis=0),
        'orig_flux': np.stack(orig_flux_stack, axis=0)
    }