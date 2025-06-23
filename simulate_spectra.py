from scipy.interpolate import interp1d
from astropy.io import fits 

def tellurize_star(wave_file, flux_file, telluric_file):
    """
    Function to apply telluric correction to a stellar spectrum using standard PHOENIX stellar and ESO telluric spectra.
    """

    # Load the wavelength array of stellar spectrum
    with fits.open(wave_file) as hdu:
        wave = hdu[0].data

    # Load the stellar spectrum
    with fits.open(flux_file) as hdu:
        flux = hdu[0].data

    # Load the telluric spectrum
    with fits.open(telluric_file) as hdu:
        telluric = hdu[1].data

    telluric_wavelength_angstrom = telluric['lam'] * 1e4

    # Interpolate telluric transmission onto the stellar wavelength grid
    telluric_interp = interp1d(telluric_wavelength_angstrom, telluric['trans'], bounds_error=False, fill_value=1.0)
    telluric_on_stellar = telluric_interp(wave)

    # Multiply the stellar flux by the telluric transmission
    combined_flux = flux * telluric_on_stellar

    return wave, combined_flux, telluric_on_stellar