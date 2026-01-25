from simLIBS import SimulatedLIBS
import numpy as np
from configs import LOW_W_nm, UPPER_W_nm, RES_nm, Te_eV, Ne_cm3, T_eff_K

# libs = SimulatedLIBS(Te=1.0,
#                      Ne=10**17,
#                      elements=["Cu"],
#                      percentages=[100],
#                      resolution=1000,
#                      low_w=200,
#                      upper_w=1000,
#                      max_ion_charge=3,
#                      webscraping='static')

def clean_lines_single(element: str) -> tuple[np.ndarray, np.ndarray]:

    libs = SimulatedLIBS(
        Te=Te_eV,
        Ne=Ne_cm3,
        elements=[element],
        percentages=[100],
        resolution=int(np.ceil((UPPER_W_nm - LOW_W_nm) / RES_nm)),
        low_w=LOW_W_nm,
        upper_w=UPPER_W_nm,
        max_ion_charge=3,
        webscraping='static'
    )

    spec = libs.get_raw_spectrum()
    wav = np.array(spec["wavelength"], dtype=np.float64)
    intensity = np.array(spec["intensity"], dtype=np.float64)

    return wav, intensity

def clean_lines_composition(elements: list[str], fracs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    specs = [clean_lines_single(element) for element in elements]

    # Create a common wavelength grid
    common_wav = np.arange(LOW_W_nm, UPPER_W_nm + RES_nm, RES_nm)

    intensities = []
    for spec in specs:
        wav, intensity = spec
        # Interpolate intensity to the common wavelength grid
        interp_intensity = np.interp(common_wav, wav, intensity, left=0, right=0)
        intensities.append(interp_intensity)

    # Normalize each intensity to its maximum
    normalized_intensities = []
    for intensity in intensities:
        max_int = np.max(intensity)
        if max_int > 0:
            normalized_intensities.append(intensity / max_int)
        else:
            normalized_intensities.append(np.zeros_like(intensity))

    # Multiply by fractions and sum
    total_intensity = np.zeros_like(normalized_intensities[0])
    for norm_int, frac in zip(normalized_intensities, fracs):
        total_intensity += norm_int * frac

    return common_wav, total_intensity


def add_continuum(wav: np.ndarray, intensity: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Add a Bremsstrahlung continuum to the spectrum.

    scale: decimal of the maximum intensity to set as continuum level
    """
    if scale <= 0:
        return wav, intensity

    wav = np.asarray(wav, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    # Temperature: prefer electron temperature; fall back to effective temperature
    if Te_eV is not None and Te_eV > 0:
        T_K = Te_eV * 11604.518  # eV -> K
    elif T_eff_K is not None and T_eff_K > 0:
        T_K = float(T_eff_K)
    else:
        T_K = 10000.0

    ne = float(Ne_cm3) if Ne_cm3 is not None and Ne_cm3 > 0 else 1e16
    ni = ne  # assume singly ionized, quasi-neutral plasma
    Z_eff = 1.0

    h = 6.62607015e-34
    c = 299792458.0
    k_B = 1.380649e-23

    lam_m = np.clip(wav * 1e-9, 1e-20, None)  # nm -> m, avoid zero
    nu = c / lam_m

    # Non-relativistic Gaunt factor (smooth, positive approximation)
    gamma = (2.0 * k_B * T_K) / (h * nu)
    g_ff = (np.sqrt(3.0) / np.pi) * np.log1p(gamma)
    g_ff = np.clip(g_ff, 0.2, 5.0)

    # Thermal bremsstrahlung emissivity shape (free-free)
    cont = (
        (Z_eff ** 2) * ne * ni *
        (T_K ** -0.5) *
        g_ff *
        np.exp(-(h * c) / (lam_m * k_B * T_K)) /
        (lam_m ** 2)
    )

    cont_max = np.nanmax(cont)
    max_intensity = np.nanmax(intensity)
    if not np.isfinite(cont_max) or cont_max <= 0 or not np.isfinite(max_intensity) or max_intensity <= 0:
        return wav, intensity

    cont = cont / cont_max * (max_intensity * scale)

    return wav, intensity + cont
