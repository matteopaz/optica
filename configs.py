import numpy as np

# SPECTROGRAM SPECS
LOW_W_nm = 200
UPPER_W_nm = 450
RES_nm = 0.33 

# LASER
PULSE_ENERGY_mJ = 100 # mJ 
PULSE_DURATION_ns = 7 # ns 
ANGLE_ON = 0 # should vary at max between 0 and 45 degrees (symmetrical about +-)

# OPTICS
SPOTSIZE_mm2 = 0.27 # mm2

# -->
SPOTSIZE_mm2 = SPOTSIZE_mm2 / np.cos(np.deg2rad(ANGLE_ON)) # effective spot size on target
SPOTSIZE_R_mm = ( (SPOTSIZE_mm2/3.1415)**0.5 ) # derived but fixed

# GATING
DELAY_us = 5 # 5 us

IRRADIANCE_W_cm2 = None
def calc_IRRADIANCE():
    global IRRADIANCE_W_cm2
    # Unit conversion: 
    # Energy: mJ -> J (1e-3)
    # Time: ns -> s (1e-9)
    # Area: mm^2 -> cm^2 (1e-2)
    # Factor = 1e-3 / (1e-9 * 1e-2) = 1e8
    # Formula: I = 2 * Energy / (Time * Area)  [Factor 2 for Gaussian peak]
    
    IRRADIANCE_W_cm2 = (2 * PULSE_ENERGY_mJ / (PULSE_DURATION_ns * SPOTSIZE_mm2)) * 1e8
    # Assuming a gaussian beam about spot center

Te_eV = None
Ne_cm3 = None
def calc_TE_NE():
    """
    Compute electron temperature Te (eV) and electron density Ne (cm^-3)
    from irradiance I (W/cm^2) and gate delay t (seconds).

    Semi-empirical LIBS surrogate model (ns-LIBS on solids, air/Ar).
    Uses:
      - saturating dependence on irradiance (plasma shielding / coupling saturation)
      - power-law decay with delay, with a small-time smoothing to avoid divergence

    Globals required:
      IRRADIANCE_W_cm2 : float  (W/cm^2)
      DELAY_us         : float  (us)

    Globals written:
      Te_eV            : float  (eV)
      Ne_cm3           : float  (cm^-3)
    """

    global IRRADIANCE_W_cm2
    global DELAY_us
    global Te_eV
    global Ne_cm3

    # ---- inputs (units) ----
    I = float(IRRADIANCE_W_cm2)      # W/cm^2
    t = float(DELAY_us) * 1e-6       # seconds

    if I <= 0.0:
        Te_eV = 0.8
        Ne_cm3 = 1e15
        return Te_eV, Ne_cm3

    if t <= 0.0:
        raise ValueError("DELAY_us must be > 0 (us).")

    # ---- model parameters (tuned for 'typical' gated LIBS spectra) ----
    Is = 1e10        # W/cm^2, irradiance scale where saturation begins
    p = 0.7          # saturation steepness

    t0 = 1e-6        # s, reference delay (1 microsecond)
    tc = 5e-8        # s, smoothing time (~50 ns) to avoid blow-up as t->0

    alpha = 0.6      # Te decay exponent
    beta  = 2.0      # Ne decay exponent

    Tmin, Tmax = 0.8, 2.2          # eV (at t0, after saturation factor)
    Nmin, Nmax = 1e15, 3e17        # cm^-3 (at t0, after saturation factor)

    x = I / Is
    S = (x**p) / (1.0 + x**p)      # in [0,1)

    time_factor = (t + tc) / (t0 + tc)

    Te = Tmin + (Tmax - Tmin) * S * (time_factor ** (-alpha))        # eV
    Ne = Nmin + (Nmax - Nmin) * S * (time_factor ** (-beta))         # cm^-3

    Te = min(max(Te, 0.5), 10.0)      # eV
    Ne = min(max(Ne, 1e14), 1e19)     # cm^-3

    Te_eV = Te
    Ne_cm3 = Ne
    return Te_eV, Ne_cm3

T_eff_K = None # assuming LTE, convert eV to K

# Master
def calc_derived():
    calc_IRRADIANCE()
    calc_TE_NE()
    global T_eff_K
    T_eff_K = Te_eV * 11604.518  # eV -> K

    
calc_derived()

print(f"Calculated irradiance: {IRRADIANCE_W_cm2:.2e} W/cm^2")
print(f"Calculated electron temperature Te: {Te_eV:.2f} eV")
print(f"Calculated electron density Ne: {Ne_cm3:.2e} cm^-3")
print(f"Calculated Teff at delay {DELAY_us} us: {T_eff_K:.2f} K")

# 1 microsecond gating