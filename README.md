# Overview

Three different types of parameters:

## Physical Parameters

Parameters relating to the actual physical setup of the conveyor and analysis system. Consists of:

- Spectrogram range and resolution (`LOW_W_nm, UPPER_W_nm, RES_nm`) - self explanatory
- Pulse energy (`PULSE_ENERGY_mJ`) - energy deposited during pulse
- Pulse duration (`PULSE_DURATION_ns`) (set to 7) - in nanoseconds, FWHM of pulse. upper bounded by 7 for the laser.
- Spot size (`SPOTSIZE_INPUT_mm2`) - area in square mm of spotsize, modulated by angle on target
- Angle on target (`ANGLE_ON`) - positive angle in degrees of the angle of incidence between analysis shot and metal surface. 
- Pulse delay e.g. gating (`DELAY_us`) - very important, delay before start of exposure after analysis shot in microseconds.

## Compositional Parameters
Simple fractional merging of elemental spectra to simulate oxidation.
- `elements` & `pcts` - linear combination pulled from NIST with wrapper library.

## Error Parameters
Two primary sources of noise, I labeled vert and spread noise:

- Vert (shot) noise (`level`) - Poisson noise above pure signal for count value. inversely controlled by level param (level 1 = lambda 1000x).
- Spread noise (`fwhm_nm`) - simple gaussian kernel smearing controlled by FWHM to simulate realistic spectrogram window.

## Derived Parameters

After the controlled physical parameters are input, other parameters can be calculated which occur in many physical equations. 

- I (`IRRADIANCE_W_cm2`) - Laser irradiance derived from physical params, in wattage per sq centimeter. 
- T_e (`Te_eV`) - Electron temperature in electron volts. Primarily dependent on irradiance and gating.
- T_eff (`T_eff_K`) - Inthe case of local thermodynamic equilibrium (LTE) which is assumed, equal to electron temp but instead in Kelvin.
- N_e (`Ne_cm3`) - In inverse cubic centimeters, the electron density of the plasma. Dependent primarily on irradiance and gating.

# Output

Across many sensible ranges of the above controllable parameters, synthetic simulated spectra are generated in the cartesian product and placed into a filesystem.
Simple plotting of relative detector counts versus wavelength on a logscale.