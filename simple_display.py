from spec import *
from instrument import *
from plotting import plot

wav, intensity = clean_lines_composition(["Cu", "O"], [0.5, 0.5]) # O, C , H are main contaminants
wav, intensity = add_continuum(wav, intensity)
wav, intensity = vert_noise(wav, intensity, level=1.5) # level probably between 0.5 and 5 logarithmically aranged
wav, intensity = spreading_noise(wav, intensity, fwhm_nm=1.5) # fwhm_nm between 0 and 2.5 aranged linearly
plot(wav, intensity, title="Cu Spectrum with Continuum") 