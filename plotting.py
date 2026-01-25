import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot(wav: np.ndarray, intensity: np.ndarray, title: str = "Spectrum") -> None:
    # Sort data by wavelength if not already sorted
    sort_idx = np.argsort(wav)
    wav_sorted = wav[sort_idx]
    intensity_sorted = intensity[sort_idx]

    # Create line segments as a list of arrays
    segments = [np.array([[wav_sorted[i], intensity_sorted[i]], [wav_sorted[i+1], intensity_sorted[i+1]]]) for i in range(len(wav_sorted)-1)]

    # Normalize for colormap
    norm = Normalize(wav_sorted.min(), wav_sorted.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(wav_sorted[:-1])

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()
    ax.invert_xaxis()  # Reverse x direction
    plt.title(title)    
    plt.xscale('log')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True)
    # Remove legend as there are no labeled elements
    # plt.legend()
    # Set serif font and ticks inside
    plt.rcParams['font.family'] = 'serif'
    plt.tick_params(direction='in')
    plt.show()