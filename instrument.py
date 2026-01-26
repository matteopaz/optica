import numpy as np


def vert_noise(wav: np.ndarray, intensity: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply vertical (shot) noise to intensity using a Poisson process.
    """

    u = 1000 / level

    wav = np.asarray(wav, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    if u <= 0:
        return wav, intensity

    intensity_noisy = intensity.copy()
    finite_mask = np.isfinite(intensity)
    if np.any(finite_mask):
        lam = np.clip(intensity[finite_mask] * u, 0.0, None)
        noisy_counts = np.random.poisson(lam=lam)
        intensity_noisy[finite_mask] = noisy_counts / u

    return wav, intensity_noisy

def spreading_noise(wav: np.ndarray, intensity: np.ndarray, fwhm_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Spreading noise over given spectrum.
    Each point is averaged with its neighbors weighted by a Gaussian kernel.
    """

    res = wav[1] - wav[0]  # uniform spacing is true
    
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma

    window_size = int(6 * sigma / res)  # cover +/- 3 sigma
    if window_size % 2 == 0:
        window_size += 1  # make it odd

    intensity_noisy = np.zeros_like(intensity)

    kernel = np.exp(-0.5 * ((np.arange(window_size) - window_size // 2) * res / sigma) ** 2)
    kernel /= np.sum(kernel)  # normalize

    for i in range(len(intensity)):
        start = max(0, i - window_size // 2)
        end = min(len(intensity), i + window_size // 2 + 1)

        k_start = window_size // 2 - (i - start)
        k_end = window_size // 2 + (end - i)

        intensity_noisy[i] = np.sum(intensity[start:end] * kernel[k_start:k_end])

    return wav, intensity_noisy