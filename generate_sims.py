import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import configs
from spec import clean_lines_single, add_continuum
from instrument import vert_noise, spreading_noise


PULSE_ENERGIES_MJ = [50, 100]
ANGLE_ON_DEG = 0.0
SPOTSIZE_MM2 = [0.15, 0.3, 0.5]
DELAYS_US = [0.5, 1, 2, 5]

ELEMENTS = ["Cu", "Al", "Zn", "Mg", "O"]

COMPOSITIONS = OrderedDict(
    [
        ("cu", (["Cu"], [1.0])),
        ("al", (["Al"], [1.0])),
        ("zn", (["Zn"], [1.0])),
        ("mg", (["Mg"], [1.0])),
        ("o", (["O"], [1.0])),
        ("cu_ox", (["Cu", "O"], [0.66, 0.33])),
        ("cu_bulk", (["Cu", "O"], [0.90, 0.10])),
        ("al_ox", (["Al", "O"], [0.40, 0.60])),
        ("al_bulk", (["Al", "O"], [0.90, 0.10])),
        ("brass_ox", (["Cu", "Zn", "O"], [0.35, 0.35, 0.30])),
        ("brass_bulk", (["Cu", "Zn", "O"], [0.45, 0.45, 0.10])),
        ("mg_ox", (["Mg", "O"], [0.50, 0.50])),
        ("mg_bulk", (["Mg", "O"], [0.90, 0.10])),
    ]
)

VERT_NOISE_LEVELS = [0.5, 1.5, 3]
SPREAD_FWHM_NM = [0.5, 1.5, 2.5]


def _fmt(value: float) -> str:
    return f"{value:g}"


def _common_wavelength_grid() -> np.ndarray:
    return np.arange(
        configs.LOW_W_nm,
        configs.UPPER_W_nm + configs.RES_nm,
        configs.RES_nm,
        dtype=np.float64,
    )


def _normalize(intensity: np.ndarray) -> np.ndarray:
    max_val = np.nanmax(intensity)
    if not np.isfinite(max_val) or max_val <= 0:
        return np.zeros_like(intensity)
    return intensity / max_val


def _interpolate_sorted(wav: np.ndarray, intensity: np.ndarray, common_wav: np.ndarray) -> np.ndarray:
    order = np.argsort(wav)
    wav_sorted = wav[order]
    intensity_sorted = intensity[order]
    return np.interp(common_wav, wav_sorted, intensity_sorted, left=0.0, right=0.0)


def _prepare_element_spectra(common_wav: np.ndarray) -> dict[str, np.ndarray]:
    element_specs: dict[str, np.ndarray] = {}
    for element in ELEMENTS:
        wav, intensity = clean_lines_single(element)
        wav, intensity = add_continuum(wav, intensity)
        interp = _interpolate_sorted(wav, intensity, common_wav)
        interp = np.clip(interp, 0.0, None)
        element_specs[element] = _normalize(interp)
    return element_specs


def _combine(elements: list[str], fracs: list[float], element_specs: dict[str, np.ndarray]) -> np.ndarray:
    total = np.zeros_like(next(iter(element_specs.values())))
    for element, frac in zip(elements, fracs):
        total += element_specs[element] * float(frac)
    return total


def _plot_and_save(
    wav: np.ndarray,
    intensity: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    try:
        ax.plot(wav, intensity, color="black", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_xlim(wav[0], wav[-1])
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
        fig.savefig(tmp_path, dpi=150)
        tmp_path.replace(path)
    finally:
        plt.close(fig)


def _is_done(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _expected_paths(comp_dir: Path) -> list[Path]:
    return [
        comp_dir / f"vert_{_fmt(vert_level)}_spread_{_fmt(spread_fwhm)}.png"
        for vert_level in VERT_NOISE_LEVELS
        for spread_fwhm in SPREAD_FWHM_NM
    ]


def _comp_complete(comp_dir: Path) -> bool:
    return all(_is_done(path) for path in _expected_paths(comp_dir))


def _combo_complete(combo_root: Path) -> bool:
    for comp_name in COMPOSITIONS:
        if not _comp_complete(combo_root / comp_name):
            return False
    return True


def generate(output_dir: Path, seed: int | None = None) -> None:
    if seed is not None:
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    common_wav = _common_wavelength_grid()

    total_params = len(PULSE_ENERGIES_MJ) * len(DELAYS_US) * len(SPOTSIZE_MM2)
    combo_index = 0

    for energy_mj in PULSE_ENERGIES_MJ:
        for delay_us in DELAYS_US:
            for spotsize_mm2 in SPOTSIZE_MM2:
                combo_index += 1
                print(
                    f"[{combo_index}/{total_params}] energy={energy_mj} mJ, "
                    f"delay={delay_us} us, spotsize={spotsize_mm2} mm2"
                )

                combo_root = (
                    output_dir
                    / f"energy_{_fmt(energy_mj)}mJ"
                    / f"delay_{_fmt(delay_us)}us"
                    / f"spotsize_{_fmt(spotsize_mm2)}mm2"
                )

                if _combo_complete(combo_root):
                    print("  -> already generated, skipping.")
                    continue

                configs.set_physical_params(
                    pulse_energy_mj=energy_mj,
                    delay_us=delay_us,
                    spotsize_mm2=spotsize_mm2,
                    angle_on_deg=ANGLE_ON_DEG,
                )

                try:
                    element_specs = _prepare_element_spectra(common_wav)
                except Exception as exc:
                    print(f"  -> failed to prepare element spectra: {exc}")
                    continue

                for comp_name, (elements, fracs) in COMPOSITIONS.items():
                    comp_dir = combo_root / comp_name
                    if _comp_complete(comp_dir):
                        continue

                    comp_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        base_intensity = _combine(elements, fracs, element_specs)
                    except Exception as exc:
                        print(f"  -> failed to combine {comp_name}: {exc}")
                        continue

                    for vert_level in VERT_NOISE_LEVELS:
                        for spread_fwhm in SPREAD_FWHM_NM:
                            out_path = (
                                comp_dir
                                / f"vert_{_fmt(vert_level)}_spread_{_fmt(spread_fwhm)}.png"
                            )
                            if _is_done(out_path):
                                continue
                            wav_noisy, intensity_noisy = vert_noise(
                                common_wav, base_intensity, level=vert_level
                            )
                            wav_noisy, intensity_noisy = spreading_noise(
                                wav_noisy, intensity_noisy, fwhm_nm=spread_fwhm
                            )
                            title = (
                                f"{comp_name} | vert {vert_level}, spread {spread_fwhm} nm"
                            )
                            try:
                                _plot_and_save(
                                    wav_noisy,
                                    intensity_noisy,
                                    title,
                                    out_path,
                                )
                            except Exception as exc:
                                print(f"  -> failed to save {out_path.name}: {exc}")
                                continue


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated LIBS spectra.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_sims"),
        help="Directory to write spectra to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for noise reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.output_dir, seed=args.seed)
