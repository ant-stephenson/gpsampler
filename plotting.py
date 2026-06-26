from matplotlib import pyplot as plt
from typing import Tuple
from pathlib import Path

from gpsampler.utils import check_exists


def _latex_available() -> bool:
    import shutil

    return shutil.which("latex") is not None


class LaTeX(object):
    def __init__(self, preamble=None):
        self.preamble = preamble
        self._usetex = False

    def __enter__(self, preamble=None):
        if _latex_available():
            self._usetex = True
            plt.rcParams["text.usetex"] = True
        else:
            import warnings

            warnings.warn(
                "LaTeX not found on PATH; falling back to matplotlib's default text renderer.",
                UserWarning,
                stacklevel=2,
            )

    def __exit__(self, type, value, traceback):
        plt.rcParams = plt.rcParamsDefault


def save_fig(
    base_path: Path,
    filename: str,
    suffix: str = "eps",
    show: bool = False,
    size_inches: Tuple[float, float] = None,
    overwrite=False,
    **kwargs,
):
    fgt = plt.gcf()
    if size_inches is not None:
        fgt.set_size_inches(*size_inches, forward=show)
    figpath = base_path.joinpath("figs", filename)
    if not overwrite:
        figpath = check_exists(figpath, suffix="." + suffix)[0]
    else:
        figpath = figpath.with_suffix("." + suffix)
    fgt.savefig(figpath, format=suffix, bbox_inches="tight", **kwargs)
    if not show:
        plt.close(fgt)

