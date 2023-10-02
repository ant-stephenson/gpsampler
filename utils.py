import pathlib
import re
import numpy as np
from functools import singledispatch
from typing import Tuple


@singledispatch
def check_exists(path: pathlib.Path, expt_no=None, overwrite=False, suffix=".csv") -> Tuple[pathlib.Path, int]:
    if path.with_suffix(suffix).exists() and not overwrite:
        file_name = path.stem
        prev_no = re.findall("\d$", str(path))
        if prev_no:
            expt_no = int(prev_no[0]) + 1
            path = path.with_name(f"{file_name[:-len(prev_no)]}{expt_no}")
        else:
            expt_no = 1
            path = path.with_name(f"{file_name}_{expt_no}")
        return check_exists(path, expt_no, overwrite, suffix)
    else:
        return path.with_suffix(suffix), expt_no


@check_exists.register
def _(path: str, expt_no=None, overwrite=False, suffix=".csv") -> Tuple[pathlib.Path, int]:
    _path = pathlib.Path(path)
    return check_exists(_path, expt_no, overwrite, suffix)


def rescale_dataset_noise(data: np.ndarray, new_nv: float, rng: np.random.Generator) -> np.ndarray:
    """ Takes in a numpy array of X-y data of shape N x d+1 where the last
    column is a vector of observations (y) and the first d columns are the input
    X data. 

    This function takes the final y vector and scales the noise variance to
    whatever is desired as the input, simultaneously rescaling the output scale
    variance (so the two sum to 1) and outputs a modified numpy array where the
    final column has been replaced with the transformed version. 

    Args:
        data (np.ndarray): Nxd+1 array of X-y data
        new_nv (float): desired noise variance
        rng (np.random.Generator): in-use random generator

    Returns:
        np.ndarray: array with same X data and transformed y data
    """
    n = data.shape[0]
    orig_nv = 0.008
    orig_ks = 1-orig_nv
    eta = (new_nv-orig_nv)/(1-orig_nv)
    xi = rng.standard_normal((n,)) * np.sqrt(np.abs(eta))
    y1 = np.sqrt(1-eta) * data[:, -1] + np.sign(eta) * xi
    data[:, -1] = y1
    return data
