import pathlib
import re
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