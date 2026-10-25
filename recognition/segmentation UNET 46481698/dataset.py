from pathlib import Path
import nibabel as nib
import math
from typing import Dict, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class Sample(NamedTuple):
    patient_id: int
    week: int
    slice: int
    image_path: Path
    mask_path: Path

def _parse_filename(filename: str):

    for suffix in (".nii.gz", ".nii"):
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)]
        else:
            continue

        split_name = stem.split("_")
        kind, patient_id, _, week, _, slice = split_name

    return kind, int(patient_id), int(week), int(slice)


print(_parse_filename("case_036_week_0_slice_2.nii.gz"))

