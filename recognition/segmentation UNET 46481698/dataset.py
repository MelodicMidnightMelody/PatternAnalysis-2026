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

def _get_pairs(root, split):

    image_dir = Path(root) / f"keras_slices_{split}"
    mask_dir = Path(root) / f"keras_slices_seg_{split}"


    samples = []
    for image_path in image_dir.glob("*.nii.gz"):
        _, patient_id, _, week, _, slice_index = image_path.name.split(".")[0].split("_")
        mask_path = mask_dir / image_path.name.replace("case_", "seg_", 1)

        if not mask_path.exists():
            raise FileNotFoundError(f"No mask for {image_path.name}")
        samples.append(Sample(int(patient_id), int(week),
                              int(slice_index), image_path, mask_path))

    sorted_samples = sorted(samples, key=lambda s: (s.patient_id, s.week, s.slice))
    return sorted_samples


split = "validate"

pairs = _get_pairs("./Data/keras_slices_data", split)
print(f"{split}: {len(pairs)} pairs, {len({s.patient_id for s in pairs})} patients")
print("  first:", pairs[0].image_path.name, "<->", pairs[0].mask_path.name)
