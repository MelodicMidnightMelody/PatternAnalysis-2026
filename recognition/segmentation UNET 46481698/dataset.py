from pathlib import Path
import nibabel as nib
import numpy as np
from setuptools.namespaces import flatten
from tqdm import tqdm
import math
from typing import Dict, NamedTuple, Sequence
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

def _get_pairs(image_dir: Path, mask_dir: Path):
    """
    Store a pair of image, mask and other variables as a Sample(Named Tuple)
    :param image_dir: Path to the directory containing images
    :param mask_dir: Path to the directory containing segmentation masks
    :return: A sorted list of Sample objects
    """
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

def get_mask_labels(sorted_samples):

    all_label_ids = set()

    for sample in sorted_samples:
        mask_path = sample.mask_path
        nifti_mask = nib.load(mask_path).get_fdata(caching = 'unchanged')
        if nifti_mask.ndim == 3:
            nifti_mask = nifti_mask[:, :, 0]
        print(nifti_mask)
        #nifti_mask = nifti_mask.astype(np.uint8)
        print(nifti_mask)
        for id in np.unique(nifti_mask):
            all_label_ids.add(int(id))

    label_ids = sorted(list(all_label_ids))
    label_channels =  {}
    for i, label in enumerate(label_ids):
        label_channels[label] = i
    num_classes = len(label_ids)
    return label_ids, label_channels, num_classes

samples = _get_pairs(Path("./Data/keras_slices_data/keras_slices_validate"), Path("./Data/keras_slices_data/keras_slices_seg_validate"))
label_ids, label_channels, num_classes = get_mask_labels(samples)

print(label_ids)
print(label_channels)
print(num_classes)

