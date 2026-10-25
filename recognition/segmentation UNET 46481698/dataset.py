from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import math
from typing import Dict, NamedTuple, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

"""
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
"""

def to_channels (arr: torch.tensor, dtype = torch.uint8) -> torch.tensor:
    channels = torch.unique(arr)
    res = torch.zeros(arr.shape + (len(channels),), dtype = dtype)
    for c in channels:
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage = False, catagorical = False,
                 dtype = torch.float32, getAffines = False, early_stop = False):
    affines = []
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if(len(first_case.shape) == 3):
        first_case = first_case[:, :, 0]

    if catagorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = torch.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = torch.zeros((num, rows, cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged')
        inImage = torch.tensor(inImage)
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.to(dtype)
        if normImage:
            inImage = (inImage - inImage.min()) / (inImage.std() - inImage.min())
        if catagorical:
            inImage = to_channels(inImage, dtype = dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break
    if getAffines:
        return images, affines
    else:
        return images

def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    orient=False,
    early_stop=False,
):
    """Load medical image data from names, cases list provided into a list for

    each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive
    memory usage.

    normImage : bool (normalise the image 0.0-1.0)
    orient : Apply orientation and resample image? Good for images with large
    slice thickness or anisotropic resolution
    dtype : Type of the data. If dtype = np.uint8, it is assumed that the data
    is labels
    early_stop : Stop loading pre-maturely? Leaves arrays mostly empty, for
    quick loading and testing scripts.
    """
    affines = []

    interp = "linear"
    if dtype == np.uint8:  # assume labels
        interp = "nearest"

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[
                i,
                : inImage.shape[0],
                : inImage.shape[1],
                : inImage.shape[2],
                :,
            ] = inImage
        else:
            images[
                i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]
            ] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images