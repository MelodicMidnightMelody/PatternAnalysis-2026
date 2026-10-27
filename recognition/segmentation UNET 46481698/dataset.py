from pathlib import Path
import nibabel as nib
import numpy as np
from typing import NamedTuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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

def get_label_ids(sorted_samples):

    all_label_ids = set()

    for sample in sorted_samples:
        mask_path = sample.mask_path
        nifti_mask = nib.load(mask_path).get_fdata(caching = 'unchanged')
        if nifti_mask.ndim == 3:
            nifti_mask = nifti_mask[:, :, 0]
        #nifti_mask = nifti_mask.astype(np.uint8)
        for id in np.unique(nifti_mask):
            all_label_ids.add(int(id))

    label_ids = sorted(list(all_label_ids))
    id_to_channels =  {}
    for channel, id in enumerate(label_ids):
        id_to_channels[id] = channel
    num_classes = len(label_ids)
    return id_to_channels, num_classes

class LoadDataset(Dataset):
    """
    Loads 2D Nifti image slices and segmentation masks from HipMRI 2D Dataset


    """
    def __init__(self, samples, num_classes, id_channels, size=(256, 256), normalized=True):

        self.samples = samples
        self.num_classes = num_classes
        self.id_channels = id_channels
        self.size = size
        self.normalized = normalized

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx].image_path
        mask_path = self.samples[idx].mask_path

        #loading image and mask slices
        loaded_np_image = nib.load(image_path).get_fdata(caching='unchanged')
        loaded_np_mask = nib.load(mask_path).get_fdata(caching='unchanged')

        if loaded_np_image.ndim == 3:
            loaded_np_image = loaded_np_image[:, :, 0]

        if loaded_np_mask.ndim == 3:
            loaded_np_mask = loaded_np_mask[:, :, 0]

        loaded_np_image = loaded_np_image.astype("float32")
        loaded_np_mask = loaded_np_mask.astype("uint8")

        #z-score normalisation
        if self.normalized:
            mean = loaded_np_image.mean()
            std = loaded_np_image.std()
            loaded_np_image = (loaded_np_image - mean) / (std + 1e-8)

        h, w = loaded_np_mask.shape

        #One hot encoding conversion
        mask_encoded = np.zeros(self.num_classes, h, w, dtype="float32")
        for label_id, channel in self.id_channels.items():
            mask_encoded[channel] = (loaded_np_mask == label_id).astype("float32")

        loaded_torch_image = torch.from_numpy(loaded_np_image).unsqueeze(0)
        loaded_torch_mask = torch.from_numpy(mask_encoded)

        #resize and fix dimensions
        loaded_torch_image = F.interpolate(loaded_torch_image.unsqueeze(0), size=self.size,
                                           mode="bilinear", align_corners=False).squeeze(0)

        loaded_torch_mask = F.interpolate(loaded_torch_mask.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)


def get_dataloaders(root_path = Path("./Data/keras_slices_data"), num_workers = 1, batch_size = 32, out_size = (256, 256)):

    root_path = Path(root_path)
    train_img_path = root_path / "keras_slices_train"
    val_img_path = root_path / "keras_slices_validate"
    test_img_path = root_path / "keras_slices_test"
    train_mask_path = root_path / "keras_slices_seg_train"
    val_mask_path = root_path / "keras_slices_seg_validate"
    test_mask_path = root_path / "keras_slices_seg_test"

    train_samples = _get_pairs(train_img_path, train_mask_path)
    val_samples = _get_pairs(val_img_path, val_mask_path)
    test_samples = _get_pairs(test_img_path, test_mask_path)

    print(len(train_samples), len(val_samples), len(test_samples))

    id_channels, num_classes = get_label_ids(train_samples)
    print(num_classes)

    train_dataset = LoadDataset(train_samples, num_classes, id_channels, size=(256, 256))
    val_dataset = LoadDataset(val_samples, num_classes, id_channels, size=(256, 256))
    test_dataset = LoadDataset(test_samples, num_classes, id_channels, size=(256, 256))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader, id_channels, num_classes

#get_dataloaders()