from pathlib import Path
from pprint import pprint
from typing import Sequence

import dicom2nifti
import nibabel
import numpy as np
import pydicom
import torch
import torchio
from matplotlib import pyplot as plt
from torchio import ScalarImage, Subject, LabelMap
from unet import UNet, UNet3D

from dataset_3d import convert_to_one_hot
from preprocessing import zscore_image, normalize_image
from run import find_precontrast_volume, preprocess_image_dicoms


def process_subject(subject_id, subjects_data_dir_path, results_dir_base_path, mappings_filepath):
    precontrast_dicom_dir_path = find_precontrast_volume(
        subject_id=subject_id,
        tcia_data_dir=subjects_data_dir_path,
        mappings_filepath=csv_mappings_filepath,
    )
    (output_dir_path := results_dir_base_path / subject_id).mkdir(exist_ok=True)
    original_smri_nifti_filepath = dicom_series_to_nifti(
        dicom_series_dir_path=precontrast_dicom_dir_path,
        output_dir_path=output_dir_path,
    )
    original_smri = ScalarImage(original_smri_nifti_filepath)
    preprocessed_by_chris_code = chris_transforms(original_smri)
    downsize_to_training_size = torchio.Resize(target_shape=(144, 144, preprocessed_by_chris_code.shape[-1]))
    downsized_smri = downsize_to_training_size(preprocessed_by_chris_code)

    inferred_mask = infer_breast_mask(downsized_smri.tensor).cpu()
    upsize_to_original_size = torchio.Resize(target_shape=preprocessed_by_chris_code.shape[1:])

    binarized_mask = torchio.transforms.OneHot(num_classes=1)(inferred_mask)

    # binarized_mask = convert_to_one_hot(inferred_mask)

    # upsized_inferred_mask = upsize_to_original_size(binarized_mask)
    # upsized_inferred_mask = upsize_to_original_size(inferred_mask)
    # binarized_mask = convert_to_one_hot(upsized_inferred_mask)

    # compare_visually(
    #     original_smri=original_smri,
    #     preprocessed_subject=preprocessed_subject,
    #     preprocessed_by_chris_code=preprocessed_by_chris_code,
    #     slice_num=(slice_num := int(156 / 2)),
    # )
    subject_with_inferred_masks = Subject(
        subject_id=subject_id,
        preprocessed_smri=preprocessed_by_chris_code,
        reast_mask=LabelMap(tensor=binarized_mask),
    )
    subject_with_inferred_masks.plot()
    compare_visually(
        images=[binarized_mask],
        slice_num=(slice_num := int(156 / 2)),
        planes=["axial"],
    )
    ...


def infer_breast_mask(image):
    unet = UNet3D(
        in_channels=1,
        out_classes=1,
        num_encoding_blocks=3,
        padding=True,
        normalization='batch'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model_path = Path(__file__).parent / "trained_models" / "breast_model.pth"
    with torch.no_grad():
        unet = torch.nn.DataParallel(unet)
        unet.load_state_dict(torch.load(saved_model_path))
        unet.to(device)
        unet.eval()
        image = image[None, :].to(device, dtype=torch.float32)
        pred = unet(image)
        pred = torch.sigmoid(pred)
        pred = torch.squeeze(pred, dim=0)
    return pred


def chris_transforms(original_smri):
    chris_transforms = torchio.Compose([
        torchio.transforms.Lambda(normalize_image),
        torchio.transforms.Lambda(zscore_image),
    ])
    preprocessed_by_chris_code = torchio.ScalarImage(
        tensor=zscore_image(normalize_image(original_smri.numpy()))
    )
    return preprocessed_by_chris_code


def my_transforms(original_smri):
    preprocessing_transforms = torchio.Compose([
        torchio.transforms.RescaleIntensity(.001),
        torchio.transforms.Lambda(torch.sigmoid, types_to_apply=[torchio.INTENSITY]),
        # torchio.transforms.Clamp(out_min=0., out_max=1.),
        torchio.transforms.ZNormalization(),

    ])
    preprocessed_subject = preprocessing_transforms(original_smri)
    return preprocessed_subject


def dicom_series_to_nifti(dicom_series_dir_path, output_dir_path):
    filename_parts = dicom_series_dir_path.name.split("-")
    filename = "_".join([filename_parts[0].rsplit(".")[0], *filename_parts[1].split()])
    original_smri_nifti_filepath = output_dir_path / f"{filename}.nii.gz"
    # if not original_smri_nifti_filepath.exists():
    dicom2nifti.convert_directory(
        dicom_directory=dicom_series_dir_path,
        output_folder=output_dir_path,
        reorient=False,
    )
    return original_smri_nifti_filepath


def compare_visually(
        images: Sequence[ScalarImage],
        slice_num: int,
        planes=("axial", "coronal", "sagittal"),
):
    for plane_ in planes:
        for pixels in images:
            if plane_ == "axial":
                plt.imshow(pixels.numpy().squeeze()[:, :, slice_num], cmap='gray')
            if plane_ == "coronal":
                plt.imshow(pixels.numpy().squeeze()[:, slice_num, :], cmap='gray')
            if plane_ == "sagittal":
                plt.imshow(pixels.numpy().squeeze()[slice_num, :, :], cmap='gray')
            plt.show()


if __name__ == '__main__':
    data_dir_base_path = Path("D:\projects-data\mazurowski\manifest-1654812109500")
    subjects_data_dir_path = data_dir_base_path / "Duke-Breast-Cancer-MRI"

    results_dir_base_path = data_dir_base_path / "quick_run_results"
    results_dir_base_path.mkdir(parents=True, exist_ok=True)
    csv_mappings_filepath = data_dir_base_path / "Breast-Cancer-MRI-filepath_filename-mapping.csv"

    subject_id = "Breast_MRI_025"
    process_subject(
        subject_id=subject_id,
        subjects_data_dir_path=subjects_data_dir_path,
        results_dir_base_path=results_dir_base_path,
        mappings_filepath=csv_mappings_filepath,
    )

...
