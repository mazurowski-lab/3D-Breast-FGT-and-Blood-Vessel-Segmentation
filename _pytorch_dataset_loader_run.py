import datetime
import os
from pathlib import Path

import nibabel
from matplotlib import pyplot as plt
import numpy as np
import pydicom

from predict import run
from preprocessing import (
    clean_filepath_filename_mapping_csv,
    normalize_image,
    read_precontrast_mri,
    zscore_image, read_precontrast_mri_and_segmentation,
)


def preprocess_and_save_image_volume(subject_id, tcia_data_dir, mappings_filepath=None):
    tcia_data_dir = Path(tcia_data_dir)
    full_sequence_dirs = get_full_sequence_dirpaths(subject_id, tcia_data_dir, mappings_filepath)

    base_save_path = (
        tcia_data_dir.parent
        / f"inference_results_{(timestamp := datetime.datetime.utcnow().strftime('%Y-%m-%d__%Hh%Mm%Ss'))}"
    )
    subject_path = base_save_path / subject_id
    preprocessed_save_path = subject_path / "preprocessed"

    volumes_npy_paths = []
    for full_sequence_dir_ in full_sequence_dirs:
        npy_save_path, nifti_save_path = make_save_paths(
            full_sequence_dir_=full_sequence_dir_,
            save_path=preprocessed_save_path,
            mappings_filepath=mappings_filepath,
            )
        npy_save_path.parent.mkdir(exist_ok=True, parents=True)
        if nifti_save_path.exists() and npy_save_path.exists():
            volumes_npy_paths.append(npy_save_path)
            continue
        image_array, dcm_data, volume_path = preprocess_image_dicoms(full_sequence_dir_)
        save_images(image_array, npy_save_path, nifti_save_path)
        volumes_npy_paths.append(npy_save_path)
    return subject_path, volumes_npy_paths


def get_full_sequence_dirpaths(subject_id, tcia_data_dir, mappings_filepath):
    if mappings_filepath:
        fpath_mapping_df = clean_filepath_filename_mapping_csv(str(mappings_filepath))
    else:
        fpath_mapping_df = None

        # Get the sequence dir from the DataFrame

        # There's also a subdir for every subject that contains the sequences
        # There is only one of these
    sub_dir = os.listdir(tcia_data_dir / subject_id)[0]
    if fpath_mapping_df is not None:
        sequence_dir = fpath_mapping_df.loc[
            fpath_mapping_df['subject_id'] == subject_id, 'precontrast_dir'
        ].iloc[0]
        full_sequence_dirs = [tcia_data_dir / subject_id / sub_dir / sequence_dir]
    else:
        full_sequence_dirs = sorted(
            [
                item
                for item in (tcia_data_dir / subject_id / sub_dir).iterdir()
                if item.is_dir()
                ]
            )
    return full_sequence_dirs


def make_save_paths(full_sequence_dir_, save_path, mappings_filepath):
    if mappings_filepath:
        npy_save_path = save_path.with_suffix(".npy")
        nifti_save_path = save_path.with_suffix(".nii.gz")
    else:
        npy_save_path = (save_path / full_sequence_dir_.name).with_suffix(".npy")
        nifti_save_path = (save_path / full_sequence_dir_.name).with_suffix(".nii.gz")
    return npy_save_path, nifti_save_path


def preprocess_image_dicoms(full_sequence_dir):
    image_array, dcm_data, volume_path = read_precontrast_mri(full_sequence_dir)
    image_array = zscore_image(normalize_image(image_array))
    return image_array, dcm_data, volume_path


def save_images(image_array, nifti_save_path, npy_save_path):
    np.save(npy_save_path, image_array)
    nibabel.save(img=(nifti_image := nibabel.Nifti1Image(image_array, affine=np.eye(4))), filename=nifti_save_path)


def preprocess_image_segmentations(subject_id, tcia_data_dir, fpath_mapping_df, segmentation_dir):
    image_array, dicom_data, nrrd_breast_data, nrrd_dv_data = read_precontrast_mri_and_segmentation(
        subject_id,
        tcia_data_dir,
        fpath_mapping_df,
        segmentation_dir
    )
    return image_array, dicom_data, nrrd_breast_data, nrrd_dv_data


def visualize(image_array, nrrd_breast_data, nrrd_dv_data):
    plt.subplot(2, 3, 1)
    plt.title('MRI Volume')
    plt.imshow(image_array[:, :, 50], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Breast Mask')
    plt.imshow(nrrd_breast_data[:, :, 50], cmap='gray')
    plt.axis('off')
    # plt.show()
    # return
    plt.subplot(2, 3, 4)
    plt.title('FGT + Blood Vessel Mask')
    plt.imshow(nrrd_dv_data[0, :, :, 50], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('FGT + Blood Vessel Mask')
    plt.imshow(nrrd_dv_data[1, :, :, 50], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('FGT + Blood Vessel Mask')
    plt.imshow(nrrd_dv_data[2, :, :, 50], cmap='gray')
    plt.axis('off')

    plt.show()


def display(subject_id, base_path):
    image_array = np.load(Path(base_path / "preprocessed" / f"{subject_id}.npy"))
    nrrd_breast_data = np.load(Path(base_path / "breast_masks" / f"{subject_id}.npy"))
    nrrd_dv_data = np.load(Path(base_path / "fgt_bv_masks" / f"{subject_id}.npy"))
    visualize(image_array, nrrd_breast_data, nrrd_dv_data)


if __name__ == '__main__':

    base_path_desktop = Path(r"D:\projects-data\mazurowski\manifest-1654812109500")
    base_path = Path(r"C:\Users\kshit\Projects\projects-data\mazurowski\data\Duke-Breast-Cancer-MRI\manifest-1654812109500")
    xlsx_mappings_filepath = base_path / "Breast-Cancer-MRI-filepath_filename-mapping.xlsx"
    csv_mappings_filepath = base_path / "Breast-Cancer-MRI-filepath_filename-mapping.csv"
    tcia_data_dir = base_path / "Duke-Breast-Cancer-MRI"

    subject_id = 'Breast_MRI_012'

    subject_dirpath = tcia_data_dir / subject_id
    # volumes_dirpath = [item for item in subject_dirpath.iterdir() if item.is_dir()][0]
    # breast_mask_save_path = base_path / "breast_masks"
    # breast_mask_save_path.mkdir(exist_ok=True, parents=True)
    # dv_masks_save_path = base_path / "fgt_bv_masks"
    # dv_masks_save_path.mkdir(exist_ok=True, parents=True)

    subject_path, volumes_npy_paths = preprocess_and_save_image_volume(
        subject_id,
        tcia_data_dir,
        mappings_filepath=csv_mappings_filepath,
    )
    # if not (breast_mask_save_path / f"{subject_id}.npy").exists():
    run(
        target_tissue="breast",
        image_dir=str(subject_path / "preprocessed"),
        input_mask_dir=None,
        save_masks_dir=str(subject_path / "breast_mask"),
    )
# if not (dv_masks_save_path / f"{subject_id}.npy").exists():
    run(
        target_tissue="dv",
        image_dir=str(subject_path / "preprocessed"),
        input_mask_dir=str(subject_path / "breast_mask"),
        save_masks_dir=str(subject_path / "fgt_bv_mask"),
    )
    # display(subject_id)
