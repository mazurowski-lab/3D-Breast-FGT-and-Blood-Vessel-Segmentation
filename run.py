import os
from pathlib import Path

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


def preprocess_and_save_image_volume(subject_id, tcia_data_dir, mappings_filepath=None, segmentation_dir=None):
    if mappings_filepath:
        fpath_mapping_df = clean_filepath_filename_mapping_csv(str(mappings_filepath))
    else:
        fpath_mapping_df = None

    tcia_data_dir = Path(tcia_data_dir)

    # Get the sequence dir from the DataFrame

    # There's also a subdir for every subject that contains the sequences
    # There is only one of these
    sub_dir = os.listdir(tcia_data_dir / subject_id)[0]
    preprocessed_array_base_save_path = Path(f'{tcia_data_dir.parent / "preprocessed" / subject_id}')

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

    volumes_npy_paths = []
    for full_sequence_dir_ in full_sequence_dirs:
        if mappings_filepath:
            npy_save_path = Path(f'{preprocessed_array_base_save_path}.npy')
        else:
            npy_save_path = Path(f'{preprocessed_array_base_save_path / full_sequence_dir_.name}.npy')
        npy_save_path.parent.mkdir(exist_ok=True, parents=True)
        if npy_save_path.exists():
            volumes_npy_paths.append(npy_save_path)
            continue
        image_array, dcm_data, volume_path = preprocess_image_dicoms(full_sequence_dir_)
        # image_array, dicom_data, nrrd_breast_data, nrrd_dv_data = preprocess_image_segmentations(
        #     subject_id=subject_id,
        #     tcia_data_dir=tcia_data_dir,
        #     fpath_mapping_df=fpath_mapping_df,
        #     segmentation_dir=segmentation_dir,
        # )
        np.save(npy_save_path, image_array)
        volumes_npy_paths.append(npy_save_path)
    return preprocessed_array_base_save_path.parent, volumes_npy_paths


def preprocess_image_dicoms(full_sequence_dir):
    image_array, dcm_data, volume_path = read_precontrast_mri(full_sequence_dir)
    image_array = zscore_image(normalize_image(image_array))
    return image_array, dcm_data, volume_path


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
    base_path = Path(r"D:\projects-data\mazurowski\manifest-1654812109500")
    xlsx_mappings_filepath = base_path / "Breast-Cancer-MRI-filepath_filename-mapping.xlsx"
    csv_mappings_filepath = base_path / "Breast-Cancer-MRI-filepath_filename-mapping.csv"
    tcia_data_dir = base_path / "Duke-Breast-Cancer-MRI"

    subject_id = 'Breast_MRI_025'

    subject_dirpath = tcia_data_dir / subject_id
    volumes_dirpath = [item for item in subject_dirpath.iterdir() if item.is_dir()][0]
    breast_mask_save_path = base_path / "breast_masks"
    breast_mask_save_path.mkdir(exist_ok=True, parents=True)
    dv_masks_save_path = base_path / "fgt_bv_masks"
    dv_masks_save_path.mkdir(exist_ok=True, parents=True)

    preprocessed_images_dirpath, volumes_npy_paths = preprocess_and_save_image_volume(
        subject_id,
        tcia_data_dir,
        mappings_filepath=csv_mappings_filepath,
        segmentation_dir=breast_mask_save_path,
    )
    # if not (breast_mask_save_path / f"{subject_id}.npy").exists():
    run(
        target_tissue="breast",
        image_dir=str(preprocessed_images_dirpath),
        input_mask_dir=str(breast_mask_save_path),
        save_masks_dir=str(breast_mask_save_path),
    )
# if not (dv_masks_save_path / f"{subject_id}.npy").exists():
    run(
        target_tissue="dv",
        image_dir=str(preprocessed_images_dirpath),
        input_mask_dir=str(breast_mask_save_path),
        save_masks_dir=str(dv_masks_save_path),
    )
    display(subject_id)

