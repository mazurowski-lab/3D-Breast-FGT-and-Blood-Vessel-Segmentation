import os
from pathlib import Path

import numpy as np
import pydicom

from predict import run
from preprocessing import (
    clean_filepath_filename_mapping_csv,
    normalize_image,
    read_precontrast_mri,
    zscore_image,
)

def preprocess_and_save_volumes(subject_id, tcia_data_dir, mappings_filepath=None):
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
        image_array, dcm_data, volume_path = preprocess_dicoms(full_sequence_dir_)
        np.save(npy_save_path, image_array)
        volumes_npy_paths.append(npy_save_path)
    return preprocessed_array_base_save_path.parent, volumes_npy_paths


def preprocess_dicoms(full_sequence_dir):
    image_array, dcm_data, volume_path = read_precontrast_mri(full_sequence_dir)
    image_array = zscore_image(normalize_image(image_array))
    return image_array,  dcm_data, volume_path


if __name__ == '__main__':
    base_path = Path(r"D:\projects-data\mazurowski\manifest-1654812109500")
    xlsx_mappings_filepath = base_path / "Breast-Cancer-MRI-filepath_filename-mapping.xlsx"
    csv_mappings_filepath = base_path / "Breast-Cancer-MRI-filepath_filename-mapping.csv"
    tcia_data_dir = base_path / "Duke-Breast-Cancer-MRI"

    subject_id = 'Breast_MRI_025'
    preprocessed_images_dirpath, volumes_npy_paths = preprocess_and_save_volumes(
        subject_id,
        tcia_data_dir,
        mappings_filepath=csv_mappings_filepath,
    )
    subject_dirpath = (
            tcia_data_dir
            / subject_id

        # / "3.000000-ax t1-95549"
    )
    volumes_dirpath = [item for item in subject_dirpath.iterdir() if item.is_dir()][0]
    run(
        target_tissue="breast",
        image_dir=str(preprocessed_images_dirpath),
        input_mask_dir=str(base_path / "saved_results"),
        save_masks_dir=str(base_path / "saved_results"),
    )
