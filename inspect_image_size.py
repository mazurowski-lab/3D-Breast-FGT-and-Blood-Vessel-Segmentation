import logging
from pathlib import Path

import numpy as np

from preprocessing import read_precontrast_mri
from run import find_precontrast_volume
from setup_logging import setup_logging

def compare_shapes(preprocessed_array_base_save_path, breast_mask_save_path):
    for subject_dirpath in preprocessed_array_base_save_path.iterdir():
        subject_id = subject_dirpath.name
        print(subject_id)
        print((preprocessed_smri_pixels := np.load(preprocessed_array_base_save_path / subject_id)).shape)
        print((breast_mask_pixels := np.load(breast_mask_save_path / subject_id)).shape)
        # print((dv_mask_pixels := np.load(dv_masks_save_path / subject_id)).shape)
        print()



if __name__ == '__main__':
    # client = Client(threads_per_worker=4, n_workers=1)

    setup_logging(level=logging.DEBUG, log_dirpath=Path(__file__).parent)
    base_path = Path(r"D:\projects-data\mazurowski\manifest-1654812109500")

    tcia_data_dir = base_path / "Duke-Breast-Cancer-MRI"
    preprocessed_array_base_save_path = tcia_data_dir.parent / "preprocessed"

    breast_mask_save_path = base_path / "breast_masks"
    breast_mask_save_path.mkdir(exist_ok=True, parents=True)

    dv_masks_save_path = base_path / "fgt_bv_masks"
    dv_masks_save_path.mkdir(exist_ok=True, parents=True)

    # compare_shapes(preprocessed_array_base_save_path, breast_mask_save_path)

    csv_mappings_filepath = base_path / "Breast-Cancer-MRI-filepath_filename-mapping.csv"
    full_sequence_dir = find_precontrast_volume(subject_id="Breast_MRI_002", tcia_data_dir=tcia_data_dir, mappings_filepath=csv_mappings_filepath)
    read_precontrast_mri(full_sequence_dir=full_sequence_dir)