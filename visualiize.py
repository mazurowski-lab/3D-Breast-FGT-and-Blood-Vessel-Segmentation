from pathlib import Path

import nibabel
import numpy as np
from torchio import Subject, ScalarImage, LabelMap

from run import visualize


def load_1():
    data_base_dirpath = Path(
        r"C:\Users\kshit\Projects\projects-data\mazurowski\data\Duke-Breast-Cancer-MRI\manifest-1654812109500"
        )
    saved_results_rel_path = "saved_results"
    subject_id = "Breast_MRI_018"
    series_name = "6.000000-ax dyn pre-74410"
    preprocessed_smri_filename = "preprocessed_smri"
    breast_mask_filename = "breast_mask"
    fgt_mask_filename = "fgt_mask"
    ext = ".nii.gz"

    dirpath = data_base_dirpath / saved_results_rel_path / subject_id / series_name

    subject = Subject(
        subject=subject_id,
        series=series_name,
        t1=ScalarImage((dirpath / preprocessed_smri_filename).with_suffix(ext)),
        breast=LabelMap((dirpath / breast_mask_filename).with_suffix(ext)),
        fgt_bv=LabelMap((dirpath / fgt_mask_filename).with_suffix(ext)),
        )
    subject.plot()


dirpath = Path(r"C:\Users\kshit\Projects\projects-data\mazurowski\data\Duke-Breast-Cancer-MRI\manifest-1654812109500")
subject_id = "Breast_MRI_012"


preprocessed_path = dirpath / "preprocessed"/ f"{subject_id}"
breast_mask_path = dirpath / "breast_masks"/ f"{subject_id}"
fgt_bv_mask_path = dirpath / "fgt_bv_masks"/ f"{subject_id}"

preprocessed_nifti = nibabel.Nifti1Image(np.load(preprocessed_path.with_suffix(".npy")), affine=np.eye(4))
breast_mask_nifti = nibabel.Nifti1Image(np.load(breast_mask_path.with_suffix(".npy")).astype(np.float32), affine=np.eye(4))
fgt_bv_mask_nifti = nibabel.Nifti1Image(np.load(fgt_bv_mask_path.with_suffix(".npy")).astype(np.float32), affine=np.eye(4))

nibabel.save(preprocessed_nifti, filename=preprocessed_path.with_suffix(".nii.gz"))
nibabel.save(breast_mask_nifti, filename=breast_mask_path.with_suffix(".nii.gz"))
nibabel.save(fgt_bv_mask_nifti, filename=fgt_bv_mask_path.with_suffix(".nii.gz"))

t1 = ScalarImage(preprocessed_path.with_suffix(".nii.gz"))
breast_mask = LabelMap(breast_mask_path.with_suffix(".nii.gz"))
fgt_bv_mask = LabelMap(fgt_bv_mask_path.with_suffix(".nii.gz"))

fgt_bv_mask.to_gif(axis=0, duration=10, output_path=fgt_bv_mask_path.with_suffix(".gif"))

subject = Subject(
    subject_id=subject_id,
    t1=t1,
    breast_mask=breast_mask,
    fgt_bv_mask=fgt_bv_mask,
    )
subject.plot()
