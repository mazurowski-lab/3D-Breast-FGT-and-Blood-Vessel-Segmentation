import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
import os
import nrrd

# Images should be downloaded from Duke-Breast-Cancer-MRI Dataset on TCIA
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903
# Download should be done using descriptive file path. There is an associated 
# file path mapping table that can be downloaded to determine which series is 
# the precontrast sequence. 

def clean_filepath_filename_mapping_csv(filepath_filename_csv_path):
    """
    This function cleans the "Breast-Cancer-MRI-filepath_filename-mapping.csv" 
    file that can be downloaded from TCIA. It is originally an excel file. It
    can be saved as a csv for use in this function. This returns a DataFrame
    that pairs subject_ids to their precontrast dir

    Parameters
    ----------
    fpath_mapping_df: str
        Path that leads to Breast-Cancer-MRI-filepath_filename-mapping.csv

    Returns
    -------
    pd.DataFrame
        Cleaned mapping DataFrame that can be used to find precontrast MRI
        sequences

    """

    # Read in csv as DataFrame
    fpath_mapping_df = pd.read_csv(
        filepath_filename_csv_path
    )
    # We only need the filename and the descriptive path
    fpath_mapping_df = fpath_mapping_df.loc[
        :, ['original_path_and_filename', 'descriptive_path']
    ]

    # We only need the precontrast sequences
    fpath_mapping_df = fpath_mapping_df.loc[
        fpath_mapping_df['original_path_and_filename'].str.contains('pre')
    ]

    # Only need the subject ID in first column
    fpath_mapping_df['original_path_and_filename'] = \
        fpath_mapping_df['original_path_and_filename'].str.split('/').str[1]
    # For second column, only need the name of the dir containing the sequence
    # Need to remove some extra slashes that are in there erroneously
    fpath_mapping_df['descriptive_path'] = \
        fpath_mapping_df['descriptive_path'].str.replace('//', '/')
    fpath_mapping_df['descriptive_path'] = \
        fpath_mapping_df['descriptive_path'].str.split('/').str[-2]

    # Drop duplicates so each subject has on entry
    fpath_mapping_df = fpath_mapping_df.drop_duplicates(
        subset='original_path_and_filename'
    )

    # Rename columns for better clarity
    fpath_mapping_df = fpath_mapping_df.rename(
        columns={
            'original_path_and_filename': 'subject_id',
            'descriptive_path': 'precontrast_dir'
        }
    )

    return fpath_mapping_df

def read_precontrast_mri(
    subject_id, 
    tcia_data_dir, 
    fpath_mapping_df
):
    """
    Reads in the precontrast MRI data given a subject ID. 
    This function also aligns the patient orientation so the patient's body
    is in the lower part of image. The slices from the beginning move inferior
    to superior. 


    Parameters
    ----------
    subject_id: str
        Subject_id (e.g. Breast_MRI_001)
    tcia_data_dir: str 
        Path of downloaded database from TCIA
    fpath_mapping_df: pd.DataFrame
        Cleaned mapping DataFrame that can be used to find precontrast MRI
        sequences

    Returns
    -------
    np.Array
        Raw MRI volume data read from all .dcm files
    pydicom.dataset.FileDataset
        Dicom data from final slice read. This is used for obtaining things
        such as pixel spacing, image orientation, etc. 

    """
    tcia_data_dir = Path(tcia_data_dir)

    # Get the sequence dir from the DataFrame
    sequence_dir = fpath_mapping_df.loc[
        fpath_mapping_df['subject_id'] == subject_id, 'precontrast_dir'
    ].iloc[0]

    # There's also a subdir for every subject that contains the sequences
    # There is only one of these
    sub_dir = os.listdir(tcia_data_dir / subject_id)[0]

    full_sequence_dir = tcia_data_dir / subject_id / sub_dir / sequence_dir

    # Now we can iterate through the files in the sequence dir and reach each
    # of them into a numpy array
    dicom_file_list = sorted(os.listdir(full_sequence_dir))
    dicom_data_list = []

    # Saving the values of first two image positions
    # This is used to orient inferior to superior
    first_image_position = 0
    second_image_position = 0

    for i in range(len(dicom_file_list)):
        dicom_data = pydicom.dcmread(full_sequence_dir / dicom_file_list[i])
        
        if i == 0:
            first_image_position = dicom_data[0x20, 0x32].value[-1]
        elif i == 1:
            second_image_position = dicom_data[0x20, 0x32].value[-1]

        dicom_data_list.append(dicom_data.pixel_array)
        
    # Stack in numpy array
    image_array = np.stack(dicom_data_list, axis=-1)

    # Rotate if inferior and superior are flipped
    if first_image_position > second_image_position:
        image_array = np.rot90(image_array, 2, (1, 2))

    # For patients in a certain orentation, also need to flip in another axis
    # This is the same in all dicom files so we can just use the last
    # dicom file that we have from the iteration. It also needs to be rounded.
    if round(dicom_data[0x20, 0x37].value[0], 0) == -1:
        image_array = np.rot90(image_array, 2)

    return image_array, dicom_data

def read_precontrast_mri_and_segmentation(
    subject_id, 
    tcia_data_dir, 
    fpath_mapping_df,
    segmentation_dir
):
    """
    Reads in the precontrast MRI data given a subject ID. 
    This function also aligns the patient orientation so the patient's body
    is in the lower part of image. The slices from the beginning move inferior
    to superior. 
    Finally, because the information used to properly orient the MRI is used
    in this function, it also properly orients the segmentation data. 

    Parameters
    ----------
    subject_id: str
        Subject_id (e.g. Breast_MRI_001)
    tcia_data_dir: str 
        Path of downloaded database from TCIA
    fpath_mapping_df: pd.DataFrame
        Cleaned mapping DataFrame that can be used to find precontrast MRI
        sequences
    segmentation_dir: str
        Directory containing segmentations in the following format
        ├── [subject_id_1]  (e.g. Breast_MRI_001)   
        ├── Segmentation_[subject_id_1]_Breast.seg.nrrd
        └── Segmentation_[subject_id_1]_Dense_and_Vessels.seg.nrrd

    Returns
    -------
    np.Array
        Raw MRI volume data read from all .dcm files
    pydicom.dataset.FileDataset
        Dicom data from final slice read. This is used for obtaining things
        such as pixel spacing, image orientation, etc. 
    np.Array
        Segmentation data from .nrrd.seg file after proper orientation

    """
    tcia_data_dir = Path(tcia_data_dir)
    segmentation_dir = Path(segmentation_dir)

    # Get the sequence dir from the DataFrame
    sequence_dir = fpath_mapping_df.loc[
        fpath_mapping_df['subject_id'] == subject_id, 'precontrast_dir'
    ].iloc[0]

    # There's also a subdir for every subject that contains the sequences
    # There is only one of these
    sub_dir = os.listdir(tcia_data_dir / subject_id)[0]

    full_sequence_dir = tcia_data_dir / subject_id / sub_dir / sequence_dir

    # Now we can iterate through the files in the sequence dir and reach each
    # of them into a numpy array
    dicom_file_list = sorted(os.listdir(full_sequence_dir))
    dicom_data_list = []

    # Saving the values of first two image positions
    # This is used to orient inferior to superior
    first_image_position = 0
    second_image_position = 0

    for i in range(len(dicom_file_list)):
        dicom_data = pydicom.dcmread(full_sequence_dir / dicom_file_list[i])
        
        if i == 0:
            first_image_position = dicom_data[0x20, 0x32].value[-1]
        elif i == 1:
            second_image_position = dicom_data[0x20, 0x32].value[-1]

        dicom_data_list.append(dicom_data.pixel_array)
        
    # Stack in numpy array
    image_array = np.stack(dicom_data_list, axis=-1)

    # Read in breast_nrrd data
    nrrd_breast_data, _ = nrrd.read(
        segmentation_dir / '{}/Segmentation_{}_Breast.seg.nrrd'.format(
            subject_id, subject_id)
    )

    # Read in dense_and_vessels (abbreviated dv) data
    # Header is used to properly write classes
    nrrd_dv_data, nrrd_header = nrrd.read(
        segmentation_dir / \
            '{}/Segmentation_{}_Dense_and_Vessels.seg.nrrd'.format(
                subject_id, subject_id)
    )

    # For dense and vessels one more step needs to be taken.
    # Since the order of the dense and vessels may be incorrect,
    # they could be labeled with the wrong classes. We will always have
    # background = 0, vessels = 1, dense = 2. 

    # There are few situations where things can be mislabeled. 
    # One is where they simply have the wrong values.
    # Another is when the volume has two layers. 

    if len(nrrd_dv_data.shape) == 3:

        # First check to see if everything is correct
        if (
            nrrd_header['Segment0_Name'] == 'Vessels' and 
            nrrd_header['Segment1_Name'] == 'Dense' and
            nrrd_header['Segment0_LabelValue'] == '1' and
            nrrd_header['Segment1_LabelValue'] == '2'
        ):
            pass

        else:
            if nrrd_header['Segment0_Name'] == 'Vessels':
                original_vessels_label = \
                    int(nrrd_header['Segment0_LabelValue'])
                original_dense_label = \
                    int(nrrd_header['Segment1_LabelValue'])
            else:
                original_vessels_label = \
                    int(nrrd_header['Segment1_LabelValue'])
                original_dense_label = \
                    int(nrrd_header['Segment0_LabelValue'])

            # Start by changing the vessel label to a high number, then
            # fix all values. This is to prevent an incorrect np.where
            # statement. 
            nrrd_dv_data = np.where(
                nrrd_dv_data == original_vessels_label, 
                50, nrrd_dv_data
            )
            nrrd_dv_data = np.where(
                nrrd_dv_data == original_dense_label, 
                2, nrrd_dv_data
            )
            nrrd_dv_data = np.where(
                nrrd_dv_data == 50, 
                1, nrrd_dv_data
            )

    else:
        # There are two layers in the data. 
        if nrrd_header['Segment0_Name'] == 'Vessels':
            vessel_layer = int(nrrd_header['Segment0_Layer'])
            vessel_label = int(nrrd_header['Segment0_LabelValue'])
            dense_layer = int(nrrd_header['Segment1_Layer'])
            dense_label = int(nrrd_header['Segment1_LabelValue'])
        else:
            vessel_layer = int(nrrd_header['Segment1_Layer'])
            vessel_label = int(nrrd_header['Segment1_LabelValue'])
            dense_layer = int(nrrd_header['Segment0_Layer'])
            dense_label = int(nrrd_header['Segment0_LabelValue'])
        
        vessel_array = nrrd_dv_data[vessel_layer, :, :, :]
        dense_array = nrrd_dv_data[dense_layer, :, :, :]

        # Change the values of them to match what we want
        vessel_array = np.where(
            vessel_array == vessel_label, 1, vessel_array
        )
        dense_array = np.where(
            dense_array == dense_label, 2, dense_array
        )

        nrrd_dv_data = vessel_array + dense_array

        # Might be some overlap; we'll change that into dense
        nrrd_dv_data = np.where(
            nrrd_dv_data == 3, 2, nrrd_dv_data
        )

    assert (np.unique(nrrd_dv_data) == np.array([0, 1, 2])).all(), \
        "{} dense/vessel array has incorrect values".format(subject_id)

    assert (image_array.shape == nrrd_breast_data.shape) and \
        (image_array.shape == nrrd_dv_data.shape), \
        """"Subject {}: Shape mismatch between arrays.
        Image, breast, dv shapes: {}, {}, {}
        """.format(
            subject_id, 
            image_array.shape,
            nrrd_breast_data.shape, 
            nrrd_dv_data.shape
        )

    # All nrrd data needs to be rotated/flipped
    # Some require additional flipping/rotation using criteria below
    nrrd_breast_data = np.flip(np.rot90(nrrd_breast_data), axis=1)
    nrrd_dv_data = np.flip(np.rot90(nrrd_dv_data), axis=1)
        
    # Rotate if inferior and superior are flipped
    if first_image_position > second_image_position:
        image_array = np.rot90(image_array, 2, (1, 2))
        nrrd_breast_data = np.flip(nrrd_breast_data, axis=1)
        nrrd_dv_data = np.flip(nrrd_dv_data, axis=1)
        
    # For patients in a certain orentation, also need to flip in another axis
    # This is the same in all dicom files so we can just use the last
    # dicom file that we have from the iteration. It also needs to be rounded.
    if round(dicom_data[0x20, 0x37].value[0], 0) == -1:
        image_array = np.rot90(image_array, 2)
    else:
        nrrd_breast_data = np.rot90(nrrd_breast_data, 2)
        nrrd_dv_data = np.rot90(nrrd_dv_data, 2)

    return image_array, dicom_data, nrrd_breast_data, nrrd_dv_data


def normalize_image(image_array, min_cutoff = 0.001, max_cutoff = 0.001):
    """
    Normalize the intensity of an image array by cutting off min and max values 
    to a certain percentile and set all values above/below that percentile to 
    the new max/min. 

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
    min_cutoff: float
        Minimum percentile of image to keep. (0.1% = 0.001)
    max_cutoff: float
        Maximum percentile of image to keep. (0.1% = 0.001)

    Returns
    -------
    np.array
        Normalized image

    """

    # Sort image values
    sorted_array = np.sort(image_array.flatten())

    # Find %ile index and get values
    min_index = int(len(sorted_array) * min_cutoff)
    min_intensity = sorted_array[min_index]

    max_index = int(len(sorted_array) * min_cutoff) * -1
    max_intensity = sorted_array[max_index]

    # Normalize image and cutoff values
    image_array = (image_array - min_intensity) / \
        (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0

    return image_array

def zscore_image(image_array):
    """
    Convert intensity values in an image to zscores:
    zscore = (intensity_value - mean) / standard_deviation

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
    Returns
    -------
    np.array
        Image with zscores for values

    """

    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    return image_array