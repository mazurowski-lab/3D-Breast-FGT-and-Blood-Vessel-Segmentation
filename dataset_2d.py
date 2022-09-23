import os
import numpy as np
import nrrd
from torch.utils.data import Dataset
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms

# Images should be stored with the following file structure. This is after 
# they are processed with functions in preprocessing.py
# .
# ├── train_dir       
# │   ├── [subject_id_1].npy
# │   ├── [subject_id_2].npy
# │   └── ...
# ├── validation_dir  # Same structure as train dir
# └── test_dir        # Same structure as train dir
# 
# Masks should be stored with the same file structure. There should be separate
# upper level directories for breast and dense/vessels. 

class Dataset2D(Dataset):
    def __init__(
        self, 
        image_dir, 
        mask_dir,
        image_transforms = None, 
        mask_transforms = None
    ):
        """
        This class converts 3D MRI volumes and segmentations into a 2D dataset.
        It holds all of the data in memory to improve read speed. 

        Parameters
        ----------
        image_dir: str
            Path that leads to directory containing images with above file
            structure. 
        mask_dir: str
            Path that leads to directory with masks with same structure
        image_transforms: torchvision.transforms, optional
            Transforms to perform on images
        mask_transforms: torchvision.transforms, optional
            Transforms to perform on images

        """

        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

        # To improve efficiency of 2D dataset, all of the data will be loaded
        # into RAM. Otherwise it would be more complicated to load each
        # slice and provide them in batches to the model without having to
        # continually reload dicom and nrrd data. 
        #
        # However, there is an issue of looking up individual slices as the
        # entire dataset is iterated through. This is because each MRI volume
        # has a different number of slices and information would be lost if
        # we interpolated them all to the same number of slices. 
        # A dictionary will be used to convert the dataset index to the 
        # indicies needed to look up the individual slice within the list. 

        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)

        # Load images/masks; not stacked due to different dim for each volume
        self.image_array_list = []
        self.mask_array_list = []

        # Setting up dictionary
        # key: dataset index; value: (mri_list_index, slice_index)
        # There are three counts that will tick up during the dict creation
        # dataset_index_count is the overall length of the dataset
        # list_index_count is the number of numpy arrays in the list
        # slice_index_count is for each slice in individual arrays
        self.slice_indicies_dict = dict()
        dataset_index_count = 0
        list_index_count = 0

        print('Loading in MRI volumes and mask volumes...')

        self.subject_id_list = [
            x.rstrip('.npy') for x in sorted(os.listdir(image_dir))
        ]

        for subject_id in sorted(os.listdir(image_dir)):
            subject_id = subject_id.rstrip('.npy')

            image_array = np.load(image_dir / '{}.npy'.format(subject_id))
            self.image_array_list.append(image_array)

            mask_array = np.load(mask_dir / '{}.npy'.format(subject_id))
            self.mask_array_list.append(mask_array)

            assert image_array.shape == mask_array.shape, \
                """Subject: {}
                Image array and mask array shape do not match: {}, {}"""\
                    .format(
                        subject_id,
                        image_array.shape, 
                        mask_array.shape
                    )

            # Set up dictionary indicies
            slice_index_count = 0

            for i in range(image_array.shape[-1]):
                self.slice_indicies_dict[dataset_index_count] = \
                    (list_index_count, slice_index_count)
                dataset_index_count += 1
                slice_index_count += 1

            list_index_count += 1

        print('Loaded in {} MRI volumes and mask volumes'.format(
            list_index_count
        ))
        print('with a total of {} slices across all volumes.'.format(
            dataset_index_count
        ))

    def __len__(self):
        return len(self.slice_indicies_dict)

    def __getitem__(self, i):
        list_index, slice_index = self.slice_indicies_dict[i]

        # Get image and mask array based on indicies from dict
        image_array = np.expand_dims(
            self.image_array_list[list_index][:, :, slice_index], 
            axis=0
        )
        mask_array = np.expand_dims(
            self.mask_array_list[list_index][:, :, slice_index], 
            axis=0
        )

        image_array = torch.from_numpy(image_array)
        mask_array = torch.from_numpy(mask_array.copy())

        # print(image_array.shape)
        # print(mask_array.shape)

        if self.image_transforms != None:
            image_array = self.image_transforms(image_array)

        if self.mask_transforms != None:
            mask_array = self.mask_transforms(mask_array)

        return {
            'image': image_array,
            'mask': mask_array
        }

class Dataset2DWithInputChannel(Dataset):
    def __init__(
        self, 
        image_dir, 
        additional_input_dir,
        mask_dir,
        resize_dims,
        image_transforms = None, 
        mask_transforms = None
    ):
        """
        This class converts 3D MRI volumes and segmentations into a 2D dataset.
        It holds all of the data in memory to improve read speed. This dataset
        additionally allow for the input of another mask (or input) that will
        be used as a second channel in images outputted. 

        Parameters
        ----------
        image_dir: str
            Path that leads to directory containing images with above file
            structure. 
        additional_input_dir: str
            Path that leads to directory containing additional inputs with
            same file structure. 
        mask_dir: str
            Path that leads to directory with masks with same structure
        resize_dims: (int, int)
            Dimensions to resize everything to. This is needed for this class
            since the additional input and images may be different sizes
            initially. 
        image_transforms: torchvision.transforms, optional
            Transforms to perform on images
        mask_transforms: torchvision.transforms, optional
            Transforms to perform on images

        """

        self.resize_transforms = transforms.Compose([
            transforms.Resize(resize_dims)
        ])

        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

        # To improve efficiency of 2D dataset, all of the data will be loaded
        # into RAM. Otherwise it would be more complicated to load each
        # slice and provide them in batches to the model without having to
        # continually reload dicom and nrrd data. 
        #
        # However, there is an issue of looking up individual slices as the
        # entire dataset is iterated through. This is because each MRI volume
        # has a different number of slices and information would be lost if
        # we interpolated them all to the same number of slices. 
        # A dictionary will be used to convert the dataset index to the 
        # indicies needed to look up the individual slice within the list. 

        image_dir = Path(image_dir)
        additional_input_dir = Path(additional_input_dir)
        mask_dir = Path(mask_dir)

        # Load images/masks; not stacked due to different dim for each volume
        self.image_array_list = []
        self.additional_input_list = []
        self.mask_array_list = []

        # Setting up dictionary
        # key: dataset index; value: (mri_list_index, slice_index)
        # There are three counts that will tick up during the dict creation
        # dataset_index_count is the overall length of the dataset
        # list_index_count is the number of numpy arrays in the list
        # slice_index_count is for each slice in individual arrays
        self.slice_indicies_dict = dict()
        dataset_index_count = 0
        list_index_count = 0

        print('Loading in MRI volumes and mask volumes...')

        self.subject_id_list = [
            x.rstrip('.npy') for x in sorted(os.listdir(image_dir))
        ]

        for subject_id in sorted(os.listdir(image_dir)):
            subject_id = subject_id.rstrip('.npy')

            image_array = np.load(image_dir / '{}.npy'.format(subject_id))
            self.image_array_list.append(image_array)

            additional_input_array = np.load(
                additional_input_dir / '{}.npy'.format(subject_id)
            )
            self.additional_input_list.append(additional_input_array)

            mask_array = np.load(mask_dir / '{}.npy'.format(subject_id))
            self.mask_array_list.append(mask_array)

            assert image_array.shape == mask_array.shape, \
                """Subject: {}
                Image array and mask array shape do not match: {}, {}"""\
                    .format(
                        subject_id,
                        image_array.shape, 
                        mask_array.shape
                    )

            # Set up dictionary indicies
            slice_index_count = 0

            for i in range(image_array.shape[-1]):
                self.slice_indicies_dict[dataset_index_count] = \
                    (list_index_count, slice_index_count)
                dataset_index_count += 1
                slice_index_count += 1

            list_index_count += 1

        print('Loaded in {} MRI volumes and mask volumes'.format(
            list_index_count
        ))
        print('with a total of {} slices across all volumes.'.format(
            dataset_index_count
        ))

    def __len__(self):
        return len(self.slice_indicies_dict)

    def __getitem__(self, i):
        list_index, slice_index = self.slice_indicies_dict[i]

        # Get image and mask array based on indicies from dict
        image_array = np.expand_dims(
            self.image_array_list[list_index][:, :, slice_index], 
            axis=0
        )
        additional_input_array = np.expand_dims(
            self.additional_input_list[list_index][:, :, slice_index], 
            axis=0
        )

        # mask_array = np.expand_dims(
        #     self.mask_array_list[list_index][:, :, slice_index], 
        #     axis=0
        # )
        mask_array = self.mask_array_list[list_index][:, :, slice_index]

        image_array = torch.from_numpy(image_array)
        additional_input_array = torch.from_numpy(additional_input_array)
        mask_array = torch.from_numpy(mask_array.copy())

        # print(mask_array.shape)
        mask_array = F.one_hot(mask_array.long(), 3)
        # print(mask_array.shape)
        mask_array = torch.permute(mask_array, (2, 0, 1))
        # print(mask_array.shape)

        # We need to make sure that both inputs are the same size before
        # putting into channel...
        image_array = self.resize_transforms(image_array)
        additional_input_array = self.resize_transforms(additional_input_array)
        mask_array = self.resize_transforms(mask_array)

        image_array = torch.cat((image_array, additional_input_array))

        # print(image_array.shape)
        # print(mask_array.shape)

        if self.image_transforms != None:
            image_array = self.image_transforms(image_array)

        if self.mask_transforms != None:
            mask_array = self.mask_transforms(mask_array)

        return {
            'image': image_array,
            'mask': mask_array
        }