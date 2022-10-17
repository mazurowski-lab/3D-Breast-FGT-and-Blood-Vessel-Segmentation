import os
import numpy as np
import nrrd
from torch.utils.data import Dataset
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
import random
import torchio as tio

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

class _Dataset3DBase(Dataset):
    def __init__(
        self, 
        image_dir, 
        mask_dir,
        additional_input_dir = None,
        transforms = None,
        one_hot_mask = False,
        image_only = False
    ):
        """
        This class converts 3D MRI volumes and segmentations into a 3D dataset.
        It holds all of the data in memory to improve read speed. 
        This class is a base class for all other datasets that use various
        methods to determine how volumes are fed into the model. 
        This is not meant for use alone. 

        Parameters
        ----------
        image_dir: str
            Path that leads to directory containing images with above file
            structure. 
        mask_dir: str
            Path that leads to directory with masks with same structure
            Set this to None if using image_only
        transforms: torchio.transforms.Transform
            Transforms to apply
        additional_input_dir: str
            Path that leads to a directory with masks or images that will be
            used as an additional channel in the input
        one_hot_mask: bool
            Will one hot encode masks for multi-class tasks. 
        image_only: bool
            The dataset will be created without true masks. This is used for
            predictions

        """

        self.transforms = transforms
        self.one_hot_mask = one_hot_mask
        self.image_only = image_only

        # To improve efficiency of dataset, all of the data will be loaded
        # into RAM. Otherwise it would be more complicated to load each
        # slice and provide them in batches to the model without having to
        # continually reload dicom and nrrd data. 

        image_dir = Path(image_dir)

        # Load images/masks; not stacked due to different dim for each volume
        self.image_array_list = []
        
        if not self.image_only:
            mask_dir = Path(mask_dir)
            self.mask_array_list = []

        if additional_input_dir:
            additional_input_dir = Path(additional_input_dir)
            self.additional_input_list = []

        # Save image shapes if needed for future use
        self.image_shape_list = []

        print('Loading in MRI volumes and mask volumes...')

        self.subject_id_list = [
            x.rstrip('.npy') for x in sorted(os.listdir(image_dir))
        ]

        for subject_id in self.subject_id_list:
            image_array = np.load(image_dir / '{}.npy'.format(subject_id))
            self.image_array_list.append(image_array)

            if not self.image_only:
                mask_array = np.load(mask_dir / '{}.npy'.format(subject_id))
                self.mask_array_list.append(mask_array)

                assert image_array.shape == mask_array.shape, \
                    'Image array and mask array shape do not match: {}, {}'\
                        .format(image_array.shape, mask_array.shape)

            if additional_input_dir:
                additional_input_array = np.load(
                    additional_input_dir / '{}.npy'.format(subject_id)
                )
                self.additional_input_list.append(additional_input_array)

                assert image_array.shape == additional_input_array.shape, \
                    """Subject: {}
                    Image array and addutional input array shape do not match: {}, {}"""\
                        .format(
                            subject_id,
                            image_array.shape, 
                            additional_input_array.shape
                        )

            self.image_shape_list.append(image_array.shape)

        print('Loaded in {} MRI volumes and mask volumes'.format(
            len(self.image_array_list)
        ))

    def get_image_mask_using_indicies(
        self, list_index, x_index, y_index, z_index
    ):

        image_array = np.expand_dims(
            self.image_array_list[list_index][
                x_index:x_index + self.input_dim,
                y_index:y_index + self.input_dim,
                z_index:z_index + self.input_dim
            ], 
            axis=0
        )
        image_array = torch.from_numpy(image_array)

        if self.image_only and not hasattr(self, 'additional_input_list'):
            return image_array
            
        elif self.image_only and hasattr(self, 'additional_input_list'):
            additional_input_array = np.expand_dims(
                self.additional_input_list[list_index][
                    x_index:x_index + self.input_dim,
                    y_index:y_index + self.input_dim,
                    z_index:z_index + self.input_dim
                ], 
                axis=0
            )
            additional_input_array = torch.from_numpy(additional_input_array)
            image_array = torch.cat((image_array, additional_input_array))

            return image_array

        mask_array = np.expand_dims(
            self.mask_array_list[list_index][
                x_index:x_index + self.input_dim,
                y_index:y_index + self.input_dim,
                z_index:z_index + self.input_dim
            ], 
            axis=0
        )
        mask_array = torch.from_numpy(mask_array.copy())

        if hasattr(self, 'additional_input_list'):
            additional_input_array = np.expand_dims(
                self.additional_input_list[list_index][
                    x_index:x_index + self.input_dim,
                    y_index:y_index + self.input_dim,
                    z_index:z_index + self.input_dim
                ], 
                axis=0
            )
            additional_input_array = torch.from_numpy(additional_input_array)
            image_array = torch.cat((image_array, additional_input_array))

        if self.one_hot_mask:
            mask_array = convert_to_one_hot(mask_array)

        return image_array, mask_array

def transform_using_torchio(
    image,
    mask,
    transforms,
    image_only = False
):
    """
    Uses a TorchIO subject to perform transforms. TorchIO has a lot of 
    powerful transforms that can be used, but required subject class to apply
    transforms consistently (and easily) to images and mask

    Parameters
    ----------
    image: torch.Tensor
        Image data, can be multiple channels
    mask: torch.Tensor
        True mask
    transforms: torchio.transforms.Transform
        Transforms to apply

    Returns
    -------
    dict
        A dictionary with image and mask keys that contain corresponding
        images and masks after transforms. 

    """

    subject = tio.Subject(
        image = tio.ScalarImage(tensor = image),
        mask = tio.LabelMap(tensor = mask)
    )

    transformed_subject = transforms(subject)

    return {
        'image': transformed_subject.image[tio.DATA].float(),
        'mask': transformed_subject.mask[tio.DATA].float()
    }

def transform_using_torchio_image_only(
    image,
    transforms
):
    """
    Uses a TorchIO subject to perform transforms. TorchIO has a lot of 
    powerful transforms that can be used, but required subject class to apply
    transforms consistently (and easily) to images and mask

    Parameters
    ----------
    image: torch.Tensor
        Image data, can be multiple channels
    transforms: torchio.transforms.Transform
        Transforms to apply

    Returns
    -------
    dict
        A dictionary with image and mask keys that contain corresponding
        images and masks after transforms. 

    """

    subject = tio.Subject(
        image = tio.ScalarImage(tensor = image)
    )

    transformed_subject = transforms(subject)

    return {
        'image': transformed_subject.image[tio.DATA].float()
    }


def convert_to_one_hot(
    mask
):
    """
    Converts a multi-class mask into a one hot encoded version. 

    Parameters
    ----------
    mask: torch.Tensor
        True mask with multiple classes

    Returns
    -------
    torch.Tensor
        A dictionary with image and mask keys that contain corresponding
        images and masks after transforms. 
    """

    mask = mask.squeeze()
    mask = F.one_hot(mask.long(), 3)
    # print(mask.shape)
    return torch.permute(mask, (3, 0, 1, 2))

class Dataset3DSimple(_Dataset3DBase):
    """
    This class uses the base class to simply use the full volume for each
    input. 
    """

    def __len__(self):
        return len(self.image_array_list)

    def __getitem__(self, i):

        # Get image and mask array based on indicies 
        image_array = np.expand_dims(self.image_array_list[i], axis=0)

        if self.image_only:
            return transform_using_torchio_image_only(image_array, self.transforms)

        mask_array = np.expand_dims(self.mask_array_list[i], axis=0)

        if self.one_hot_mask:
            mask_array = convert_to_one_hot(mask_array)

        return transform_using_torchio(image_array, mask_array, self.transforms)

def generate_random_boxes_dict(
    subject_count,
    total_samples
):
    """
    Generates a dictionary of indicies that can be used for the random
    subsamples in the Dataset3DRandom class. 

    Parameters
    ----------
    subject_count: int
        Total number of subjects
    total_samples: int
        Total number of samples to do across all subjects

    Returns
    -------
    dict
        Dictionary containing the indicies for each box
    """

    # Setting up dictionary
    # key: dataset index; value: mri_list_index
    # There are two counts that will tick up during the dict creation
    # dataset_index_count is the overall length of the dataset
    # list_index_count is the number of numpy arrays in the list
    #
    # The number of boxes possible in one volume is 
    # (total_samples) / (total number of images)
    box_indicies_dict = dict()
    dataset_index_count = 0
    list_index_count = 0

    boxes_per_subject = int(total_samples / subject_count)

    for i in range(subject_count):
        for j in range(boxes_per_subject):
            box_indicies_dict[dataset_index_count] = list_index_count
            dataset_index_count += 1

        list_index_count += 1

    return box_indicies_dict

class Dataset3DRandom(_Dataset3DBase):
    def __init__(
        self, 
        image_dir, 
        mask_dir,
        input_dim,
        total_samples,
        additional_input_dir = None,
        transforms = None,
        one_hot_mask = False
    ):
        """
        This class random samples the full volume at a fixed input size. 

        Parameters
        ----------
        input_dim: int
            Length of the cube that is taken from the full volume. 
        total_samples: int
            How many boxes should be sampled in total during one full iteration

        """

        super().__init__(
            image_dir = image_dir, 
            mask_dir = mask_dir,
            additional_input_dir = additional_input_dir,
            transforms = transforms,
            one_hot_mask = one_hot_mask
        )

        self.input_dim = input_dim

        self.box_indicies_dict = generate_random_boxes_dict(
            len(self.subject_id_list), total_samples
        )

        print('with a total of {} subvolumes across all volumes'.format(
            len(self.box_indicies_dict)
        ))

    def __len__(self):
        return len(self.box_indicies_dict)

    def __getitem__(self, i):
        list_index = self.box_indicies_dict[i]

        # We need to figure out the indicies of the box we're going to grab
        x_length, y_length, z_length = self.image_array_list[list_index].shape

        # The box will be sampled randomly along each axis. 
        x_index = random.randrange(x_length - self.input_dim)
        y_index = random.randrange(y_length - self.input_dim)
        z_index = random.randrange(z_length - self.input_dim)

        image_array, mask_array = self.get_image_mask_using_indicies(
            list_index, x_index, y_index, z_index
        )

        return transform_using_torchio(image_array, mask_array, self.transforms)

def generate_divided_boxes_dict(
    input_dim,
    x_y_divisions,
    z_division,
    shape_list
):
    """
    Generates a dictionary of indicies that can be used for the random
    subsamples in the Dataset3DDivided class. 

    Parameters
    ----------
    input_dim: int
        Length of the cube to sample
    x_y_divisions: int
        Number of times to sample along x_y axis
    z_division: int
        Number of times to sample along z axis

    Returns
    -------
    dict
        Dictionary containing the indicies for each box. See below
    """
    # Setting up dictionary
    # key: dataset index; value: (mri_list_index, box_index)
    # There are three counts that will tick up during the dict creation
    # dataset_index_count is the overall length of the dataset
    # list_index_count is the number of numpy arrays in the list
    # box_index_count is for each box that can be created in the array

    box_indicies_dict = dict()
    dataset_index_count = 0
    list_index_count = 0

    for shape in shape_list:
        x_length, y_length, z_length = shape
        x_step_size = int((x_length - input_dim) / (x_y_divisions - 1))
        y_step_size = int((y_length - input_dim) / (x_y_divisions - 1))
        z_step_size = int((z_length - input_dim) / (z_division - 1))

        for z in range(z_division):
            for y in range(x_y_divisions):
                for x in range(x_y_divisions):
                    
                    if x != x_y_divisions - 1:
                        x_index = x * x_step_size
                    else:
                        x_index = x_length - input_dim

                    if y != x_y_divisions - 1:
                        y_index = y * y_step_size
                    else:
                        y_index = y_length - input_dim

                    if z != z_division - 1:
                        z_index = z * z_step_size
                    else:
                        z_index = z_length - input_dim
                    
                    box_indicies_dict[dataset_index_count] = \
                        [list_index_count, (x_index, y_index, z_index)]
                    
                    dataset_index_count += 1
        
        list_index_count += 1

    return box_indicies_dict

class Dataset3DDivided(_Dataset3DBase):
    def __init__(
        self, 
        image_dir, 
        mask_dir,
        input_dim,
        x_y_divisions,
        z_division,
        additional_input_dir = None,
        transforms = None,
        one_hot_mask = False,
        image_only = False
    ):
        """
        This class samples the full volume by sample an equal number of times
        along each axis as specified.  

        Parameters
        ----------
        input_dim: int
            Length of the cube that is taken from the full volume. 
        x_y_divisions: int
            Number of times to sample along x_y axis
        z_division: int
            Number of times to sample along z axis

        """

        super().__init__(
            image_dir = image_dir, 
            mask_dir = mask_dir,
            additional_input_dir = additional_input_dir,
            transforms = transforms,
            one_hot_mask = one_hot_mask,
            image_only = image_only
        )

        self.input_dim = input_dim

        self.box_indicies_dict = generate_divided_boxes_dict(
            input_dim,
            x_y_divisions,
            z_division,
            self.image_shape_list
        )

        print('with a total of {} subvolumes across all volumes'.format(
            len(self.box_indicies_dict)
        ))

    def __len__(self):
        return len(self.box_indicies_dict)

    def __getitem__(self, i):
        list_index, box_index = self.box_indicies_dict[i]
        x_index, y_index, z_index = box_index

        if self.image_only:
            image_array = self.get_image_mask_using_indicies(
                list_index, x_index, y_index, z_index
            )

            return transform_using_torchio_image_only(image_array, self.transforms)
        else:
            image_array, mask_array = self.get_image_mask_using_indicies(
                list_index, x_index, y_index, z_index
            )

            return transform_using_torchio(image_array, mask_array, self.transforms)

def generate_stack_dict(
    z_input_dim,
    z_step_size,
    shape_list
):

    """
    Generates a dictionary of indicies that can be used for the random
    subsamples in the Dataset3DDivided class. 

    Parameters
    ----------
    z_input_dim: int
        Number of slices to use for inpiut
    z_step_size: int
        How far to step sampling area along z axis. 
    shape_list: list
        List containing the dimensions of each volume, in order

    Returns
    -------
    dict
        Dictionary containing the indicies for each box. See below
    """

    # Setting up dictionary
    # key: dataset index; value: (mri_list_index, box_index)
    # There are three counts that will tick up during the dict creation
    # dataset_index_count is the overall length of the dataset
    # list_index_count is the number of numpy arrays in the list
    # box_index_count is for each box that can be created in the array
    box_indicies_dict = dict()
    dataset_index_count = 0
    list_index_count = 0

    for shape in shape_list:
        _, _, z_length = shape

        boxes_in_z_layer = int((z_length - z_input_dim)/z_step_size) + 1

        for i in range(boxes_in_z_layer):
            z_index = i * z_step_size

            box_indicies_dict[dataset_index_count] = \
                [list_index_count, z_index]
            
            dataset_index_count += 1
        
        list_index_count += 1

    return box_indicies_dict

class Dataset3DVerticalStack(_Dataset3DBase):
    def __init__(
        self, 
        image_dir, 
        mask_dir,
        z_input_dim,
        z_step_size,
        additional_input_dir = None,
        transforms = None,
        one_hot_mask = False
    ):
        """
        This class samples the full volume by sampling along the z axis a
        certain number of times. The z axis should not be resized by the x,y 
        will be

        Parameters
        ----------
        z_input_dim: int
            Number of slices to use for inpiut
        z_step_size: int
            How far to step sampling area along z axis. 

        """

        super().__init__(
            image_dir = image_dir, 
            mask_dir = mask_dir,
            additional_input_dir = additional_input_dir,
            transforms = transforms,
            one_hot_mask = one_hot_mask
        )

        self.z_input_dim = z_input_dim

        self.box_indicies_dict = generate_stack_dict(
            z_input_dim,
            z_step_size,
            self.image_shape_list
        )

        print('with a total of {} subvolumes across all volumes'.format(
            len(self.box_indicies_dict)
        ))

    def __len__(self):
        return len(self.box_indicies_dict)

    def __getitem__(self, i):
        list_index, z_index = self.box_indicies_dict[i]
        

        image_array = np.expand_dims(
            self.image_array_list[list_index][
                :, :, 
                z_index:z_index + self.z_input_dim
            ], 
            axis=0
        )
        mask_array = np.expand_dims(
            self.mask_array_list[list_index][
                :, :, 
                z_index:z_index + self.z_input_dim
            ], 
            axis=0
        )

        image_array = torch.from_numpy(image_array)
        mask_array = torch.from_numpy(mask_array.copy())

        if hasattr(self, 'additional_input_list'):
            additional_input_array = np.expand_dims(
                self.additional_input_list[list_index][
                    :, :, 
                    z_index:z_index + self.z_input_dim
                ], 
                axis=0
            )
            additional_input_array = torch.from_numpy(additional_input_array)
            image_array = torch.cat((image_array, additional_input_array))

        if self.one_hot_mask:
            mask_array = convert_to_one_hot(mask_array)

        return transform_using_torchio(image_array, mask_array, self.transforms)
