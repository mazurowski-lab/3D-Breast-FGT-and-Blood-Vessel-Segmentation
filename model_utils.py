import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from pathlib import Path

from losses import dice_coeff

import torchio as tio

import gc

def pred_and_save_masks_2d(
    model,
    saved_model_path,
    dataset,
    save_masks_dir,
    n_classes,
    num_workers = 8,
    use_parallel = True
):

    """
    This function performs predictions on a 2D dataset and saves the them
    as .npy 3D volumes for use later. 

    Parameters
    ----------
    model: unet
        Initialized model
    saved_model_path: str
        Path to saved model
    dataset: dataset.Dataset2D
        Dataset used to serve 2D images that used to perform predictions
    n_classes: int
        Number of classes in target segmentation
    save_masks_dir: str
        Directory to save predicted masks
    num_workers: int
        Number of workers to use in DataLoaders
    use_parallel: int
        Indicates whether the model to be loaded was trained on parallel GPUs

    """
    save_masks_dir = Path(save_masks_dir)

    assert os.path.isdir(save_masks_dir), \
        "{} directory does not exist".format(save_masks_dir)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    if use_parallel:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size = 1, 
        shuffle = False,
        num_workers = num_workers
    )

    # We will create a list of lists
    # Each list will be for an individual subject and have np arrays appended to it
    # At the end we'll stack them. 

    subject_list = []

    for i in range(len(dataset.image_array_list)):
        subject_list.append([])

    print('Predicting Masks...')
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        list_index, _ = dataset.slice_indicies_dict[i]
        
        image = batch['image']
        mask = batch['mask']
        
        image = image.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        
        with torch.no_grad():
            pred = model(image)
            
        if n_classes == 1:
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
        else:
            pred = F.softmax(pred, dim=1)

        pred = pred.cpu().detach().numpy()
        
        subject_list[list_index].append(torch.squeeze(pred))
    
    print('Saving Predictions...')
    for i in range(len(subject_list)):
        mask_array = np.stack(subject_list[i], axis=-1)

        np.save(save_masks_dir / '{}.npy'.format(dataset.subject_id_list[i]), mask_array)

def eval_2d_breast_model(
    model,
    breast_saved_model_path,
    breast_dataset,
    batch_size, 
    num_workers = 8,
    use_parallel = True
):

    """
    This function evaluates breast predictions on a 2D dataset. 

    Parameters
    ----------
    model: unet
        Initialized model
    breast_saved_model_path: str
        Path to saved model
    breast_dataset: dataset.Dataset2D
        Dataset used to serve 2D images that used to perform predictions
    batch_size: int
        Batch size
    num_workers: int
        Number of workers to use in DataLoaders
    use_parallel: int
        Indicates whether the model to be loaded was trained on parallel GPUs

    """

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    breast_loader = DataLoader(
        breast_dataset,
        batch_size = batch_size, 
        shuffle = False,
        num_workers = num_workers
    )

    if use_parallel:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(breast_saved_model_path))

    model.to(device)
    model.eval()

    criteron = nn.BCEWithLogitsLoss()

    val_loss = 0
    dice_loss = 0
    
    for batch in tqdm(breast_loader):
        images = batch['image']
        masks = batch['mask']
        
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        
        with torch.no_grad():
            preds = model(images)
            
        loss = criteron(preds, masks)
        
        sig_preds = torch.sigmoid(preds)
        sig_preds = (sig_preds > 0.5).float()

        dice = dice_coeff(sig_preds, masks)
        dice_loss += dice.item()

        val_loss += loss.item()

    val_loss = val_loss/len(breast_loader)

    dice_loss = dice_loss/len(breast_loader)
    print('Breast: Val BCE Loss: {}; Dice Coeff: {}'.format(
        val_loss, dice_loss
    ))

def eval_2d_dv_model(
    model,
    dv_saved_model_path,
    dv_dataset,
    batch_size, 
    num_workers = 8,
    use_parallel = True
):
    """
    This function evaluates FGT and blood vessel predictions on a 2D dataset. 

    Parameters
    ----------
    model: unet
        Initialized model
    dv_saved_model_path: str
        Path to saved model
    dv_dataset: dataset.Dataset2D
        Dataset used to serve 2D images that used to perform predictions
    batch_size: int
        Batch size
    num_workers: int
        Number of workers to use in DataLoaders
    use_parallel: int
        Indicates whether the model to be loaded was trained on parallel GPUs

    """

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dv_loader = DataLoader(
        dv_dataset,
        batch_size = batch_size, 
        shuffle = False,
        num_workers = num_workers
    )

    # Set up model

    if use_parallel:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(dv_saved_model_path))

    model.to(device)
    model.eval()

    criteron = nn.CrossEntropyLoss()

    val_loss = 0
    vessel_dice_loss = 0
    dense_dice_loss = 0
    
    for batch in tqdm(dv_loader):
        images = batch['image']
        masks = batch['mask']
        
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        
        with torch.no_grad():
            preds = model(images)
            
        loss = criteron(preds, masks)
        
        # Lets calculate the dice score for vessels and dense
        # Softmax and then get the max value for each voxel
        # print(preds)
        softmax_preds = F.softmax(preds, dim=1)
        # print(softmax_preds)
        softmax_preds = torch.argmax(softmax_preds, dim=1, keepdim=True)
        # print(softmax_preds)

        # We need to create an array for vessels and dense
        vessel_pred = torch.clone(softmax_preds)
        # Change all dense labels into background
        vessel_pred = torch.where(vessel_pred == 2, 0, vessel_pred).float()

        dense_pred = torch.clone(softmax_preds)
        # Vice versa for dense
        dense_pred = torch.where(dense_pred == 1, 0, dense_pred)
        # Also change values into 1
        dense_pred = torch.where(dense_pred == 2, 1, dense_pred).float()

        # Split true masks into vessel and dense
        vessel_mask = masks[:, 1:2, :, :]
        dense_mask = masks[:, 2:, :, :]

        # Calculate dice scores

        vessel_dice = dice_coeff(vessel_pred, vessel_mask)
        vessel_dice_loss += vessel_dice.item()

        dense_dice = dice_coeff(dense_pred, dense_mask)
        dense_dice_loss += dense_dice.item()

        val_loss += loss.item()

    val_loss = val_loss/len(dv_loader)
    vessel_dice_loss = vessel_dice_loss/len(dv_loader)
    dense_dice_loss = dense_dice_loss/len(dv_loader)

    print("""Dense/Vessels: Val CE Loss: {}
    Vessels Dice Coeff: {}; Dense Dice Coeff: {}""".format(
        val_loss, vessel_dice_loss, dense_dice_loss
    ))

def pred_and_save_masks_3d_divided(
    unet,
    saved_model_path,
    dataset,
    n_classes,
    save_masks_dir,
    num_workers = 8,
    target_subjects = None
):

    """
    This function performs predictions on a 3D dataset using the divided
    dataset and saves the them as .npy 3D volumes for use later. 

    Parameters
    ----------
    unet: unet,
        Initialized model
    saved_model_path: str
        Path to saved model
    dataset: dataset.Dataset3DDivided
        Dataset used to serve 3D images that used to perform predictions
    n_classes: int
        Number of classes in output
    save_masks_dir: str
        Directory to save predicted masks
    num_workers: int
        Number of workers to use in DataLoaders
    target_subjects: [str, str, ...]
        List of subjects to perform predictions on

    """
    save_masks_dir = Path(save_masks_dir)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = nn.DataParallel(unet)

    unet.load_state_dict(torch.load(saved_model_path))
    unet.to(device)
    unet.eval()

    # Dataset must be Dataset3DDividided since it spans across entire volume
    loader = DataLoader(
        dataset,
        batch_size = 1, 
        shuffle = False,
        num_workers = num_workers
    )

    # This is where it gets complicated
    # We will create a numpy array with an array at every single voxel
    # For each model prediction, we will have a box within the volume with
    # actual prediction values. We'll figure out where this box is within the
    # volume and fill all other voxels with np.nan. We can then concat this
    # with the initial numpy array. Eventually we'll have a bunch of guesses
    # with np.nans in each voxel. We can then take the mean (ignoring nan) 
    # along the correct axis to obtain the fully estimation across the whole
    # volume. 

    # Due to the nature of this huge np array, it would be best to save it
    # as we move through the subjects. It's not the cleanest, but we can 
    # keep track of which subject we're on and when we reach the next subject,
    # save the old array and create the new one. 

    pred_volume_list = []

    if target_subjects:
        target_subjects = sorted(target_subjects)
        current_subject = target_subjects[0]
    else:
        current_subject = dataset.subject_id_list[0]

    print('Predicting Masks...')
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        
        list_index, box_index = dataset.box_indicies_dict[i]

        # Continue if 
        if target_subjects != None and \
            not dataset.subject_id_list[list_index] in target_subjects:
            
            continue

        # If we have moved onto the next subject, we need to save and reinitialize
        if dataset.subject_id_list[list_index] != current_subject:
            # print('Completing subject {}'.format(current_subject))
            # Take the nan mean along the last axis
            volume_pred_array = np.concatenate(pred_volume_list, axis=-1)
            volume_pred_array = np.nanmean(volume_pred_array, axis=-1)
            # Verify that there are no nans left in the array anymore
            assert np.isnan(np.min(volume_pred_array)) == False, \
                '{} still contains nan values when trying to save'.format(
                    current_subject
                )

            # Save the array; we'll keep the raw values
            np.save(
                save_masks_dir / '{}.npy'.format(current_subject), 
                volume_pred_array
            )

            del pred_volume_list
            del volume_pred_array
            gc.collect()

            # print(current_subject, dataset.subject_id_list[list_index])

            # Now we can change the subject and create a new array
            current_subject = dataset.subject_id_list[list_index] 
            pred_volume_list = []

        x_index, y_index, z_index = box_index
        
        # Debugging
        # print('{}\t{}:{}\t{}:{}\t{}:{}'.format(
        #     i,
        #     x_index, x_index + dataset.input_dim,
        #     y_index, y_index + dataset.input_dim,
        #     z_index, z_index + dataset.input_dim
        # ))

        # Get preds
        image = batch['image']
        # mask = batch['mask']

        image = image.to(device, dtype=torch.float32)
        # mask = mask.to(device, dtype=torch.float32)

        with torch.no_grad():
            pred = unet(image)

        # Sigmoid
        if n_classes == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = F.softmax(pred, dim=1)

        # Turn into numpy array and fix dims
        pred = pred.cpu().detach().numpy()
        pred = np.squeeze(pred)
        pred = np.expand_dims(pred, axis=-1).astype(np.half)

        # Make an empty array that will be filled in the correct area with preds
        x_length, y_length, z_length = dataset.image_array_list[list_index].shape

        if n_classes == 1:
            current_pred_array = np.empty(
                (x_length, y_length, z_length, 1), dtype=np.half
            )
            current_pred_array[:] = np.nan

            current_pred_array[
                x_index:x_index + dataset.input_dim,
                y_index:y_index + dataset.input_dim,
                z_index:z_index + dataset.input_dim
            ] = pred
        else:
            current_pred_array = np.empty(
                (n_classes, x_length, y_length, z_length, 1), dtype=np.half
            )
            current_pred_array[:] = np.nan

            current_pred_array[
                :, 
                x_index:x_index + dataset.input_dim,
                y_index:y_index + dataset.input_dim,
                z_index:z_index + dataset.input_dim
            ] = pred

        # print(pred.dtype)

        pred_volume_list.append(current_pred_array)

    # Need to do it once more for the final subject
    # Take the nan mean along the last axis
    volume_pred_array = np.concatenate(pred_volume_list, axis=-1)
    volume_pred_array = np.nanmean(volume_pred_array, axis=-1)
    # Verify that there are no nans left in the array anymore
    assert np.isnan(np.min(volume_pred_array)) == False, \
        '{} still contains nan values when trying to save'.format(
            current_subject
        )

    # Save the array; we'll keep the raw values
    np.save(
        save_masks_dir / '{}.npy'.format(current_subject), 
        volume_pred_array
    )
    

def pred_and_save_masks_3d_stacked(
    saved_model_path,
    dataset,
    unet,
    n_classes,
    n_channels,
    save_masks_dir,
    num_workers = 8,
    target_subjects = None
):

    """
    This function performs predictions on a 3D dataset using the stacked
    dataset and saves the them as .npy 3D volumes for use later. 

    Parameters
    ----------
    unet: unet,
        Initialized model
    saved_model_path: str
        Path to saved model
    dataset: dataset.Dataset3DStacked
        Dataset used to serve 3D images that used to perform predictions
    n_classes: int
        Number of classes in output
    save_masks_dir: str
        Directory to save predicted masks
    num_workers: int
        Number of workers to use in DataLoaders
    target_subjects: [str, str, ...]
        List of subjects to perform predictions on

    """
    save_masks_dir = Path(save_masks_dir)
    assert os.path.isdir(save_masks_dir), \
        "{} directory does not exist".format(save_masks_dir)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = nn.DataParallel(unet)

    unet.load_state_dict(torch.load(saved_model_path))
    unet.to(device)
    unet.eval()

    # Dataset must be Dataset3DStacked since it spans across entire volume
    loader = DataLoader(
        dataset,
        batch_size = 1, 
        shuffle = False,
        num_workers = num_workers
    )

    # This is where it gets complicated
    # We will create a numpy array with an array at every single voxel
    # For each model prediction, we will have a box within the volume with
    # actual prediction values. We'll figure out where this box is within the
    # volume and fill all other voxels with np.nan. We can then concat this
    # with the initial numpy array. Eventually we'll have a bunch of guesses
    # with np.nans in each voxel. We can then take the mean (ignoring nan) 
    # along the correct axis to obtain the fully estimation across the whole
    # volume. 

    # Due to the nature of this huge np array, it would be best to save it
    # as we move through the subjects. It's not the cleanest, but we can 
    # keep track of which subject we're on and when we reach the next subject,
    # save the old array and create the new one. 

    pred_volume_list = []

    if target_subjects:
        target_subjects = sorted(target_subjects)
        current_subject = target_subjects[0]
    else:
        current_subject = dataset.subject_id_list[0]

    print('Predicting Masks...')
    for i, batch in tqdm(enumerate(loader), total=len(loader)):

        list_index, z_index = dataset.box_indicies_dict[i]
        # Make an empty array that will be filled in the correct area with preds
        x_length, y_length, z_length = dataset.image_shape_list[list_index]

        # Continue if 
        if target_subjects != None and \
            not dataset.subject_id_list[list_index] in target_subjects:
            
            continue

        # If we have moved onto the next subject, we need to save and reinitialize
        if dataset.subject_id_list[list_index] != current_subject:
            # print('Completing subject {}'.format(current_subject))
            # Take the nan mean along the last axis
            volume_pred_array = np.concatenate(pred_volume_list, axis=-1)
            volume_pred_array = np.nanmean(volume_pred_array, axis=-1)
            # Verify that there are no nans left in the array anymore
            assert np.isnan(np.min(volume_pred_array)) == False, \
                '{} still contains nan values when trying to save'.format(
                    current_subject
                )

            # Save the array; we'll keep the raw values
            np.save(
                save_masks_dir / '{}.npy'.format(current_subject), 
                volume_pred_array
            )

            del pred_volume_list
            del volume_pred_array
            gc.collect()

            # print(current_subject, dataset.subject_id_list[list_index])

            # Now we can change the subject and create a new array
            current_subject = dataset.subject_id_list[list_index] 
            pred_volume_list = []

        # Get preds
        image = batch['image']
        mask = batch['mask']

        image = image.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)

        with torch.no_grad():
            pred = unet(image)

        # Sigmoid
        if n_classes == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = F.softmax(pred, dim=1)

        pred = torch.squeeze(pred, dim=0)

        resize_transform = tio.Resize((x_length, y_length, dataset.z_input_dim))
    
        # Turn into numpy array and fix dims
        pred = pred.cpu().detach().numpy()
        pred = resize_transform(pred)
        pred = np.squeeze(pred)
        pred = np.expand_dims(pred, axis=-1).astype(np.half)

        # Debugging
        # print('{}\t{}:{}\t{}:{}\t{}:{}'.format(
        #     i,
        #     x_index, x_index + dataset.input_dim,
        #     y_index, y_index + dataset.input_dim,
        #     z_index, z_index + dataset.input_dim
        # ))
    
        if n_classes == 1:
            current_pred_array = np.empty(
                (x_length, y_length, z_length, 1), dtype=np.half
            )
            current_pred_array[:] = np.nan

            current_pred_array[
                :,
                :,
                z_index:z_index + dataset.z_input_dim
            ] = pred
        else:
            current_pred_array = np.empty(
                (n_classes, x_length, y_length, z_length, 1), dtype=np.half
            )
            current_pred_array[:] = np.nan

            current_pred_array[
                :, 
                :,
                :,
                z_index:z_index + dataset.z_input_dim
            ] = pred

        # print(pred.dtype)

        pred_volume_list.append(current_pred_array)

    # Need to do it once more for the final subject
    # Take the nan mean along the last axis
    volume_pred_array = np.concatenate(pred_volume_list, axis=-1)
    volume_pred_array = np.nanmean(volume_pred_array, axis=-1)
    # Verify that there are no nans left in the array anymore
    assert np.isnan(np.min(volume_pred_array)) == False, \
        '{} still contains nan values when trying to save'.format(
            current_subject
        )

    # Save the array; we'll keep the raw values
    np.save(
        save_masks_dir / '{}.npy'.format(current_subject), 
        volume_pred_array
    )

def pred_and_save_masks_3d_simple(
    saved_model_path,
    dataset,
    unet,
    n_classes,
    n_channels,
    save_masks_dir,
    num_workers = 8
):

    """
    This function performs predictions on a 3D dataset using the simple
    dataset and saves the them as .npy 3D volumes for use later. 

    Parameters
    ----------
    unet: unet,
        Initialized model
    saved_model_path: str
        Path to saved model
    dataset: dataset.Dataset3DSimple
        Dataset used to serve 3D images that used to perform predictions
    n_classes: int
        Number of classes in output
    save_masks_dir: str
        Directory to save predicted masks
    num_workers: int
        Number of workers to use in DataLoaders
    target_subjects: [str, str, ...]
        List of subjects to perform predictions on

    """
    save_masks_dir = Path(save_masks_dir)
    assert os.path.isdir(save_masks_dir), \
        "{} directory does not exist".format(save_masks_dir)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = nn.DataParallel(unet)

    unet.load_state_dict(torch.load(saved_model_path))
    unet.to(device)
    unet.eval()

    # Dataset must be Dataset3DSimple since it spans across entire volume
    loader = DataLoader(
        dataset,
        batch_size = 1, 
        shuffle = False,
        num_workers = num_workers
    )

    print('Predicting Masks...')
    for i, batch in tqdm(enumerate(loader), total=len(loader)):

        current_subject = dataset.subject_id_list[i]
        # Make an empty array that will be filled in the correct area with preds
        x_length, y_length, z_length = dataset.image_shape_list[i]

        # Get preds
        image = batch['image']
        mask = batch['mask']

        image = image.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)

        with torch.no_grad():
            pred = unet(image)

        # Sigmoid
        if n_classes == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = F.softmax(pred, dim=1)

        pred = torch.squeeze(pred, dim=0)

        resize_transform = tio.Resize((x_length, y_length, z_length))
    
        # Turn into numpy array and fix dims
        pred = pred.cpu().detach().numpy()
        pred = resize_transform(pred)
        pred = np.squeeze(pred).astype(np.half)

        # Save the array; we'll keep the raw values
        np.save(
            save_masks_dir / '{}.npy'.format(current_subject), 
            pred
        )

        del pred
        gc.collect()


def eval_3d_volumes_breast(
    true_mask_dir,
    saved_preds_dir
):

    """
    Calculates BCE and DSC score for breast predictions on saved segemtations.

    Parameters
    ----------
    true_mask_dir: str,
        Directory containing saved true masks
    saved_preds_dir: str
        Directory containing saved predicted masks
    """

    true_mask_dir = Path(true_mask_dir)
    saved_preds_dir = Path(saved_preds_dir)

    # First make sure that we have an equal number of volumes in each dir
    subject_file_list = sorted([x for x in os.listdir(true_mask_dir) if '.npy' in x])

    assert subject_file_list == sorted(
        [x for x in os.listdir(saved_preds_dir) if '.npy' in x]
        ), "Mismatch between subjects in true mask dir and pred mask dir"

    # Now we can iterate through each subject
    # Sigmoid already applied so no logits needed
    criteron = nn.BCELoss()

    val_loss = 0
    dice_loss = 0

    for subject_file in tqdm(subject_file_list):
        true_mask = np.load(true_mask_dir / subject_file)
        pred_mask = np.load(saved_preds_dir / subject_file)

        true_mask = torch.from_numpy(true_mask).float()
        pred_mask = torch.from_numpy(pred_mask).float()

        if not pred_mask.is_contiguous():
            pred_mask = pred_mask.contiguous()

        assert true_mask.shape == pred_mask.shape, \
            """Subject: {}
            True mask and predict mask shape do not match: {}, {}"""\
                .format(
                    subject_file,
                    true_mask.shape, 
                    pred_mask.shape
                )

        loss = criteron(pred_mask, true_mask)

        # Threshold and get losses
        pred_mask = (pred_mask > 0.5).float()

        dice = dice_coeff(pred_mask, true_mask)
        dice_loss += dice.item()

        val_loss += loss.item()

    val_loss = val_loss/len(subject_file_list)
    dice_loss = dice_loss/len(subject_file_list)
    print('Breast: Val BCE Loss: {}; Dice Coeff: {}'.format(
        val_loss, dice_loss
    ))



def eval_3d_volumes_dv(
    true_mask_dir,
    saved_preds_dir
):

    """
    Calculates BCE and DSC score for FGT and blood vessel predictions on 
    saved segemtations.

    Parameters
    ----------
    true_mask_dir: str,
        Directory containing saved true masks
    saved_preds_dir: str
        Directory containing saved predicted masks
    """

    true_mask_dir = Path(true_mask_dir)
    saved_preds_dir = Path(saved_preds_dir)

    # First make sure that we have an equal number of volumes in each dir
    subject_file_list = sorted([x for x in os.listdir(true_mask_dir) if '.npy' in x])

    assert subject_file_list == sorted(
        [x for x in os.listdir(saved_preds_dir) if '.npy' in x]
        ), "Mismatch between subjects in true mask dir and pred mask dir"

    # Track losses and scores
    criteron = nn.CrossEntropyLoss()

    val_loss = 0
    vessel_dice_loss = 0
    dense_dice_loss = 0
    
     # Now we can iterate through each subject
    for subject_file in tqdm(subject_file_list):
        true_mask = np.load(true_mask_dir / subject_file)
        pred_mask = np.load(saved_preds_dir / subject_file)

        true_mask = torch.from_numpy(true_mask)
        pred_mask = torch.from_numpy(pred_mask).float()

        true_mask = F.one_hot(true_mask.long(), 3)
        true_mask = torch.permute(true_mask, (3, 0, 1, 2)).float()

        assert true_mask.shape == pred_mask.shape, \
            """Subject: {}
            True mask and predict mask shape do not match: {}, {}"""\
                .format(
                    subject_file,
                    true_mask.shape, 
                    pred_mask.shape
                )

        # Get CE loss
        loss = criteron(pred_mask, true_mask)

        # Now we need to break down things into pieces
        pred_mask = torch.argmax(pred_mask, dim=0, keepdim=True)

        # We need to create an array for vessels and dense
        vessel_pred = torch.clone(pred_mask)
        # Change all dense labels into background
        vessel_pred = torch.where(vessel_pred == 2, 0, vessel_pred).float()

        dense_pred = torch.clone(pred_mask)
        # Vice versa for dense
        dense_pred = torch.where(dense_pred == 1, 0, dense_pred)
        # Also change values into 1
        dense_pred = torch.where(dense_pred == 2, 1, dense_pred).float()

        # Split true masks into vessel and dense
        vessel_mask = true_mask[1:2, :, :, :]
        dense_mask = true_mask[2:, :, :, :]

        # print(vessel_pred.shape)
        # print(dense_pred.shape)
        # print(vessel_mask.shape)
        # print(dense_mask.shape)

        vessel_dice = dice_coeff(vessel_pred, vessel_mask)
        vessel_dice_loss += vessel_dice.item()

        dense_dice = dice_coeff(dense_pred, dense_mask)
        dense_dice_loss += dense_dice.item()

        val_loss += loss.item()

    val_loss = val_loss/len(subject_file_list)
    vessel_dice_loss = vessel_dice_loss/len(subject_file_list)
    dense_dice_loss = dense_dice_loss/len(subject_file_list)

    print("""Dense/Vessels: Val CE Loss: {}
    Vessels Dice Coeff: {}; Dense Dice Coeff: {}""".format(
        val_loss, vessel_dice_loss, dense_dice_loss
    ))
    
