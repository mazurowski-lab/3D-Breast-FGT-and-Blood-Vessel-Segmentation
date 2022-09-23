import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from pathlib import Path
import os
import numpy as np

from losses import DiceLoss, dice_coeff


def train_model(
    model,
    train_dataset,
    val_dataset,
    n_classes,
    n_channels,
    batch_size,
    learning_rate,
    epochs,
    model_save_dir,
    model_save_name,
    loss = 'cross',
    num_workers = 8,
    load_model_path = None
):
    """
    Trains a unet (or similar model)

    Parameters
    ----------
    model: torch.nn.Module
        Model that will be used for training
    train_dataset: Dataset2D or Dataset3D
        Dataset used to serve images that are used for training
    val_dataset: Dataset2D or Dataset3D
        Dataset used to serve images that are used for validation
    n_classes: int
        Number of classes in target segmentation
    n_channels: int
        Number of channels in input images
    batch_size: int
        Batch size
    learning_rate: float
        Learning rate
    epochs: int
        Number of epochs to train for
    model_save_dir: str
        Directory to save model
    model_save_name: str
        Name that will be used when saving model. Saved model will include .pth
    loss: str
        Loss function to use. Can either be 'dice' for dice-loss or 'cross' for CE
    num_workers: int
        Number of workers to use in DataLoaders
    load_model_path: str
        If included, then a model will be loaded and trained from then

    Returns
    -------
    unet.UNet
        Trained UNet

    """

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model_save_dir = Path(model_save_dir)
    assert os.path.isdir(model_save_dir), \
        "{} directory does not exist".format(model_save_dir)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size, 
        shuffle = True,
        num_workers = num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size, 
        shuffle = False,
        num_workers = num_workers
    )

    # Create network
    model = nn.DataParallel(model)
    model.to(device)

    if load_model_path:
        print('Using model with loaded weights from {}'.format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))

    # Define optimizer and criterion
    optimizer = Adam(model.parameters(), lr=learning_rate)

    data_type = torch.float32

    if n_classes == 1:
        history_array = np.array({
            'train_loss': [], 
            'val_loss': [], 
            'dce': []
        })
        if loss == 'cross':
            criterion = nn.BCEWithLogitsLoss()
        elif loss == 'dice':
            criterion = DiceLoss(normalization='sigmoid')
        else:
            print('Please use appropriate loss function')
            raise
    else:
        history_array = np.array({
            'train_loss': [], 
            'val_loss': []
        })
        if loss == 'cross':
            criterion = nn.CrossEntropyLoss()
        elif loss == 'dice':
            criterion = DiceLoss(normalization='softmax')
        else:
            print('Please use appropriate loss function')
            raise

    if n_classes == 1:
        best_score = 0
    else:   
        best_score = float('inf')

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # Training 
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader):
            images = batch['image']
            masks = batch['mask']

            images = images.to(device, dtype=data_type)
            masks = masks.to(device, dtype=data_type)

            # images = images.to(device)
            # masks = masks.to(device)
            
            preds = model(images)

            # print(masks)
            # print(preds)

            # print(masks.shape)
            # print(preds.shape)

            loss = criterion(preds, masks)

            epoch_loss += loss.item()

            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()

        val_loss = 0
        dice_score = 0
        
        for batch in val_loader:
            images = batch['image']
            masks = batch['mask']
            
            images = images.to(device, dtype=data_type)
            masks = masks.to(device, dtype=data_type)
            
            with torch.no_grad():
            
                preds = model(images)
                
                # For multi-class, we will report cross entropy. 
                # Otherwise, use dice and BCE

            if n_classes == 1:
                loss = criterion(preds, masks)
                
                sig_preds = torch.sigmoid(preds)
                sig_preds = (sig_preds > 0.5).float()
                dice = dice_coeff(sig_preds, masks)
                dice_score += dice.item()
            else:
                loss = criterion(preds, masks)
            
            val_loss += loss.item()

        val_loss = val_loss/len(val_loader)
                            
        if n_classes == 1:
            dice_score = dice_score/len(val_loader)
            print('Train BCE Loss: {}; Val BCE Loss: {}; Dice Coeff: {}'.format(
                train_loss, val_loss, dice_score
            ))
            history_array.item()['train_loss'].append(train_loss)
            history_array.item()['val_loss'].append(val_loss)
            history_array.item()['dce'].append(dice_score)
        else:
            print('Train CE Loss: {}; Val CE Loss: {}'.format(
                train_loss, val_loss))
            history_array.item()['train_loss'].append(train_loss)
            history_array.item()['val_loss'].append(val_loss)

        # Save the best model only
        # For one class, we can use dice score to determine this
        # Otherwise, cross entropy or other criterion. 

        if n_classes == 1:
            if dice_score > best_score:
                torch.save(
                    model.state_dict(), 
                    model_save_dir / (model_save_name + '.pth')
                )
                best_score = dice_score
        else:
            if val_loss < best_score:
                torch.save(
                    model.state_dict(), 
                    model_save_dir / (model_save_name + '.pth')
                )
                best_score = val_loss

        # Save npy object every epoch incase of a crash
        np.save(
            model_save_dir /  '{}_history'.format(model_save_name), 
            history_array
        )
    
    print('Training complete. {} epochs completed.'.format(epochs))
    print('Best {} score: {}'.format(
        'dice' if n_channels == 1 else 'cross entropy',
        best_score
    ))

    return model