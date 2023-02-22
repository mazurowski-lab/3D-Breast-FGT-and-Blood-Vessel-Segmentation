import argparse
from pathlib import Path

import daiquiri
import torchio as tio
from unet import UNet3D

from dataset_3d import Dataset3DSimple, Dataset3DDivided
from model_utils import pred_and_save_masks_3d_simple, pred_and_save_masks_3d_divided

logger = daiquiri.getLogger(__name__)


# Performs predictions using a trained model.
# Predictions are performed the same method we used and are saved to a
# target directory. 

def get_args():
    parser = argparse.ArgumentParser(
        description='Train UNet Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-c', '--target-tissue', metavar='C', type=str,
        help='Target tissue, either breast or dv', dest='target_tissue'
    )

    parser.add_argument(
        '-i', '--image', metavar='T', type=str,
        help='Directory of images', dest='image_dir'
    )
    parser.add_argument(
        '-m', '--input-mask', metavar='M', type=str,
        help='Directory of input masks', dest='input_mask_dir'
    )

    parser.add_argument(
        '-s', '--save-masks-dir', metavar='S', type=str,
        help='Directory to save masks', dest='save_masks_dir'
    )
    parser.add_argument(
        '-p', '--model-save-path', metavar='P', type=str,
        help='Path to saved model', dest='model_save_path'
    )

    return parser.parse_args()


def run(target_tissue, image_dir, input_mask_dir, save_masks_dir, max_count=None):
    model_dirpath = Path(__file__).parent / "trained_models"

    if target_tissue == 'breast':
        model_save_path = model_dirpath / "breast_model.pth"
        n_channels = 1
        n_classes = 1
    elif target_tissue == 'dv':
        model_save_path = model_dirpath / "dv_model.pth"
        n_channels = 2
        n_classes = 3
    else:
        raise ValueError('Target tissue must either be breast or dv')

    unet = UNet3D(
        in_channels = n_channels,
        out_classes = n_classes,
        num_encoding_blocks = 3,
        padding = True,
        normalization = 'batch'
    )

    if target_tissue == 'breast':

        input_dim = (144, 144, 96)

        transforms = tio.Compose([
            tio.Resize(input_dim)
        ])

        dataset = Dataset3DSimple(
            image_dir = image_dir,
            mask_dir = None,
            transforms = transforms,
            image_only = True,
            save_masks_dir=save_masks_dir,
            max_count=max_count,
        )
        logger.info("Inferring Breast mask.")

        pred_and_save_masks_3d_simple(
            saved_model_path=model_save_path,
            dataset=dataset,
            unet=unet,
            n_classes=n_classes,
            n_channels=n_channels,
            save_masks_dir=save_masks_dir,
            num_workers=0,  # Multiprocessing sometimes reveals unpickling error.
        )
    else:

        x_y_divisions = 8
        z_division = 3

        transforms = tio.Compose([
        ])

        dataset = Dataset3DDivided(
            image_dir = image_dir,
            mask_dir = input_mask_dir,
            additional_input_dir = input_mask_dir,
            input_dim = 96,
            x_y_divisions = x_y_divisions,
            z_division = z_division,
            transforms = transforms,
            one_hot_mask = True,
            image_only = False,
            max_count = max_count,

        )
        logger.info("Inferring FGT & DV mask.")

        pred_and_save_masks_3d_divided(
            saved_model_path=model_save_path,
            dataset=dataset,
            unet=unet,
            n_classes=n_classes,
            # n_channels=n_channels,
            save_masks_dir=save_masks_dir,
        )
    logger.info("Masks saved at %s", save_masks_dir)
    return save_masks_dir


if __name__ == '__main__':
    args = get_args()

    run(
        target_tissue=args.target_tissue,
        image_dir=args.image_dir,
        input_mask_dir=args.input_mask_dir,
        model_save_path=args.model_save_path,
        save_masks_dir=args.save_masks_dir,
    )