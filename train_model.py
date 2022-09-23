import argparse

# Trains a new model using the same method described in our publication. 
# We used a batch size of 16 and 20 epochs compared to the default params here.

def get_args():
    parser = argparse.ArgumentParser(
        description='Train UNet Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-c', '--target-tissue', metavar='C', type=str,
        help='Target tissue, either breast or dv', dest='target_tissue'
    )

    parser.add_argument(
        '-t', '--train-image', metavar='T', type=str,
        help='Directory of training images', dest='train_image_dir'
    )
    parser.add_argument(
        '-v', '--val-image', metavar='V', type=str,
        help='Directory of val images', dest='val_image_dir'
    )

    parser.add_argument(
        '-m', '--train-mask', metavar='M', type=str,
        help='Directory of training masks', dest='train_mask_dir'
    )
    parser.add_argument(
        '-n', '--val-mask', metavar='N', type=str,
        help='Directory of val masks', dest='val_mask_dir'
    )

    parser.add_argument(
        '-i', '--train-input-mask', metavar='I', type=str,
        help='Directory of training input masks', dest='train_input_mask_dir'
    )
    parser.add_argument(
        '-j', '--val-input-mask', metavar='J', type=str,
        help='Directory of val input masks', dest='val_input_mask_dir'
    )

    parser.add_argument(
        '-e', '--epochs', metavar='E', type=int, default=10,
        help='Epochs', dest='epochs'
    )
    parser.add_argument(
        '-b', '--batch-size', metavar='B', type=int, default=8,
        help='Batch size', dest='batch_size'
    )

    parser.add_argument(
        '-s', '--model-save-dir', metavar='S', type=str,
        help='Directory to save model', dest='model_save_dir'
    )
    parser.add_argument(
        '-u', '--model-save-name', metavar='U', type=str,
        help='Name to save model as', dest='model_save_name'
    )

    parser.add_argument(
        '-d', '--dataset-type', metavar='D', type=str,
        help='Dataset Type (random or simple)', dest='dataset_type'
    )
    return parser.parse_args()


if __name__ == '__main__':
    from dataset_3d import *
    from train import train_model
    from unet import UNet3D

    args = get_args()

    if args.target_tissue == 'breast':
        n_channels = 1
        n_classes = 1
    elif args.target_tissue == 'dv':
        n_channels = 2
        n_classes = 3
    else:
        print('Target tissue must either be breast or dv')
        raise

    unet = UNet3D(
        in_channels = n_channels, 
        out_classes = n_classes,
        num_encoding_blocks = 3,
        padding = True,
        normalization = 'batch'
    )

    if args.target_tissue == 'breast':

        input_dim = (144, 144, 96)

        train_transforms = tio.Compose([
            tio.Resize(input_dim),
            tio.RandomBiasField(),
            tio.RandomMotion(degrees=5, translation=5, num_transforms=2),
            tio.RandomNoise(mean=0, std=(0.1, 0.5))
        ])

        val_transforms = tio.Compose([
            tio.Resize(input_dim)
        ])

        train_dataset = Dataset3DSimple(
            image_dir = args.train_image_dir,
            mask_dir = args.train_mask_dir,
            transforms = train_transforms
        )

        val_dataset = Dataset3DSimple(
            image_dir = args.val_image_dir,
            mask_dir = args.val_mask_dir,
            transforms = val_transforms
        )

        trained_unet = train_model(
            model = unet,
            train_dataset = train_dataset,
            val_dataset = val_dataset,
            n_classes = n_channels,
            n_channels = n_classes,
            batch_size = args.batch_size,
            learning_rate = 3e-4,
            epochs = args.epochs,
            model_save_dir = args.model_save_dir,
            model_save_name = args.model_save_name,
            num_workers = 8,
            loss='cross',
        )

    else:

        train_transforms = tio.Compose([
            tio.RandomNoise(mean=0, std=(0.1, 0.5))
        ])

        val_transforms = tio.Compose([
        ])

        train_dataset = Dataset3DRandom(
            image_dir = args.train_image_dir,
            mask_dir = args.train_mask_dir,
            additional_input_dir = args.train_input_mask_dir,
            input_dim = 96,
            total_samples = 20000,
            transforms = train_transforms,
            one_hot_mask = True
        )

        val_dataset = Dataset3DRandom(
            image_dir = args.val_image_dir,
            mask_dir = args.val_mask_dir,
            additional_input_dir = args.val_input_mask_dir,
            input_dim = 96,
            total_samples = 4000,
            transforms = val_transforms,
            one_hot_mask = True
        )

        trained_unet = train_model(
            model = unet,
            train_dataset = train_dataset,
            val_dataset = val_dataset,
            n_classes = n_channels,
            n_channels = n_classes,
            batch_size = args.batch_size,
            learning_rate = 3e-4,
            epochs = args.epochs,
            model_save_dir = args.model_save_dir,
            model_save_name = args.model_save_name,
            num_workers = 8,
            loss='dice',
        )