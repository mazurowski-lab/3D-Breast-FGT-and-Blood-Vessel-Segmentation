import argparse

# Evaluates the dice score of segmentations given directories with saved
# segmentations in npy format. 

def get_args():
    parser = argparse.ArgumentParser(
        description='Train UNet Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-c', '--target-tissue', metavar='C', type=str,
        help='Target tissue, either breast or dv', dest='target_tissue'
    )

    parser.add_argument(
        '-t', '--true-mask', metavar='p', type=str,
        help='Directory of true masks', dest='true_mask_dir'
    )
    parser.add_argument(
        '-p', '--predicted-mask', metavar='P', type=str,
        help='Directory of predicted masks', dest='predicted_mask_dir'
    )

    return parser.parse_args()


if __name__ == '__main__':
    from model_utils import eval_3d_volumes_breast, eval_3d_volumes_dv

    args = get_args()

    if args.target_tissue == 'breast':
        eval_3d_volumes_breast(
            args.true_mask_dir,
            args.predicted_mask_dir
        )
    elif args.target_tissue == 'dv':
        eval_3d_volumes_dv(
            args.true_mask_dir,
            args.predicted_mask_dir
        )
    else:
        print('Target tissue must either be breast or dv')
        raise
