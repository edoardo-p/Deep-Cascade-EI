import torch
from torch.utils.data import DataLoader
import argparse

from ei.ei import EI
from physics.ct import CT
from dataset.ctdb import CTData
from transforms.rotate import Rotate

parser = argparse.ArgumentParser(description="EI experiment parameters.")

parser.add_argument(
    "--schedule",
    default=[2000, 3000, 4000],
    nargs="+",
    type=int,
    help="learning rate schedule (when to drop lr by 10x),"
    "(default: [2000, 3000, 4000])",
)
parser.add_argument(
    "--cos", default=False, action="store_true", help="use cosine lr schedule"
)
parser.add_argument(
    "--epochs",
    default=5000,
    type=int,
    metavar="N",
    help="number of total epochs to run " "(default: 5000)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=5e-4,
    type=float,
    metavar="LR",
    help="initial learning rate " "(default: 5e-4)",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-8,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-8)",
    dest="weight_decay",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=2,
    type=int,
    metavar="N",
    help="mini-batch size (default: 2)",
)
parser.add_argument(
    "--ckp-interval",
    default=500,
    type=int,
    help="save checkpoints interval epochs (default: 1000)",
)
parser.add_argument(
    "--dataset",
    default="./dataset/CT100_128x128.mat",
    type=str,
    metavar="PATH",
    help="path to the dataset MATLAB file (default: ./dataset/CT100_128x128.mat)",
)

# EI specific configs:
parser.add_argument(
    "--ei-trans",
    default=5,
    type=int,
    help="number of transformations for EI (default: 5)",
)
parser.add_argument(
    "--ei-alpha",
    default=100,
    type=float,
    help="equivariance strength (default: 100)",
)

# Inverse problem task configs:
parser.add_argument(
    "--views",
    default=50,
    type=int,
    help="number of radon views (default: 50)",
)


def main():
    args = parser.parse_args()

    lr = {"G": args.lr, "WD": args.weight_decay}

    dataset = CTData(mode="train", root_dir=args.dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    # forward model A
    physics = CT(img_width=128, radon_view=args.ct_views, circle=False)
    # transformations group G (used in Equivariant Imaging)
    transform = Rotate(n_trans=args.ei_trans)
    # define Equivariant Imaging model
    ei = EI(
        in_channels=1,
        out_channels=1,
        img_width=128,
        img_height=128,
        dtype=torch.float,
    )

    ei.train_ei(
        dataloader,
        physics,
        transform,
        args.epochs,
        lr,
        args.ei_alpha,
        args.ckp_interval,
        schedule="cos" if args.cos else args.schedule,
        loss_type="l2",
        report_psnr=True,
    )


if __name__ == "__main__":
    main()
