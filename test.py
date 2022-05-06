import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset.ctdb import CTData
from models.ccnn import CCNN
from physics.ct import CT
from utils.metric import calc_psnr


parser = argparse.ArgumentParser(
    description="Deep Cascade CNN with EI test parameters."
)

parser.add_argument(
    "--sample-to-show",
    default=[9],
    nargs="*",
    type=int,
    help="the test sample id for visualization" "default [9]",
)
parser.add_argument(
    "--ckp",
    default="./ckp/ckp_final.pth.tar",
    type=str,
    metavar="PATH",
    help="path to checkpoint of a trained model",
)
parser.add_argument(
    "--model-name",
    default="CCNN with EI",
    type=str,
    help="name of the trained model (default: 'CCNN with EI')",
)
parser.add_argument(
    "--dataset",
    default="./dataset/CT100_128x128.mat",
    type=str,
    metavar="PATH",
    help="path to the dataset MATLAB file (default: ./dataset/CT100_128x128.mat)",
)


def main():
    args = parser.parse_args()

    cnn_net = CCNN(
        masks=None,
        in_channels=1,
        out_channels=1,
        filters=64,
        depth=2,
        convolutions=2,
    )
    forw = CT(img_width=128, radon_view=50, circle=False)
    dataset = CTData(mode="test", root_dir=args.dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    checkpoint = torch.load(args.ckp)
    cnn_net.load_state_dict(checkpoint["state_dict"])
    cnn_net.eval()

    def test(net, fbp):
        return net(fbp)

    for i, x in enumerate(dataloader):
        if i in args.sample_to_show:
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            x = x.type(torch.float)

            y = forw.A(x)
            fbp = forw.A_dagger(y)
            x_hat = test(cnn_net, fbp)

            plt.subplot(1, 4, 1)
            plt.imshow(y[0].detach().permute(1, 2, 0).numpy())
            plt.title("y")

            plt.subplot(1, 4, 2)
            plt.imshow(fbp[0].detach().permute(1, 2, 0).numpy())
            plt.title(f"FBP ({calc_psnr(x, fbp):.2f})")

            plt.subplot(1, 4, 1)
            plt.imshow(x_hat[0].detach().permute(1, 2, 0).numpy())
            plt.title(f"{args.model_name} (({calc_psnr(x, fbp):.2f}))")

            plt.subplot(1, 4, 2)
            plt.imshow(x[0].detach().permute(1, 2, 0).numpy())
            plt.title("x")

            ax = plt.gca()
            ax.set_xticks([]), ax.set_yticks([])
            plt.subplots_adjust(
                left=0.1, bottom=0.1, top=0.9, right=0.9, hspace=0.02, wspace=0.02
            )
            plt.show()


if __name__ == "__main__":
    main()
