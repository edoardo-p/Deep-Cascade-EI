import math
import os

import torch
from models.ccnn import CCNN
from torch.optim import Adam
from utils.logger import Logger, get_timestamp

from .closure import closure_ei


def adjust_learning_rate(schedule, epoch, epochs, lr):
    """Decay the learning rate based on schedule"""
    if schedule == "cos":  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    elif isinstance(schedule, list):  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    else:
        raise ValueError('Schedule must either be "cos" or a list of integers')

    return lr


class EI(object):
    def __init__(self, in_channels, out_channels, img_width, img_height, dtype):
        super(EI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_width = img_width
        self.img_height = img_height
        self.dtype = dtype

    def train_ei(
        self,
        dataloader,
        physics,
        transform,
        epochs,
        lr,
        alpha,
        ckp_interval,
        schedule,
        loss_type="l2",
        report_psnr=False,
    ):
        save_path = f"./ckp/{get_timestamp()}_ei"

        os.makedirs(save_path, exist_ok=True)

        generator = CCNN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            filters=64,
            depth=2,
            convolutions=2,
        )

        losses = {
            "l1": torch.nn.L1Loss(),
            "l2": torch.nn.MSELoss(),
        }

        optimizer = Adam(generator.parameters(), lr=lr["G"], weight_decay=lr["WD"])

        field_names = ["epoch", "loss_mc", "loss_ei", "loss_total"]

        if report_psnr:
            field_names.append("psnr")
            field_names.append("mse")

        log = Logger(
            save_path,
            filename="training_loss",
            field_names=field_names,
        )

        for epoch in range(1, epochs + 1):
            lr["G"] = adjust_learning_rate(schedule, epoch, epochs, lr["G"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr["G"]

            loss = closure_ei(
                generator,
                dataloader,
                physics,
                transform,
                optimizer,
                losses[loss_type],
                losses[loss_type],
                alpha,
                self.dtype,
                report_psnr,
            )

            log.record(epoch, *loss)
            print(f"{get_timestamp()}\tEpoch {epoch}/{epochs}", end="\t")
            for key, val in loss:
                print(f"{key}={val}", end="\t")

            if epoch % ckp_interval == 0 or epoch == epochs:
                state = {
                    "epoch": epoch,
                    "state_dict": generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(save_path, f"ckp_{epoch}.pth.tar"))
        log.close()
