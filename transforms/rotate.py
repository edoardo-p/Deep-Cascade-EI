import random

import kornia as dgm
import numpy as np
import torch


class Rotate:
    def __init__(self, n_trans, random_rotate=True):
        self.n_trans = n_trans
        self.random_rotate = random_rotate

    def apply(self, x):
        return rotate_dgm(x, self.n_trans, self.random_rotate)


def rotate_dgm(data, n_trans=5, random_rotate=False):
    """Rotates the data n_trans times"""
    if random_rotate:
        theta_list = random.sample(list(np.arange(1, 359)), n_trans)
    else:
        theta_list = np.arange(10, 360, int(360 / n_trans))

    data = torch.cat(
        [
            data
            if theta == 0
            else dgm.geometry.rotate(data, torch.tensor([float(theta)]))
            for theta in theta_list
        ],
        dim=0,
    )
    return data
