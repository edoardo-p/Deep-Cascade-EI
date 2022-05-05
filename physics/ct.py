import torch
import numpy as np
from .radon.radon import Radon, IRadon


class CT:
    def __init__(self, img_width, radon_view, uniform=True, circle=False):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta, circle)
        self.iradon = IRadon(img_width, theta, circle)

    def A(self, x):
        return self.radon(x)

    def A_dagger(self, y):
        return self.iradon(y)
