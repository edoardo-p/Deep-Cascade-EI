import scipy.io as scio
import torch
from torch.utils.data.dataset import Dataset


class CTData(Dataset):
    def __init__(self, mode="train", root_dir=".\\dataset\\CT100_128x128.mat"):
        """
        The original CT100 dataset can be downloaded from
        https://www.kaggle.com/kmader/siim-medical-images
        the images have been resized and saved in Matlab.
        """

        mat_data = scio.loadmat(root_dir)
        x = torch.from_numpy(mat_data["DATA"])

        if mode == "train":
            self.x = x[0:90]
        if mode == "test":
            self.x = x[90:100, ...]

        self.x = self.x.type(torch.FloatTensor)

    def __getitem__(self, index):
        x = self.x[index]
        return x

    def __len__(self):
        return len(self.x)
