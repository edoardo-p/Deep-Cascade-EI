import torch
import torch.nn as nn


class CCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        filters=64,
        # residual=True,
        # circular_padding=False,
        # cat=True,
        depth=3,
        convolutions=3,
    ):
        super(CCNN, self).__init__()
        self.name = "ccnn"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.net = self.build_net(convolutions, depth)

    def forward(self, x):
        return self.net(x)

    def build_net(self, convolutions, depth):
        blocks = []
        for _ in range(convolutions):
            blocks.append(self.conv_block(self.filters, depth))
            # blocks.append(self.dc_layer())

        return nn.Sequential(*blocks)

    def conv_block(self, filters, layers):
        channel_pairs = (
            [(self.in_channels, filters)]
            + [(filters, filters) for _ in range(layers - 2)]
            + [(filters, self.out_channels)]
        )
        cnn = []
        for f_in, f_out in channel_pairs:
            cnn.append(nn.Conv2d(f_in, f_out, kernel_size=3, padding=1))
            cnn.append(nn.ReLU())
        cnn.pop()  # removes the last ReLU layer
        return nn.Sequential(*cnn)

    # def dc_layer(self):
    #     torch.fft.fft()
