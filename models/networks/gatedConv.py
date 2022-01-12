from .custom_layers import EqualizedConv2d
import torch
import torch.nn as nn


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=0,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True),
                 equalized=True,
                 initBiasToZero=True):

        super(GatedConv2dWithActivation, self).__init__()
        self.activation = activation
        self.equalizedlR = equalized
        self.initBiasToZero = initBiasToZero
        self.conv2d = EqualizedConv2d(in_channels, out_channels, kernel_size, padding, equalized=self.equalizedlR,
            initBiasToZero=self.initBiasToZero)
        self.mask_conv2d = EqualizedConv2d(in_channels, out_channels, kernel_size, padding, equalized=self.equalizedlR,
            initBiasToZero=self.initBiasToZero)
        self.sigmoid = torch.nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):

        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        return x