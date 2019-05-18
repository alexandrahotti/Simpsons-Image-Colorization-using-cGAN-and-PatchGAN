import torch
import torch.nn as nn


# Packages used for debugging
from scipy.misc import imread
import numpy as np


class Discriminator(nn.Module):
    """ The Discriminator used in the PatchGan
    """

    def __init__(self):

        super(Discriminator, self).__init__()

        # The first 2 dimensional convolutional layer where batch normalization is applied
        # on the convolutional output before leaky reLu is used as an activation function.
        self.conv_1 = nn.Sequential(
            # in channels, out channels, filter kernel size, stride, padding, bias
            nn.Conv2d(3, 64, 6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # At the last layer batch normalization is not used and
        # sigmoid is applied to the convolutional output.
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1, 3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, three_channel_image):
        """ The discriminators forward pass.
            Input: three_channel_image: 3 x 256 x 256
            Output: 1 x 32 x 32
        """

        # h1: 64 x 128 x 128
        h1 = self.conv_1(three_channel_image)

        # h2: 128 x 64 x 64
        h2 = self.conv_2(h1)

        # h3: 256 x 32 x 32
        h3 = self.conv_3(h2)

        # h4: 512 x 32 x 32
        h4 = self.conv_4(h3)

        # h5: 1 x 32 x 32
        h5 = self.conv_5(h4)

        return h5


if __name__ == "__main__":

    # Code for debugging. Used to look at the intermediary dimensions of an image
    #during the forward pass

    # Create a discriminator
    discriminator = Discriminator()

    image_name = 'TheSimpsonsS10E01LardoftheDance.mp40005.jpg'
    image = imread(image_name)

    #Image preprocessing
    image = np.matrix.transpose(image)
    image = torch.tensor([image]).type('torch.FloatTensor')

    # Forward pass
    output = discriminator(image)
