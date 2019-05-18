import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np


class VanillaGenerator(nn.Module):

    """ The vanilla generator is a generator without skip connections. The most
    significant difference compared to GeneratorWithSkipConnections
    is the skip connections as well as the filters used. The vanilla generator
    uses the same filter kernel size: 3x3 throughout the entire network, while
    GeneratorWithSkipConnections has filters which
    deacrese in size as the intermediary outputs of the network gets smaller.
    """

    def __init__(self):
        super(VanillaGenerator, self).__init__()

        self.createVanillaGenerator()

    def createVanillaGenerator(self):

        # Encoding layers

        self.conv_1 = nn.Sequential(

            # Strided Convolutions accoridng to: https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            # in channel, out channel, filter kernel size, stride, padding, bias
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),

            # Batch Normalization according to https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # Decoding layers

        self.conv_trans_1 = nn.Sequential(

            # Transposed Strided Convolutions accoridng to: https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            nn.ConvTranspose2d(512, 512, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(512),

            # ReLu accoridng to: https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            nn.ReLU(),
        )

        self.conv_trans_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_trans_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_trans_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_trans_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, gray_image):
        """ The generators forward pass.
            Input: gray_image: 1 x 256 x 256
            Output: 2 x 256 x 256
        """

        # Encoding

        # batch size x 64 x 128 x 128
        c1 = self.conv_1(gray_image)

        # batch size x 128 x 64 x 64
        c2 = self.conv_2(c1)

        # batch size x 256 x 32 x 32
        c3 = self.conv_3(c2)

        # batch size x 512 x 16 x 16
        c4 = self.conv_4(c3)

        # batch size x 512 x 8 x 8
        c5 = self.conv_5(c4)

        # Decoding

        # batch size x 512 x 16 x 16
        c1_de = self.conv_trans_1(c5)

        # batch size x 1024 x 16 x 16
        skip1_de = torch.cat((c4, c1_de), 1)

        # batch size x 256 x 32 x 32
        c2_de = self.conv_trans_2(c1_de)

        # batch size x 128 x 64 x 64
        c3_de = self.conv_trans_3(c2_de)

        # batch size x 64 x 128 x 128
        c4_de = self.conv_trans_4(c3_de)

        # batch size x 128 x 64 x 64
        c5_de = self.conv_trans_5(c4_de)

        return c5_de


class GeneratorWithSkipConnections(nn.Module):

    """ The GAN generator that has an U-Net architecture modelled according to:
        https://arxiv.org/abs/1505.04597.
        More precisely it uses an encoder and decoder architecture, with skip connections.
        Also, GeneratorWithSkipConnections has filters which deacrese in size as the
        intermediary outputs of the network gets smaller.
    """

    def __init__(self):
        super(GeneratorWithSkipConnections, self).__init__()

        self.createNetwork()

    def createNetwork(self):

        # Encoding layers
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 6, stride=2, padding=2, bias=False),
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
            nn.Conv2d(256, 512, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # Decoding layers

        self.conv_trans_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 2, stride=2,
                               padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv_trans_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 2, stride=2,
                               padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_trans_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2,
                               padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_trans_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 6, stride=2, padding=2,
                               output_padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_trans_5 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 6, stride=2, padding=2,
                               output_padding=0, bias=False),
            # Tanh accoridng to: https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            nn.Tanh()
        )

    def forward(self, gray_image):
        """ The forward pass of the generator with skip connections
        """

        # batch size x 64 x 128 x 128
        c1 = self.conv_1( gray_image )

        # batch size x 128 x 64 x 64
        c2 = self.conv_2(c1)

        # batch size x 256 x 32 x 32
        c3 = self.conv_3(c2)

        # batch size x 512 x 16 x 16
        c4 = self.conv_4(c3)

        # batch size x 512 x 8 x 8
        c5 = self.conv_5(c4)

        # Decoding

        # batch size x 512 x 16 x 16
        c1_de = self.conv_trans_1(c5)

        # batch size x 1024 x 16 x 16
        skip1_de = torch.cat((c4, c1_de), 1)

        # batch size x 256 x 32 x 32
        c2_de = self.conv_trans_2(skip1_de)

        # batch size x 256 x 32 x 32
        skip2_de = torch.cat((c3, c2_de), 1)

        # batch size x 128 x 64 x 64
        c3_de = self.conv_trans_3(skip2_de)

        # batch size x 128 x 64 x 64
        skip3_de = torch.cat((c2, c3_de), 1)

        # batch size x 64 x 128 x 128
        c4_de = self.conv_trans_4(skip3_de)

        # batch size x 64 x 128 x 128
        skip4_de = torch.cat((c1, c4_de), 1)

        # batch size x 3 x 256 x 256
        c5_de = self.conv_trans_5(skip4_de)

        return c5_de


if __name__ == "__main__":

    # Code for debugging. Used to look at the intermediary dimensions of an image
    # during the forwardpass where gray scale image is used as input in the generator
    # to colorize the gray scale image.

    # Create a generator
    generator = GeneratorWithSkipConnections()

    image_name = 'TheSimpsonsS10E01LardoftheDance.mp40005_gray.jpg'
    image = imread(image_name)

    # Image preprocessing
    image = np.matrix.transpose(image)
    image = torch.tensor([[image]]).type('torch.FloatTensor')

    # Forward pass
    colorizes_image = generator(image)

    # Convert tensor into numpy vector
    colorizes_image = colorizes_image.data.numpy()

    # Extract the first image in the batch
    colorizes_image = colorizes_image[0, :, :, :]

    # change the range of the values from [-1, 1] to RGB [0, 255]
    colorizes_image = np.round((colorizes_image + 1) * 255 / 2)

    # Round the values to integers
    colorizes_image = colorizes_image.astype(int)

    # Plot the generated image
    plt.imshow(colorizes_image.transpose())
    plt.axis('off')
    plt.show()
