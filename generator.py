import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np


class VanillaGenerator(nn.Module):

    """ The vanilla generator is a generator without skip connections. The most significant difference compared
    to GeneratorWithSkipConnections is the skip connections as well as the filters used. The vanilla generator
    uses the same filter kernel size: 3x3 throughout the entire network, while GeneratorWithSkipConnections has
    filters which deacrese in size as the intermediary outputs of the network gets smaller.
    """

    def __init__(self):
        super(VanillaGenerator, self).__init__()

        self.createVanillaGenerator()

    def createVanillaGenerator(self):

        # Encoding layers

        self.conv_1 = nn.Sequential(

            # Strided Convolutions accoridng to: https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            # in channel, out channel, filter kernel size, stride, padding, bias
            nn.Conv2d(1, 80, 8, stride = 2, padding = 1, bias = False ),

            # Batch Normalization according to https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(80, 160, 8, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(160, 320, 6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.LeakyReLU(0.2)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(320, 640, 6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(640),
            nn.LeakyReLU(0.2)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(640, 640, 2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(640),
            nn.LeakyReLU(0.2)
        )

        # Decoding layers

        self.conv_trans_1 = nn.Sequential(

            # Transposed Strided Convolutions accoridng to: https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            nn.ConvTranspose2d(640, 640, 2, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(640),

            # ReLu accoridng to: https://arxiv.org/pdf/1511.06434.pdf?fbclid=IwAR2l2Otqnh-TYbrvHvlqV1V-a8-Tpep-_6xlH2aAuMT-__Y4-W1ppfwIhGs
            nn.ReLU(),
        )

        self.conv_trans_2 = nn.Sequential(
            nn.ConvTranspose2d(640, 320, 6, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(),
        )

        self.conv_trans_3 = nn.Sequential(
            nn.ConvTranspose2d(320, 160, 6, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )

        self.conv_trans_4 = nn.Sequential(
            nn.ConvTranspose2d(160, 80, 8, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )

        self.conv_trans_5 = nn.Sequential(
            nn.ConvTranspose2d(80, 3, 8, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, gray_image):
        """ The generators forward pass.
            Input: gray_image: 1 x 256 x 256
            Output: 2 x 256 x 256
        """

        # Encoding

        c1 = self.conv_1(gray_image)

        c2 = self.conv_2(c1)

        c3 = self.conv_3(c2)

        c4 = self.conv_4(c3)

        c5 = self.conv_5(c4)

        # Decoding

        c1_de = self.conv_trans_1(c5)

        c2_de = self.conv_trans_2(c1_de)

        c3_de = self.conv_trans_3(c2_de)

        c4_de = self.conv_trans_4(c3_de)

        c5_de = self.conv_trans_5(c4_de)

        return c5_de


if __name__ == "__main__":

    # Code for debugging. Used to look at the intermediary dimensions of an image
    # during the forwardpass where gray scale image is used as input in the generator
    # to colorize the gray scale image.

    # Create a generator
    generator = VanillaGenerator()

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
