import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

from skimage import color

class GeneratorWithSkipConnections(nn.Module):
""" The GAN generator that has an U-Net architecture modelled according to:
        https://arxiv.org/abs/1505.04597
        More precisely it uses an encoder and decoder architecture, with skip connections.
        Also, GeneratorWithSkipConnections has filters which deacrese in size as the
        intermediary cs of the network gets smaller.
    """

    def __init__(self):
        super(GeneratorWithSkipConnections, self).__init__()

        self.createNetwork()

    def createNetwork(self):

        # Encoding layers
        self.conv_1 = nn.Sequential(
        nn.Conv2d(1, 64, 6, stride=2, padding=2, bias=False),
        nn.BatchNorm2d( 64 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_2 = nn.Sequential(
        nn.Conv2d(64, 128, 6, stride=2, padding=2, bias=False),
        nn.BatchNorm2d( 128 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_3 = nn.Sequential(
        nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d( 256 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_4 = nn.Sequential(
        nn.Conv2d(256, 512, 2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_5 = nn.Sequential(
        nn.Conv2d(512, 512, 2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.2 )
        )

        # Decoding layers

        self.conv_trans_1 = nn.Sequential(
        nn.ConvTranspose2d(512, 512, 2, stride=2, padding=0, c_padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        )

        self.conv_trans_2 = nn.Sequential(
        nn.ConvTranspose2d(1024, 256, 2, stride=2, padding=0, c_padding=0, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        )

        self.conv_trans_3 = nn.Sequential(
        nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, c_padding=0, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        )

        self.conv_trans_4 = nn.Sequential(
        nn.ConvTranspose2d(256, 64, 6, stride=2, padding=2, c_padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )

        self.conv_trans_5 = nn.Sequential(
        nn.ConvTranspose2d(128, 3, 6, stride=2, padding=2, c_padding=0, bias=False),
        nn.Tanh()
        )


    def forward(self, gray_scale_image):

        c1 = self.conv_1( gray_scale_image ) #batch_sizex64x128x128

        c2 = self.conv_2(c1) #batch_sizex128x64x64

        c3 = self.conv_3(c2) #batch_sizex256x32x32

        c4 = self.conv_4(c3) #batch_sizex512x16x16

        c5 = self.conv_5(c4) #batch_sizex512x8x8

        # Decoding

        c1_de = self.conv_trans_1(c5) #batch_sizex512x16x16

        skip1_de = torch.cat((c4, c1_de), 1) #batch_sizex1024x16x16

        c2_de = self.conv_trans_2(skip1_de) #batch_sizex256x32x32

        skip2_de = torch.cat((c3, c2_de), 1) #batch_sizex512x16x16

        c3_de = self.conv_trans_3(skip2_de) #batch_sizex128x64x64

        skip3_de = torch.cat((c2, c3_de), 1) #batch_sizex256x16x16

        c4_de = self.conv_trans_4(skip3_de) #batch_sizex64x128x128

        skip4_de = torch.cat((c1, c4_de), 1) #batch_sizex128x16x16

        c5_de = self.conv_trans_5(skip4_de) #batch_sizex128x64x64

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
