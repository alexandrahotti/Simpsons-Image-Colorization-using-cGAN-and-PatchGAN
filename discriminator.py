import torch
#import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

class Discriminator(nn.Module):
    """ The GAN discriminator.
    """

    def __init__(self):
        super(Discriminator, self).__init__()


        self.conv_1 = nn.Sequential(
        nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d( 64 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_2 = nn.Sequential(
        nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d( 128 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_3 = nn.Sequential(
        nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d( 256 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_4 = nn.Sequential(
        nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_5 = nn.Sequential(
        nn.Conv2d(512, 1, 3, stride=1, padding=1, bias=False),
        nn.Sigmoid()
        )



    def forward(self, data):

        c1 = self.conv_1(data)

        c2 = self.conv_2(c1)

        c3 = self.conv_3(c2)

        c4 = self.conv_4(c3)

        c5 = self.conv_5(c4)

        return c5

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
