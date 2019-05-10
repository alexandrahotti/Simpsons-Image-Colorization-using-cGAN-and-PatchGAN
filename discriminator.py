import torch
#import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

class Discriminator(nn.Module):
    """ The GAN generator that uses a encoder decoder architecture
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        #här
        self.conv_1 = nn.Sequential(
        nn.Conv2d(3, 64, 6, stride=2, padding=2, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 64 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_2 = nn.Sequential(
        nn.Conv2d(64, 128, 6, stride=2, padding=2, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 128 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_3 = nn.Sequential(
        nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 256 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_4 = nn.Sequential(
        nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_5 = nn.Sequential(
        nn.Conv2d(512, 1, 3, stride=1, padding=1, bias=False), # in channel, out channel, filter kernel size

        nn.Sigmoid()
        )





    def forward(self, data):
        #
        # print('data.size()')
        # print(data.size())
        output1 = self.conv_1(data)
        # print(output1.size())

        output2 = self.conv_2(output1)
        # print(output2.size())

        output3 = self.conv_3(output2)
        # print(output3.size())

        output4 = self.conv_4(output3)
        # print(output4.size())


        output5 = self.conv_5(output4)
        # print(output5.size())

        # output6 = self.conv_6(output5)
        # print(output6.size())
        #
        # output7 = self.conv_7(output6)
        # print(output7.size())
        #
        # output8 = self.conv_8(output7)
        # print(output8.size())

        # Decoding

        # print(output5.size())

        return output5

if __name__ == "__main__":
    discriminator = Discriminator()
    #gan = GeneratorWithSkipConnections()
    #hl_graph = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
    #hl_graph = hl.build_graph(gan, torch.zeros([1, 1, 256, 256]))

    image_array = imread('TheSimpsonsS10E01LardoftheDance.mp40060.jpg')
    image_array = np.matrix.transpose(image_array)
    image_array = torch.tensor([image_array]) # nu int 32
    image_array = image_array.type('torch.FloatTensor')
    #generated_im = gan.forward(image_array)
    output = discriminator(image_array)
