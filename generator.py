import torch
import torch.nn as nn


import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

class Generator(nn.Module):
    """ The GAN generator that uses a encoder decoder architecture
    """

    def __init__(self):
        super(Generator, self).__init__()

        # Encoding layers
        self.conv_1 = nn.Sequential(
        nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 64 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_2 = nn.Sequential(
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 128 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_3 = nn.Sequential(
        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 256 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_4 = nn.Sequential(
        nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_5 = nn.Sequential(
        nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.1 )
        )


        # Decoding layers
        self.conv_trans_1 = nn.Sequential(
        nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        )

        self.conv_trans_2 = nn.Sequential(
        nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        )

        self.conv_trans_3 = nn.Sequential(
        nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        )

        self.conv_trans_4 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )

        self.conv_trans_5 = nn.Sequential(
        nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1, bias=False),
        #nn.BatchNorm2d(3),
        nn.Tanh()
        )
        # Tanh at last layer. Source: https://github.com/soumith/ganhacks


    def forward(self, data):


        output1 = self.conv_1(data)


        output2 = self.conv_2(output1)

        output3 = self.conv_3(output2)

        output4 = self.conv_4(output3)

        output5 = self.conv_5(output4)

        # Decoding
        
        output1_de = self.conv_trans_1(output5)

        output2_de = self.conv_trans_2(output1_de)


        output3_de = self.conv_trans_3(output2_de)

        output4_de = self.conv_trans_4(output3_de)

        output5_de = self.conv_trans_5(output4_de)

        return output5_de




if __name__ == "__main__":
    gan = Generator()

    image_array = imread('test_im.jpg')
    image_array = np.matrix.transpose(image_array)
    image_array = torch.tensor([[image_array]]) # nu int 32
    image_array = image_array.type('torch.FloatTensor')
    generated_im = gan.forward_pass(image_array)
    generated = generated_im.data.numpy()
    generated = generated[0,:,:,:]
    generated = np.round((generated + 1) * 255 / 2)
    generated =generated.astype(int)

    # print(generated.transpose())
    # plt.show(plt.imshow( generated.transpose() ))
