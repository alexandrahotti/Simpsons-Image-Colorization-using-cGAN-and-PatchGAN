import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

from skimage import color

class Generator(nn.Module):
    """ The GAN generator that uses a encoder decoder architecture
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.createVanillaGan()

    def createVanillaGan(self):

        #this was cut from teh constructor

        # Encoding layers
        self.conv_1 = nn.Sequential(
        nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 64 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_2 = nn.Sequential(
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 128 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_3 = nn.Sequential(
        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 256 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_4 = nn.Sequential(
        nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_5 = nn.Sequential(
        nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.2 )
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
        #print('data.size()')
        #print(data.size())


        output1 = self.conv_1(data) #10x64x128x128

        #print(output1.size())
        #print(type(output1))

        output2 = self.conv_2(output1) #10x128x64x64
        #print(output2.size())


        output3 = self.conv_3(output2) #10x256x32x32
        #print(output3.size())

        output4 = self.conv_4(output3) #10x512x16x16
        #print(output4.size())

        output5 = self.conv_5(output4) #10x512x8x8
        #print(output5.size())

        # Decoding
        #print("decoding")

        output1_de = self.conv_trans_1(output5) #10x512x16x16
        #print(output1_de.size())

        skip1_de = torch.cat((output4, output1_de), 1) #10x1024x16x16

        output2_de = self.conv_trans_2(output1_de) #10x256x32x32
        #print(output2_de.size())



        output3_de = self.conv_trans_3(output2_de) #10x128x64x64
        #print(output3_de.size())

        output4_de = self.conv_trans_4(output3_de) #10x64x128x128
        #print(output4_de.size())

        output5_de = self.conv_trans_5(output4_de) #10x128x64x64
        #print(output5_de.size())

        return output5_de

class GeneratorWithSkipConnections(nn.Module):
    """ The GAN generator that uses a encoder decoder architecture, with skip connections
    https://stackoverflow.com/questions/51773208/pytorch-skip-connection-in-a-sequential-model
    https://stackoverflow.com/questions/55812474/implementing-u-net-with-skip-connection-in-pytorch
    """

    def __init__(self):
        super(GeneratorWithSkipConnections, self).__init__()

        self.createNetwork()

    def createNetwork(self):

        # Encoding layers
        self.conv_1 = nn.Sequential(
        nn.Conv2d(1, 64, 6, stride=2, padding=2, bias=False), # in channel, out channel, filter kernel size
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
        nn.Conv2d(256, 512, 2, stride=2, padding=0, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.2 )
        )

        self.conv_5 = nn.Sequential(
        nn.Conv2d(512, 512, 2, stride=2, padding=0, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.2 )
        )


        # Decoding layers

        #to concat t3 = torch.cat((t1, t2), 1)

        self.conv_trans_1 = nn.Sequential(
        nn.ConvTranspose2d(512, 512, 2, stride=2, padding=0, output_padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        )


        self.conv_trans_2 = nn.Sequential(
        nn.ConvTranspose2d(1024, 256, 2, stride=2, padding=0, output_padding=0, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        )

        self.conv_trans_3 = nn.Sequential(
        nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, output_padding=0, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        )

        self.conv_trans_4 = nn.Sequential(
        nn.ConvTranspose2d(256, 64, 6, stride=2, padding=2, output_padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )

        self.conv_trans_5 = nn.Sequential(
        nn.ConvTranspose2d(128, 3, 6, stride=2, padding=2, output_padding=0, bias=False),
        #nn.BatchNorm2d(3),
        nn.Tanh()
        )
        # Tanh at last layer. Source: https://github.com/soumith/ganhacks


    def forward(self, data):
        # print('data.size()')
        # print(data.size())


        output1 = self.conv_1(data) #10x64x128x128

        # print(output1.size())
        # print(type(output1))



        output2 = self.conv_2(output1) #10x128x64x64
        # print(output2.size())


        output3 = self.conv_3(output2) #10x256x32x32
        # print(output3.size())

        output4 = self.conv_4(output3) #10x512x16x16
        # print(output4.size())

        output5 = self.conv_5(output4) #10x512x8x8
        # print(output5.size())

        # Decoding
        # print("decoding")

        output1_de = self.conv_trans_1(output5) #10x512x16x16
        # print(output1_de.size())

        skip1_de = torch.cat((output4, output1_de), 1) #10x1024x16x16

        output2_de = self.conv_trans_2(skip1_de) #10x256x32x32
        # print(output2_de.size())

        skip2_de = torch.cat((output3, output2_de), 1) #10x512x16x16

        output3_de = self.conv_trans_3(skip2_de) #10x128x64x64
        # print(output3_de.size())

        skip3_de = torch.cat((output2, output3_de), 1) #10x256x16x16

        output4_de = self.conv_trans_4(skip3_de) #10x64x128x128
        # print(output4_de.size())

        skip4_de = torch.cat((output1, output4_de), 1) #10x128x16x16

        output5_de = self.conv_trans_5(skip4_de) #10x128x64x64
        # print(output5_de.size())

        return output5_de


if __name__ == "__main__":
    #gan = Generator()
    gan = GeneratorWithSkipConnections()
    #hl_graph = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
    #hl_graph = hl.build_graph(gan, torch.zeros([1, 1, 256, 256]))

    image_array = imread('TheSimpsonsS10E02TheWizardofEvergreenTerrace.mp40038.jpg')

    img_lab = color.rgb2lab(image_array)
    print(img_lab)
    gan(torch.tensor(  [np.transpose(img_lab[:, : ,[0]])]  ).float())

    # image_array = np.matrix.transpose(image_array)
    # image_array = torch.tensor([[image_array]]) # nu int 32
    # image_array = image_array.type('torch.FloatTensor')
    # #generated_im = gan.forward(image_array)
    # output = discriminator(image_array)


    # generated = generated_im.data.numpy()
    # generated = generated[0,:,:,:]
    # generated = np.round((generated + 1) * 255 / 2)
    # generated =generated.astype(int)

    # print(generated.transpose())
    # plt.show(plt.imshow( generated.transpose() ))
