import torch
#import torch.nn.functional as F
import torch.nn as nn

class Discriminator(nn.Module):
    """ The GAN generator that uses a encoder decoder architecture
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # Encoding layers
        self.conv_1 = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False), # in channel, out channel, filter kernel size
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

        self.conv_6 = nn.Sequential(
        nn.Conv2d(512, 512, 5, stride=1, padding=0, bias=False),
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_7 = nn.Sequential(
        nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False), # in channel, out channel, filter kernel size
        nn.BatchNorm2d( 1 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_8 = nn.Sequential(
        nn.Conv2d(1, 1, 4, stride=1, padding=0, bias=False), # in channel, out channel, filter kernel size
        nn.Sigmoid()
        )

    def forward(self, data):
        #
        print('data.size()')
        print(data.size())
        output1 = self.conv_1(data)
        print(output1.size())

        output2 = self.conv_2(output1)
        print(output2.size())

        output3 = self.conv_3(output2)
        print(output3.size())

        output4 = self.conv_4(output3)
        print(output4.size())


        output5 = self.conv_5(output4)
        print(output5.size())

        output6 = self.conv_6(output5)
        print(output6.size())

        output7 = self.conv_7(output6)
        print(output7.size())

        output8 = self.conv_8(output7)
        print(output8.size())

        # Decoding

        # print(output5.size())

        return output8
