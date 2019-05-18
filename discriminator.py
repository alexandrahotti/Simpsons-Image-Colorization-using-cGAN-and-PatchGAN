import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """ The GAN generator that uses a encoder decoder architecture
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # Encoding layers
        self.conv_1 = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False), 
        nn.BatchNorm2d( 64 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_2 = nn.Sequential(
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d( 128 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_3 = nn.Sequential(
        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d( 256 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_4 = nn.Sequential(
        nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_5 = nn.Sequential(
        nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_6 = nn.Sequential(
        nn.Conv2d(512, 512, 5, stride=1, padding=0, bias=False),
        nn.BatchNorm2d( 512 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_7 = nn.Sequential(
        nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d( 1 ),
        nn.LeakyReLU( 0.1 )
        )

        self.conv_8 = nn.Sequential(
        nn.Conv2d(1, 1, 4, stride=1, padding=0, bias=False),
        nn.Sigmoid()
        )

    def forward(self, RGB_image):

        h1 = self.conv_1(RGB_image)

        h2 = self.conv_2(h1)

        h3 = self.conv_3(h2)

        h4 = self.conv_4(h3)

        h5 = self.conv_5(h4)

        h6 = self.conv_6(h5)

        h7 = self.conv_7(h6)

        h8 = self.conv_8(h7)


        return h8
