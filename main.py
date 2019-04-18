import Discriminator from 'discriminator.py'
import Generator from 'generator.py'

import torch
import torch.nn as nn


import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

def GAN_training():
    epochs = 10
    generator = Generator()
    discriminator = Discriminator()


    return

def loadData ():




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
