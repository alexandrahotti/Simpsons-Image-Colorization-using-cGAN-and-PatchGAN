from __future__ import print_function

from discriminator import Discriminator
from generator import Generator

import torch
import torch.nn as nn
#import cv2

# from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
from torch.autograd import Variable
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as torch_data
import torchvision.datasets as dset


from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler, SubsetRandomSampler
from torchvision import transforms, utils
import torchvision.utils as vutils

from skimage import io, transform

import random
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

seed = 123
random.seed(seed)
torch.manual_seed(seed)


from torch.utils.data import Dataset

class SimpsonsDataset(Dataset):
    def __init__(self, datafolder, transform = None):
        self.datafolder = datafolder
        all_files_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(datafolder)) for f in fn]
        #self.image_files_list = [s for s in os.listdir(datafolder) if
        #                         '_gray.jpg' not in s and os.path.isfile(os.path.join(datafolder, s))]
        self.image_files_list = [s for s in all_files_list if
                                 '_gray.jpg' not in s and '.jpg' in s and os.path.isfile(os.path.join(datafolder, s))]

        # Same for the labels files
        self.transform = transform

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.datafolder,
                                self.image_files_list[idx])
        img_name_gray = img_name[0:-4] + '_gray.jpg'
        image = io.imread(img_name)
        image = self.transform(image)
        image_gray = io.imread(img_name_gray)
        image_gray = self.transform(image_gray)
        return image, image_gray




def loadData():
    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 10


    dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\alex_trainset_22apr\\trainset_gray"

    trainset = SimpsonsDataset(datafolder = dataroot_train, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=workers)

    return dataloader



def optimizers(generator, discriminator, learningrate=1e-4, amsgrad=False, b=0.9, momentum=0.9):
    # https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
    # use adam for generator and SGD for discriminator. source: https://github.com/soumith/ganhacks

    Discriminator_optimizer = optim.SGD(
        discriminator.parameters(), lr=learningrate, momentum=momentum)
    Generator_optimizer = optim.Adam(
        generator.parameters(), lr=learningrate, betas=(b, 0.999))

    return Discriminator_optimizer, Generator_optimizer


def loss_function(BCE):

    if BCE:
        return nn.BCELoss()

    return False


def get_labels():

    true_im = 1
    false_im = 0

    return true_im, false_im





def GAN_training():
    epochs = 10
    ngpu = 1
    device = "cpu"
    generator = Generator()  # ngpu) #.to(device) add later to make meory efficient

    discriminator = Discriminator()

    dataloader = loadData()


    true_im_label, false_im_label = get_labels()

    # Set the mode of the discriminator to training
    discriminator = discriminator.train()

    # Set the mode of the generator to training
    generator = generator.train()

    Discriminator_optimizer, Generator_optimizer = optimizers(generator, discriminator)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(1, epochs):

        for i, batch in enumerate(dataloader):
            # image_size = 256
            #
            # # Number of channels in the training images. For color images this is 3
            # nc = 3
            #
            # device = "cpu"
            # plt.subplot(2, 1, 1)
            # #plt.figure(figsize=(8,8))
            # plt.axis("off")
            # plt.title("Training Images")
            # plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            # #plt.show()
            # plt.subplot(2, 1, 2)
            # #plt.figure(figsize=(8,8))
            # plt.axis("off")
            # plt.title("Training Images")
            # plt.imshow(np.transpose(vutils.make_grid(batch_gray[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            # plt.show()
            #
            # input()

            iters, G_losses, D_losses = train_GAN(discriminator,generator,Discriminator_optimizer, Generator_optimizer , batch[0], batch[1], true_im_label, false_im_label, i, epoch, iters)

            if criteria_validate_generator(iters, epoch, epochs, i):
                img = validate_generator(bw_im, generator)
                img_list.append(img)


    return 0


def validate_generator(bw_im, generator, padding_sz=2,  norm=True):
    with torch.no_grad():
        fake_im = generator(bw_im).detach().cpu()
    img = vutils.make_grid(fake, padding = padding_sz, normalize = norm )

    return img


def criteria_validate_generator(iters, epoch, epochs, current_batch):
    return (iters % 500 == 0) or ((epoch == epochs - 1) and (current_batch == len(dataloader) - 1))


def train_GAN(discriminator,generator,Discriminator_optimizer, Generator_optimizer , batch, batch_gray, true_im_label, false_im_label, current_batch, epoch, iters):
    ### update the discriminator ###
    device = "cpu"
    # Train with real colored data

    discriminator.zero_grad()

    # forward pass
    output = discriminator(batch)

    # format labels into tensor vector
    labels_real = tensor_format_labels(batch, device, true_im_label) #tensor_format_labels(batch, true_im_label)

    BCE_loss = loss_function(BCE = True)

    # The loss on the real batch data

    # print(reshape_to_vector(output))
    # input()
    # print(labels_real)
    # input()
    D_loss_real = BCE_loss(reshape_to_vector(output), labels_real)

    # Compute gradients for D via a backward pass
    D_loss_real.backward()

    D_x = output.mean().item()

    # Generate fake data - i.e. fake images by inputting black and white images

    # batch_gray = reformat_no_channels(batch_gray)
    # print(batch_gray)
    # input()
    print(batch_gray.shape)
    print(batch.shape)
    input()
    batch_fake = generator(batch_gray)

    # Train with the Discriminator with fake data

    labels_fake = tensor_format_labels(batch_fake, false_im_label)

    # use detach since  ????
    output = discriminator(batch_fake.detach())

    # Compute the loss
    D_loss_fake = BCE_loss(reshape_to_vector(output), labels_fake)

    D_loss_fake.backward()

    D_G_x1 = output.mean().item()

    D_loss = D_loss_fake + D_loss_real

    # Walk a step - gardient descent
    Discriminator_optimizer.step()

    ### Update the generator ###
    # maximize log(D(G(z)))

    generator.zero_grad()

    # format labels into tensor vector
    labels_real = tensor_format_labels(batch, true_im_label)



    output = discriminator(batch_fake)

    # The generators loss
    G_loss = BCE_loss(reshape_to_vector(output), labels_real)

    G_loss.backward()

    D_G_x2 = output.mean().item()

    Generator_optimizer.step()

    if current_batch % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, num_epochs, i, len(dataloader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())


        iters += 1

    return iters, G_losses, D_losses

def reformat_no_channels(batch):
    reformatted_tensors = []
    it = 0;
    for b in batch:
        #print(batch.data[it])
        print(batch.data[it][1])
        input()
        print(print(batch.data[it][0]))
        input()
        print(print(batch.data[it][2]))
        input()

        print(batch.data[it][1].shape)
        input('klart')
        #print(batch.data[it][0].shape)
        reformatted_tensors.append(torch.tensor(batch.data[it][0]))
        # print(reformatted_tensors)
        #input()
        it+=1

    tensors = torch.stack(reformatted_tensors)
    print(batch)
    input()
    return batch

def tensor_format_labels(batch, device, label_vec):

    cpu = batch[0].to(device)
    b_size = cpu.size(0)
    label = torch.full((10,8,8), label_vec, device=device)

    input()
    return label


def reshape_to_vector(output):
    return torch.squeeze(output) #output.view(-1)


if __name__ == "__main__":
    GAN_training()
    # gan = Generator()
    #
    # image_array = imread('test_im.jpg')
    # image_array = np.matrix.transpose(image_array)
    # image_array = torch.tensor([[image_array]])  # nu int 32
    # image_array = image_array.type('torch.FloatTensor')
    # generated_im = gan.forward_pass(image_array)
    # generated = generated_im.data.numpy()
    # generated = generated[0, :, :, :]
    # generated = np.round((generated + 1) * 255 / 2)
    # generated = generated.astype(int)
    #
    # # print(generated.transpose())
    # plt.show(plt.imshow( generated.transpose() ))
