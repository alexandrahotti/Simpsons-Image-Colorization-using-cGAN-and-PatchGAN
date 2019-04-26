from __future__ import print_function

from discriminator_v5 import Discriminator
from generator_v5 import GeneratorWithSkipConnections

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
#from torchsample.transforms import RangeNormalize

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
    def __init__(self, datafolder, transform = None, rgb = True ):
        self.datafolder = datafolder
        all_files_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(datafolder)) for f in fn]
        #self.image_files_list = [s for s in os.listdir(datafolder) if
        #                         '_gray.jpg' not in s and os.path.isfile(os.path.join(datafolder, s))]
        self.image_files_list = [s for s in all_files_list if
                                 '_gray.jpg' not in s and '.jpg' in s and os.path.isfile(os.path.join(datafolder, s))]

        # Same for the labels files
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.datafolder,
                                self.image_files_list[idx])
        img_name_gray = img_name[0:-4] + '_gray.jpg'
        image = io.imread(img_name)

        if self.rgb:
            image = self.transform(image)
            img_name_gray = img_name[0:-4] + '_gray.jpg'
            image_gray = io.imread(img_name_gray)
            image_gray = self.transform(image_gray)
            return image, image_gray, img_name, img_name_gray

        else:
            #lab
            image_lab = color.rgb2lab(np.array(image)) # np array
            image_l = image_lab[:,:,[0]]
            image_ab = image_lab[:,:,[1,2]]

            image = self.transform(image)
            image_l = self.transform(image_l)
            image_ab = self.transform(image_ab)

            return image, image_l, image_ab, img_name, img_name_gray

def loadData():
    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 100


    #dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\alex_trainset_22apr\\trainset_gray\\temp"
    #dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\alex_trainset_22apr\\trainset_gray"
    #dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\Dataset\\trainset"
    #dataroot_train = "/home/projektet/dataset/trainset/"
    dataroot_train = "/Users/Marcus/Downloads/kth_simps_gray_v1/trainset"
    #dataroot_train = "/home/jacob/Documents/DD2424 dataset/trainset/"

    trainset = SimpsonsDataset(datafolder = dataroot_train, transform=transforms.Compose([
        transforms.ToTensor()
    ]), rgb = False))

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=workers)

    return dataloader, batch_size



def optimizers(generator, discriminator, learningrate=2e-4, amsgrad=False, b=0.9, momentum=0.9):
    # https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
    # Use adam for generator and SGD for discriminator. source: https://github.com/soumith/ganhacks

    Discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=learningrate, betas=(b, 0.5))
    Generator_optimizer = optim.Adam(
        generator.parameters(), lr=learningrate, betas=(b, 0.5))

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
    epochs = 20
    ngpu = 1
    #device = "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    generator = GeneratorWithSkipConnections()  # ngpu) #.to(device) add later to make meory efficient
    generator = generator.to(device)
    generator.apply(weights_init)
    #generator.load_state_dict(torch.load("/home/projektet/dataset/models/"))

    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    discriminator.apply(weights_init)
    #discriminator.load_state_dict(torch.load("/home/projektet/dataset/models/"))

    dataloader, batch_size = loadData()

    true_im_label, false_im_label = get_labels()

    # Set the mode of the discriminator to training
    discriminator = discriminator.train()

    # Set the mode of the generator to training
    generator = generator.train()

    Discriminator_optimizer, Generator_optimizer = optimizers(generator, discriminator)

    img_list = []
    reference_list = []
    G_losses = []
    D_losses = []
    iters = 0

    lam = 100

    for epoch in range(0, epochs):

        for current_batch, b in enumerate(dataloader):
            #discriminator, generator, Discriminator_optimizer, Generator_optimizer, iters, G_losses, D_losses = train_GAN(discriminator, generator, Discriminator_optimizer, Generator_optimizer , batch[0], batch[1], true_im_label, false_im_label,
            #i, epoch, iters, epochs, dataloader, G_losses, D_losses, batch_size)
#def train_GAN(discriminator,generator,Discriminator_optimizer, Generator_optimizer , batch, batch_gray, true_im_label, false_im_label,
#current_batch, epoch, iters, epochs, dataloader, G_losses, D_losses, batch_size):


            batch = b[0]
            batch  = batch.to(device)
            batch_gray = b[1]
            batch_gray  = batch_gray.to(device)
            grey_file = b[4]
            color_file = b[3]


            ### update the discriminator ###
            #device = "cpu"
            ngpu = 1
            device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
            # Train with real colored data

            discriminator.zero_grad()

            # forward pass
            output = discriminator(batch)

            # format labels into tensor vector
            true_im_label_soft = random.uniform(0.9, 1.0)
            labels_real = torch.full((batch_size,), true_im_label_soft, device=device)

            BCE_loss = loss_function(BCE = True)

            # The loss on the real batch data

            D_loss_real = BCE_loss(output.squeeze(), labels_real)

            # Compute gradients for D via a backward pass
            D_loss_real.backward()

            D_x = output.mean().item()

            # Generate fake data - i.e. fake images by inputting black and white images


            batch_fake = generator(batch_gray)

            # Train with the Discriminator with fake data

            false_im_label_soft = random.uniform(0.0, 0.1)

            labels_fake = torch.full((batch_size,), false_im_label_soft, device=device)


            # use detach since  ????
            output = discriminator(batch_fake.detach())

            # Compute the loss

            D_loss_fake = BCE_loss(output.squeeze(), labels_fake)

            D_loss_fake.backward()

            D_G_x1 = output.mean().item()

            D_loss = D_loss_fake + D_loss_real

            # Walk a step - gardient descent
            Discriminator_optimizer.step()

            ### Update the generator ###
            # maximize log(D(G(z)))

            generator.zero_grad()

            # format labels into tensor vector

            labels_real = torch.full((batch_size,), true_im_label, device=device)


            output = discriminator(batch_fake)

            # The generators loss
            G_loss_bce = BCE_loss(output.squeeze(), labels_real)
            L1 = nn.L1Loss()
            G_loss_L1 = L1(batch_fake.view(batch_fake.size(0),-1), batch.view(batch.size(0),-1))

            G_loss = G_loss_bce + lam * G_loss_L1
            G_loss.backward()

            D_G_x2 = output.mean().item()

            Generator_optimizer.step()
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())

            if current_batch % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, epochs, current_batch, len(dataloader), D_loss.item(), G_loss.item(), D_x, D_G_x1, D_G_x2))
                plot_losses(G_losses, D_losses, epoch, current_batch)
                # Save Losses for plotting later
                #G_losses.append(G_loss.item())
                #D_losses.append(D_loss.item())


                iters += 1

            if current_batch == 50:
            #if criteria_validate_generator(dataloader, iters, epoch, epochs, current_batch):

                wrapped_bw_im = batch_gray[50].unsqueeze(0)
                #print("Filename of colorifle: " + color_file[-1])
                #input()
                save_image(batch_gray[50], epoch, current_batch, device, False)
                save_image(batch[50], epoch, current_batch, device, True)
                wrapped_bw_im = wrapped_bw_im.to(device)

                file_name_reference = "/Users/Marcus/Downloads/kth_simps_gray_v1/testset/TheSimpsonsS10E16MakeRoomforLisa.mp40002_gray.jpg"
                reference_bw = io.imread(file_name_reference)
                reference_bw = transforms.ToTensor()(reference_bw)
                reference_bw = reference_bw.unsqueeze(0)
                reference_bw =reference_bw.to(device)

                img = validate_generator(epoch, current_batch, wrapped_bw_im, generator)
                reference_img = validate_generator(epoch, current_batch, reference_bw, generator, True)

                img_list.append(img)
                reference_list.append(reference_img)


            if (current_batch == 50) and (device != "cpu"):
                file_name_generator = "generator_model"
                file_name_discriminator = "discriminator_model"

                torch.save(discriminator.state_dict(), "/home/projektet/network_v5/models/discriminator_model" + "_"  +str(current_batch) + "_" + str(epoch) +  ".pt")
                # #torch.save(discriminator_optimizer, model_path)
                torch.save(generator.state_dict(), "/home/projektet/network_v5/models/generator_model" + "_" + str(current_batch) + "_" + str(epoch) +  ".pt")
                # #torch.save(generator_optimizer, model_path)


    return 0


def validate_generator(epoch, current_batch, bw_im, generator, padding_sz=2,  norm=True, reference = False):
    with torch.no_grad():
        fake_im = generator(bw_im).detach().cpu()

        # generated = fake_im.data.numpy()
        # generated = generated[0, :, :, :]
        # generated = np.round((generated + 1) * 255 / 2)
        # generated = generated.astype(int)
        #
        # # print(generated.transpose())
        # #plt.show(plt.imshow( generated.transpose() ))
        #
        # plt.imshow( generated.transpose() )
        # filename = str(it) + '.png'
        # plt.savefig(filename)



    img = vutils.make_grid(fake_im, padding = padding_sz, normalize = norm )
    plt.imshow(np.transpose(img,(1,2,0)), animated=True)

    if reference:
        plt.savefig("/home/projektet/network_v5/resultPics/" + str(epoch) + "_" + str(current_batch) + "_color_generated_reference.png")
    else:
        plt.savefig("/home/projektet/network_v5/resultPics/" + str(epoch) + "_" + str(current_batch) + "_color_generated.png")

    #plt.clf()
    plt.close()


    return img


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def criteria_validate_generator(dataloader, iters, epoch, epochs, current_batch, ):
    return  (iters % 10 == 0) or ((epoch == epochs - 1) and (current_batch == len(dataloader) - 1))


def save_image(img, epoch, current_batch, device, color):
    """ Saves a black and white image
    """

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(img.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    if(color):
        plt.savefig("/home/projektet/network_v5/resultPics/" + str(epoch) + "_" + str(current_batch) + "_color.png")
    else:
        plt.savefig("/home/projektet/network_v5/resultPics/" + str(epoch) + "_" + str(current_batch) + "_BW.png")
    #plt.clf()
    plt.close()

    #plt.show()


def plot_losses(G_losses, D_losses, epoch, current_batch):
    """ creates two plots. One for the Generator loss and one for the Discriminator loss and saves these figures
    """
    D_loss_fig = plt.figure('D_loss' + str(epoch) + '_' + str(current_batch))
    plt.plot(D_losses, color='b', linewidth=1.5, label='D_loss')  # axis=0
    plt.legend(loc='upper left')
    D_loss_fig.savefig("/home/projektet/network_v5/graphPics/" + 'D_loss' + str(epoch) + '_' + str(current_batch) +'.png', dpi=D_loss_fig.dpi)
    #plt.clf()
    plt.close(D_loss_fig)

    G_loss_fig = plt.figure('G_loss' + str(epoch) + '_' + str(current_batch))
    plt.plot(G_losses, color='b', linewidth=1.5, label='G_loss')  # axis=0
    plt.legend(loc='upper left')
    G_loss_fig.savefig("/home/projektet/network_v5/graphPics/" + 'G_loss' + str(epoch) + '_' + str(current_batch) +'.png', dpi=G_loss_fig.dpi)
    #plt.clf()
    plt.close(G_loss_fig)


def tensor_format_labels(b_size, label_vec, device):

    label = torch.full((b_size,8,8), label_vec, device=device)


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
