from __future__ import print_function

from discriminator import Discriminator
from generator import GeneratorWithSkipConnections

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
    batch_size = 64


    #dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\alex_trainset_22apr\\trainset_gray\\temp"
    #dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\alex_trainset_22apr\\trainset_gray"

    #dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\Dataset\\trainset"
    dataroot_train = "/home/projektet/dataset/trainset/"
    #dataroot_test = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\Dataset\\testset"
    dataroot_test = "/home/projektet/dataset/testset/"
    #dataroot_val = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\Dataset\\validationset"
    dataroot_val = "/home/projektet/dataset/validationset/"

    #dataroot_train = "/home/projektet/dataset/trainset/"
    #dataroot_train = "/Users/Marcus/Downloads/kth_simps_gray_v1/trainset"
    #dataroot_train = "/home/jacob/Documents/DD2424 dataset/trainset/"

    trainset = SimpsonsDataset(datafolder = dataroot_train, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    validationset = SimpsonsDataset(datafolder = dataroot_val, transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=workers)
    dataloader_validation = torch.utils.data.DataLoader(trainset, batch_size=int(batch_size/4),shuffle=True, num_workers=workers)

    return dataloader, batch_size, dataloader_validation





def optimizers(generator, discriminator1, discriminator2, learningrate=2e-4, amsgrad=False, b=0.9, momentum=0.9):
    # https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
    # Use adam for generator and SGD for discriminator. source: https://github.com/soumith/ganhacks

    Discriminator1_optimizer = optim.SGD(
        discriminator1.parameters(), lr=learningrate, momentum=momentum)

    Discriminator2_optimizer = optim.SGD(
        discriminator2.parameters(), lr=learningrate, momentum=momentum)

    Generator_optimizer = optim.Adam(
        generator.parameters(), lr=learningrate, betas=(b, 0.999))

    return Discriminator1_optimizer, Discriminator2_optimizer , Generator_optimizer


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
    #generator.load_state_dict(torch.load("/home/projektet/network_v2/models/generator_model_3_19.pt"))

    # We select 2 different seeds for the discriminators
    torch.manual_seed(55)
    discriminator1 = Discriminator()
    discriminator1 = discriminator1.to(device)
    discriminator1.apply(weights_init)
    #discriminator.load_state_dict(torch.load("/home/projektet/network_v2/models/discriminator_model_3_19.pt"))

    torch.manual_seed(976654)
    discriminator2 = Discriminator()
    discriminator2 = discriminator2.to(device)
    discriminator2.apply(weights_init)


    dataloader, batch_size, dataloader_validation = loadData()

    true_im_label, false_im_label = get_labels()

    # Set the mode of the discriminator to training
    discriminator1 = discriminator1.train()
    discriminator2 = discriminator2.train()

    # Set the mode of the generator to training
    generator = generator.train()

    Discriminator_optimizer1, Discriminator_optimizer2 , Generator_optimizer = optimizers(generator, discriminator1, discriminator2)

    img_list = []

    D_losses = []

    G_losses = []
    G_losses_val = []

    D1_losses = []
    D2_losses = []

    D1_losses_val = []
    D2_losses_val = []
    iters = 0

    lam = 100

    for epoch in range(0, epochs):

        #for current_batch, b in enumerate(dataloader):

        for current_batch,((image, image_gray, img_name, img_name_gray), (image_val, image_gray_val, img_name_val, img_name_gray_val)) in enumerate(zip(dataloader, dataloader_validation)):

            # batch = b[0]
            # batch  = batch.to(device)
            # batch_gray = b[1]
            # batch_gray  = batch_gray.to(device)

            image = image.to(device)
            image_gray = image_gray.to(device)

            image_val = image_val.to(device)
            image_gray_val = image_gray_val.to(device)


            ### Update Discriminator1 ###
            #device = "cpu"
            ngpu = 1
            device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
            # Train with real colored data

            discriminator1.zero_grad()

            # forward pass
            output = discriminator1(image)

            # format labels into tensor vector
            true_im_label_soft = random.uniform(0.9, 1.0)
            labels_real = torch.full((batch_size,), true_im_label_soft, device=device)

            BCE_loss = loss_function(BCE = True)

            # The loss on the real batch data

            D1_loss_real = BCE_loss(output.squeeze(), labels_real)

            # Compute gradients for D via a backward pass
            D1_loss_real.backward()

            D1_x = output.mean().item()

            # Generate fake data - i.e. fake images by inputting black and white images


            batch_fake = generator(image_gray)

            # Train with the Discriminator with fake data

            false_im_label_soft = random.uniform(0.0, 0.1)

            labels_fake = torch.full((batch_size,), false_im_label_soft, device=device)


            # use detach since  ????
            output = discriminator1(batch_fake.detach())

            # Compute the loss

            D1_loss_fake = BCE_loss(output.squeeze(), labels_fake)

            D1_loss_fake.backward()

            D1_G_x1 = output.mean().item()

            D1_loss = D1_loss_fake + D1_loss_real

            # Walk a step - gardient descent
            Discriminator_optimizer1.step()
            ################################


            ### Update Discriminator2 ###

            # Train with real colored data

            discriminator2.zero_grad()

            # forward pass
            output = discriminator2(image)

            # format labels into tensor vector
            true_im_label_soft = random.uniform(0.9, 1.0)
            labels_real = torch.full((batch_size,), true_im_label_soft, device=device)

            BCE_loss = loss_function(BCE = True)

            # The loss on the real batch data

            D2_loss_real = BCE_loss(output.squeeze(), labels_real)

            # Compute gradients for D2 via a backward pass
            D2_loss_real.backward()

            D2_x = output.mean().item()


            # Generate fake data - i.e. fake images by inputting black and white images
            #batch_fake = generator(batch_gray)



            # Train with the Discriminator2 with fake data

            false_im_label_soft = random.uniform(0.0, 0.1)

            labels_fake = torch.full((batch_size,), false_im_label_soft, device=device)


            # use detach since  ????
            output = discriminator2(batch_fake.detach())

            # Compute the loss

            D2_loss_fake = BCE_loss(output.squeeze(), labels_fake)

            D2_loss_fake.backward()

            D2_G_x1 = output.mean().item()

            D2_loss = D2_loss_fake + D2_loss_real

            # Walk a step - gardient descent
            Discriminator_optimizer2.step()

            ###############################




            ### Update the generator ###
            # maximize log(D(G(z)))

            generator.zero_grad()

            # format labels into tensor vector

            labels_real = torch.full((batch_size,), true_im_label, device=device)

            # Now we need 2 outputs since we have 2 discriminators
            output1 = discriminator1(batch_fake)
            output2 = discriminator2(batch_fake)


            # The generators losses for the 2 discriminators
            G1_loss_bce = BCE_loss(output1.squeeze(), labels_real)
            G2_loss_bce = BCE_loss(output2.squeeze(), labels_real)


            L1 = nn.L1Loss()
            G_loss_L1 = L1(batch_fake.view(batch_fake.size(0),-1), image.view(image.size(0),-1))

            G1_loss = G1_loss_bce + lam * G_loss_L1
            G2_loss = G2_loss_bce + lam * G_loss_L1

            # We give equal weight to both discriminators
            G_weighted_loss = 0.5 * G1_loss + 0.5 * G2_loss

            G_weighted_loss.backward()


            D_G_1_x2 = output1.mean().item()
            D_G_2_x2 = output2.mean().item()

            Generator_optimizer.step()
            G_losses.append(G_weighted_loss.item())
            D1_losses.append(D1_loss.item())
            D2_losses.append(D2_loss.item())


            ############### VALIDATION ###################


            # ********************* Validation code ***********************

            # Set the mode of the discriminator and generator to training
            discriminator1 = discriminator1.eval()
            discriminator2 = discriminator2.eval()
            generator = generator.eval()

            # image_l_val = image_l_val.to(device)
            # image_ab_val = image_ab_val.to(device)
            # image_val = image_val.to(device)

            # G_losses_val = []
            # D_losses_val = []

            # Discriminator 1

            # forward pass
            output_val = discriminator1(image_val.float())

            # format labels into tensor vector
            true_im_label_soft = random.uniform(0.9, 1.0)
            labels_real_val = torch.full((int(batch_size/4),), true_im_label_soft, device=device)

            # The loss on the real batch data
            D1_loss_real_val = BCE_loss(output_val.squeeze(), labels_real_val)

            # Generate fake data - i.e. fake images by inputting black and white images
            batch_fake_val = generator(image_gray_val.float())

            # Train with the Discriminator with fake data

            false_im_label_soft = random.uniform(0.0, 0.1)

            labels_fake_val = torch.full((int(batch_size/4),), false_im_label_soft, device=device)

            # use detach since  ????
            output1_val = discriminator1(batch_fake_val.detach())

            # Compute the loss

            D1_loss_fake_val = BCE_loss(output1_val.squeeze(), labels_fake_val)

            D1_loss_val = D1_loss_fake_val + D1_loss_real_val



            # Discriminator 2

            # forward pass
            output_val = discriminator2(image_val.float())

            # format labels into tensor vector
            true_im_label_soft = random.uniform(0.9, 1.0)
            labels_real_val = torch.full((int(batch_size/4),), true_im_label_soft, device=device)

            # The loss on the real batch data
            D2_loss_real_val = BCE_loss(output_val.squeeze(), labels_real_val)

            # Generate fake data - i.e. fake images by inputting black and white images
            # Do not need to do this row twice
            #batch_fake_val = generator(image_gray_val.float())

            # Train with the Discriminator with fake data

            false_im_label_soft = random.uniform(0.0, 0.1)

            labels_fake_val = torch.full((int(batch_size/4),), false_im_label_soft, device=device)

            # use detach since  ????
            output2_val = discriminator2(batch_fake_val.detach())

            # Compute the loss

            D2_loss_fake_val = BCE_loss(output2_val.squeeze(), labels_fake_val)

            D2_loss_val = D2_loss_fake_val + D2_loss_real_val





            # Generator loss

            labels_real_val = torch.full((int(batch_size/4),), true_im_label, device=device)

            output1_val = discriminator1(batch_fake_val)

            # The generators loss
            G1_loss_bce_val = BCE_loss(output1_val.squeeze(), labels_real_val)
            L1_val = nn.L1Loss()
            G_loss_L1_val = L1(batch_fake_val.view(batch_fake_val.size(0),-1), image_val.view(image_val.size(0),-1).float())

            G1_loss_val = G1_loss_bce_val + lam * G_loss_L1_val



            output2_val = discriminator2(batch_fake_val)

            # The generators loss
            G2_loss_bce_val = BCE_loss(output2_val.squeeze(), labels_real_val)
            L1_val = nn.L1Loss()
            G2_loss_L1_val = L1(batch_fake_val.view(batch_fake_val.size(0),-1), image_val.view(image_val.size(0),-1).float())

            G2_loss_val = G2_loss_bce_val + lam * G_loss_L1_val


            G_loss_val = 0.5 * G1_loss_val + 0.5 * G2_loss_val




            G_losses_val.append(G_loss_val.item())
            D1_losses_val.append(D2_loss_val.item())
            D2_losses_val.append(D1_loss_val.item())

            ##############################################


            if current_batch % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D1: %.4f\tLoss_D2: %.4f\tLoss_G: %.4f'% (epoch, epochs, current_batch, len(dataloader), D1_loss.item(), D2_loss.item(), G_weighted_loss.item()))
                plot_losses(G_losses, D1_losses, D2_losses, G_losses_val, D1_losses_val, D2_losses_val, epoch, current_batch)
                # Save Losses for plotting later
                #G_losses.append(G_loss.item())
                #D_losses.append(D_loss.item())


                iters += 1

            if current_batch == 3:
            #if criteria_validate_generator(dataloader, iters, epoch, epochs, current_batch):

                wrapped_bw_im = image_gray_val[0].unsqueeze(0)
                save_image(image_gray_val[0], epoch, current_batch, device, False)
                save_image(image_val[0], epoch, current_batch, device, True)
                wrapped_bw_im = wrapped_bw_im.to(device)
                img = validate_generator(epoch, current_batch, wrapped_bw_im, generator)
                img_list.append(img)


            if current_batch == 3:
                file_name_generator = "generator_model"

                file_name_discriminator1 = "discriminator1_model"
                file_name_discriminator2 = "discriminator2_model"

                file_name_discriminator1_optimizer = "Discriminator1_optimizer"
                file_name_discriminator2_optimizer = "Discriminator2_optimizer"
                file_name_generator_optimizer = "Generator_optimizer"

                # /home/projektet/
                torch.save(discriminator1.state_dict(),  "/home/projektet/network_v7/models/"+file_name_discriminator1 + "_"  +str(current_batch) + "_" + str(epoch) +  ".pt")
                torch.save(Discriminator_optimizer1, "/home/projektet/network_v7/models/"+ file_name_discriminator1_optimizer + "_"  +str(current_batch) + "_" + str(epoch) +  ".pt")

                torch.save(discriminator2.state_dict(), "/home/projektet/network_v7/models/"+ file_name_discriminator2+"_"  +str(current_batch) + "_" + str(epoch) +  ".pt")
                torch.save(Discriminator_optimizer2, "/home/projektet/network_v7/models/"+ file_name_discriminator2_optimizer + "_"  +str(current_batch) + "_" + str(epoch) +  ".pt")

                torch.save(generator.state_dict(), "/home/projektet/network_v7/models/" + file_name_generator +"_" + str(current_batch) + "_" + str(epoch) +  ".pt")
                torch.save(Generator_optimizer, "/home/projektet/network_v7/models/" + file_name_generator_optimizer + "_"  +str(current_batch) + "_" + str(epoch) +  ".pt")

            # Set the mode of the discriminator to training
            discriminator1 = discriminator1.train()
            discriminator2 = discriminator2.train()

            # Set the mode of the generator to training
            generator = generator.train()


    return 0


def validate_generator(epoch, current_batch, bw_im, generator, padding_sz=2,  norm=True):
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
    plt.axis("off")
    plt.savefig("/home/projektet/network_v7/result_pics/" + str(epoch) + "_" + str(current_batch) + "_Color_generated.png")
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

def criteria_validate_generator(dataloader, iters, epoch, epochs, current_batch):
    return  (iters % 10 == 0) or ((epoch == epochs - 1) and (current_batch == len(dataloader) - 1))


def save_image(img, epoch, current_batch, device, color):
    """ Saves a black and white image
    """

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(img.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    if(color):
        plt.savefig("/home/projektet/network_v7/result_pics/" +str(epoch) + "_" + str(current_batch) + "_color.png")
    else:
        plt.savefig("/home/projektet/network_v7/result_pics/" +str(epoch) + "_" + str(current_batch) + "_BW.png")
    #plt.clf()
    plt.close()

    #plt.show()


def plot_losses(G_losses, D1_losses, D2_losses, G_losses_val, D1_losses_val, D2_losses_val, epoch, current_batch):

    """ creates two plots. One for the Generator loss and one for the Discriminator loss and saves these figures
    """
    D1_loss_fig = plt.figure('D1_cost' + str(epoch) + '_' + str(current_batch))
    plt.plot(D1_losses, color='b', linewidth=1.5, label='D1 cost')  # axis=0
    plt.plot(D1_losses_val, color='purple', linewidth=1.5, label='D1 loss validation')  # axis=0
    plt.legend(loc='upper left')
    D1_loss_fig.savefig('plots/D1_loss' + str(epoch) + '_' + str(current_batch) +'.png', dpi=D1_loss_fig.dpi)
    #plt.clf()
    plt.close(D1_loss_fig)

    D2_loss_fig = plt.figure('plots/D2_loss' + str(epoch) + '_' + str(current_batch))
    plt.plot(D2_losses, color='b', linewidth=1.5, label='D2_cost')  # axis=0
    plt.plot(D2_losses_val, color='purple', linewidth=1.5, label='D2 loss validation')  # axis=0
    plt.legend(loc='upper left')
    D2_loss_fig.savefig('plots/D2_loss' + str(epoch) + '_' + str(current_batch) +'.png', dpi=D2_loss_fig.dpi)
    #plt.clf()
    plt.close(D2_loss_fig)




    G_loss_fig = plt.figure('G_cost' + str(epoch) + '_' + str(current_batch))
    plt.plot(G_losses, color='b', linewidth=1.5, label='G cost')  # axis=0
    plt.plot(G_losses_val, color='purple', linewidth=1.5, label='G loss validation')  # axis=0
    plt.legend(loc='upper left')
    G_loss_fig.savefig('plots/G_loss' + str(epoch) + '_' + str(current_batch) +'.png', dpi=G_loss_fig.dpi)
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
