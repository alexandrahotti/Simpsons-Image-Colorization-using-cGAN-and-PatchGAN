from __future__ import print_function

from discriminator import Discriminator
from generator import * #GeneratorWithSkipConnections

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

from PIL import Image
from skimage import color

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

seed = 1234
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
        image_rgb = image

        if self.rgb:
            image_norm = self.transform(image)
            img_name_gray = img_name[0:-4] + '_gray.jpg'
            image_gray = io.imread(img_name_gray)
            image_gray_un_norm = image_gray
            image_gray = self.transform(image_gray)
            return image_norm, image_gray, image_gray_un_norm, image_rgb, img_name, img_name_gray

        else:
            #lab
            image_lab = color.rgb2lab(np.array(image)) # np array
            image_l = image_lab[:,:,[0]]
            #image_ab = image_lab[:,:,[1,2]]
            image_tanh = (image - 127.5) / 127.5

            image_tanh = self.transform(image_tanh)
            image = self.transform(image)
            image_l = self.transform(image_l)
            #image_ab = self.transform(image_ab)
            image_lab = self.transform(image_lab)


            return image, image_l, image_lab, img_name, img_name_gray, image_tanh

def loadData():
    # Number of workers for dataloader
    workers = 3

    # Batch size during training
    batch_size = 56


    #dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\alex_trainset_22apr\\trainset_gray\\temp"
    #dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\alex_trainset_22apr\\trainset_gray"

    # dataroot_train = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\Dataset\\trainset"
    # dataroot_test = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\Dataset\\testset"
    # dataroot_val = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\Dataset\\validationset"


    dataroot_train = "/home/projektet/dataset/trainset/"
    dataroot_test = "/home/projektet/dataset/testset/"
    dataroot_val = "/home/projektet/dataset/validationset/"

    #
    # dataroot_train = "/home/jacob/Documents/DD2424 dataset/trainset/"
    # dataroot_test = "/home/jacob/Documents/DD2424 dataset/testset/"
    # dataroot_val = "/home/jacob/Documents/DD2424 dataset/validationset/"

    trainset = SimpsonsDataset(datafolder = dataroot_train, transform=transforms.Compose([
        transforms.ToTensor()
    ]), rgb = True)
    validationset = SimpsonsDataset(datafolder = dataroot_val, transform=transforms.Compose([
            transforms.ToTensor()
        ]), rgb = True)

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=workers)
    dataloader_validation = torch.utils.data.DataLoader(trainset, batch_size=int(batch_size/4),shuffle=True, num_workers=workers)

    return dataloader, batch_size, dataloader_validation



def convert_rgb_to_lab(img_rgb):

    image_lab = color.rgb2lab(np.array(img_rgb)) # np array
    image_l = image_lab[:,:,[0]]
    image_l = transforms.ToTensor()(image_l)

    return image_l

def optimizers(generator, discriminator, learningrate=2e-4, amsgrad=False, b=0.9, momentum=0.9):
    # https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
    # Use adam for generator and SGD for discriminator. source: https://github.com/soumith/ganhacks

    Discriminator_optimizer = optim.SGD(
        discriminator.parameters(), lr=learningrate, momentum=momentum)
    Generator_optimizer = optim.Adam(
        generator.parameters(), lr=learningrate, betas=(b, 0.999))

    return Discriminator_optimizer, Generator_optimizer


#
# def optimizers(generator, discriminator1, learningrate=2e-4, amsgrad=False, b=0.9, momentum=0.9):
#     # https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
#     # Use adam for generator and SGD for discriminator. source: https://github.com/soumith/ganhacks
#
#
#     Discriminator1_optimizer = optim.Adam(
#         discriminator1.parameters(), lr=0.001, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#
#
#     Generator_optimizer = optim.Adam(
#         generator.parameters(), lr=0.001, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    return Discriminator1_optimizer , Generator_optimizer



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
    start_epoch = 0
    ngpu = 1
    #device = "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    network_v = "network_v1_fixed"


    #jacobs path
    #path = "/home/jacob/Documents/DD2424 projekt/network_v2_fixed/"

    #alex
    # path = "C:\\Users\\Alexa\\Desktop\\KTH\\årskurs_4\\DeepLearning\\Assignments\\github\\Deep-Learning-in-Data-Science\\Project\\results\\network_v2\\buggsremoved_v2"


    path = "/home/projektet/" + network_v +"/"

    generator = Generator()  # ngpu) #.to(device) add later to make meory efficient
    generator = generator.to(device)
    #generator.apply(weights_init)
    #generator.load_state_dict(torch.load("/home/projektet/network_v2/models/generator_model_3_7.pt"))

    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    #discriminator.apply(weights_init)
    #discriminator.load_state_dict(torch.load("/home/projektet/network_v2_final/models/discriminator_model_400_29.pt"))''
    #generator.load_state_dict(torch.load("/home/projektet/network_v2_final/models/generator_model_400_29.pt"))

    dataloader, batch_size, dataloader_validation = loadData()

    true_im_label, false_im_label = get_labels()

    # Set the mode of the discriminator to training
    discriminator = discriminator.train()

    # Set the mode of the generator to training
    generator = generator.train()

    Discriminator_optimizer, Generator_optimizer = optimizers(generator, discriminator)

    #Discriminator_optimizer = torch.load("/home/projektet/network_v9/models/Discriminator_optimizer_400_29.pt")
    #Generator_optimizer = torch.load("/home/projektet/network_v9/models/Generator_optimizer_400_29.pt")

    #Discriminator_optimizer_scheduler = StepLR(Discriminator_optimizer, step_size=5, gamma=0.5, last_epoch = start_epoch-1)

    #Discriminator_optimizer_scheduler = StepLR(Discriminator_optimizer, step_size=5, gamma=0.5)


    reference_list = []

    D_losses = []
    D_losses_val = []

    G_losses = []
    G_losses_val = []
    iters = 0

    lam = 100/255


    for epoch in range(start_epoch, epochs):


        #Discriminator_optimizer_scheduler.step()
        #print('Epoch:', epoch,'LR:', Discriminator_optimizer_scheduler.get_lr())

        for current_batch,((image_norm, image_gray, image_gray_un_norm, image_rgb, img_name, img_name_gray), (image_norm_val, image_gray_val,image_gray_un_norm_val, image_rgb_val , img_name_val, img_name_gray_val)) in enumerate(zip(dataloader, dataloader_validation)):
# image_norm, image_gray, image_gray_un_norm, image_rgb, img_name, img_name_gray

            image_norm = image_norm.to(device)
            image_gray = image_gray.to(device)
            image_rgb  = image_rgb.to(device)



            ### update the discriminator ###
            #device = "cpu"


            # Train with real colored data

            discriminator.zero_grad()

            # forward pass

            output = discriminator(image_norm.float())

            # Using soft labels for the discriminator on real data during training
            labels_real = torch.tensor([random.uniform(0.8, 1.0) for _ in range(batch_size)]).to(device)


            BCE_loss = loss_function(BCE = True)

            # The loss on the real batch data

            D_loss_real = BCE_loss(output.squeeze(), labels_real)

            # Compute gradients for D via a backward pass
            D_loss_real.backward()

            D_x = output.mean().item()



            # Generate fake data - i.e. fake images by inputting black and white images
            batch_fake_rgb = generator(image_gray.float())



            # Train with the Discriminator with fake data

            #false_im_label_soft = random.uniform(0.0, 0.1)

            labels_fake = torch.full((batch_size,), false_im_label, device=device)

            #normalized_batch_fake_rgb = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
            normalized_batch_fake_rgb = (((batch_fake_rgb + 1 ) * (1)) / (2))

            output = discriminator(normalized_batch_fake_rgb.detach())

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



            output = discriminator(normalized_batch_fake_rgb)



            # The generators loss
            G_loss_bce = BCE_loss(output.squeeze(), labels_real)
            L1 = nn.L1Loss()

            batch_fake_rgb = ((batch_fake_rgb + 1) * 255 / 2) #go form tanh range to rgb range



            image_rgb_transpose = torch.tensor(np.transpose(np.array(image_rgb.cpu()),(0,3,1,2))).to(device)

            G_loss_L1 = L1(batch_fake_rgb.to(device).view( batch_fake_rgb.to(device).size(0), -1), image_rgb_transpose.to(device).view( image_rgb_transpose.to(device).size(0) , -1 ).float())

            G_loss = G_loss_bce + lam * G_loss_L1
            G_loss.backward()

            D_G_x2 = output.mean().item()

            Generator_optimizer.step()

            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())

            ############### VALIDATION ###################




            # ********************* Validation code ***********************

            # Set the mode of the discriminator and generator to training
            discriminator = discriminator.eval()
            # discriminator2 = discriminator2.eval()
            generator = generator.eval()



            image_norm_val = image_norm_val.to(device)
            image_gray_val = image_gray_val.to(device)
            image_rgb_val  = image_rgb_val.to(device)


            # forward pass
            output_val = discriminator(image_norm_val.float())


            #mean_output = torch.mean(output_val, dim = 3).mean( dim = 2 ).squeeze(1)

            # format labels into tensor vector

            labels_real_val = torch.full((int(batch_size/4),), true_im_label, device=device)

            # The loss on the real batch data


            D1_loss_real_val = BCE_loss(output_val.squeeze(), labels_real_val)


            # Generate fake data - i.e. fake images by inputting black and white images
            batch_fake_rgb_val = generator(image_gray_val.float())

            # change range from [-1, 1] to LAB range so L1 can be computed


            # Train with the Discriminator with fake data

            labels_fake_val = torch.full((int(batch_size/4),), false_im_label, device=device)


            #normalized_batch_fake_rgb = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
            normalized_batch_fake_rgb_val = (((batch_fake_rgb_val + 1 ) * (1)) / (2))

            output1_val = discriminator(normalized_batch_fake_rgb_val.detach())
            #mean_output = torch.mean(output1_val, dim = 3).mean( dim = 2 ).squeeze(1)

            # Compute the loss

            D1_loss_fake_val = BCE_loss(output1_val.squeeze(), labels_fake_val)

            D_loss_val = D1_loss_fake_val + D1_loss_real_val






            # Generator loss

            labels_real_val = torch.full((int(batch_size/4),), true_im_label, device=device)


            output2_val = discriminator(normalized_batch_fake_rgb_val)




            # The generators loss
            G1_loss_bce_val = BCE_loss(output2_val.squeeze(), labels_real_val)
            L1_val = nn.L1Loss()


            batch_fake_rgb = ((batch_fake_rgb_val + 1) * 255 / 2) #go form tanh range to rgb range

            image_rgb_transpose = torch.tensor(np.transpose(np.array(image_rgb_val.cpu()),(0,3,1,2)))


            G_loss_L1_val = L1(batch_fake_rgb.to(device).view(batch_fake_rgb.to(device).size(0),-1), image_rgb_transpose.to(device).view(image_rgb_transpose.to(device).size(0),-1).float())


            G_loss_val = G1_loss_bce_val + lam * G_loss_L1_val

            G_losses_val.append(G_loss_val.item())
            D_losses_val.append(D_loss_val.item())





            ##############################################

            #print('[%d/%d][%d/%d]\tLoss_D1: %.4f\tLoss_G: %.4f'% (epoch, epochs, current_batch, len(dataloader), D_loss_val.item(), G_loss_val.item()))
            if current_batch % 50 == 0:
                #print('[%d/%d][%d/%d]\tLoss val D: %.4f\tLoss val G:\tLoss training D: %.4f\tLoss training G: %.4f'% (epoch, epochs, current_batch, len(dataloader), D_loss_val.item(), G_loss_val.item(), D_loss.item(), G_loss.item() ))
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_val_D: %.4f\tLoss_val_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, epochs, current_batch, len(dataloader), D_loss.item(), G_loss.item(), D_loss_val.item(), G_loss_val.item(),  D_x, D_G_x1, D_G_x2))

                #plot_losses(path+"/", G_losses, D_losses, G_losses_val, D_losses_val, epoch, current_batch)
                plot_losses(path, G_losses, D_losses, G_losses_val, D_losses_val, epoch, current_batch)


            if current_batch % 100 == 0:

                # print(image_gray_un_norm_val[-1].shape)
                # input()
                # save_image(path+"/", image_gray_un_norm_val[-1], epoch, current_batch, device, False)
                # save_image(path+"/", image_rgb_val[-1], epoch, current_batch, device, True)


                save_image(path, image_gray_un_norm_val[-1], epoch, current_batch, device, False)
                save_image(path, image_rgb_val[-1], epoch, current_batch, device, True)


                #plot_losses(G_losses, D_losses, epoch, current_batch)
                wrapped_bw_im = image_gray_val[-1].unsqueeze(0)
                wrapped_bw_im = wrapped_bw_im.to(device)
                #img_rgb = convert_lab_to_rgb(image_l[-1], image_ab[-1])


                #img = validate_generator(path + "/result_pics/",device, epoch, current_batch, wrapped_bw_im, generator)
                #save_path, epoch, current_batch, bw_im, generator
                validate_generator(path + "result_pics/", epoch, current_batch, wrapped_bw_im, generator)
                #reference_img = validate_generator("reference_pics",device, epoch, current_batch, reference_bw, generator, True)


                #reference_list.append(reference_img)


            if current_batch == 400:
                file_name_generator = "generator_model"

                file_name_discriminator = "discriminator_model"


                file_name_discriminator_optimizer = "Discriminator_optimizer"

                file_name_generator_optimizer = "Generator_optimizer"

                # /home/projektet/
                torch.save(discriminator.state_dict(),path + "models/" +file_name_discriminator + "_"  +str(current_batch) + "_" + str(epoch) +  ".pt")
                #torch.save(Discriminator_optimizer, path + "models/" + file_name_discriminator_optimizer + "_"  +str(current_batch) + "_" + str(epoch) +  ".pt")


                torch.save(generator.state_dict(), path + "models/" + file_name_generator +"_" + str(current_batch) + "_" + str(epoch) +  ".pt")
                #torch.save(Generator_optimizer, path + "models/" + file_name_generator_optimizer + "_"  +str(current_batch) + "_" + str(epoch) +  ".pt")

                np.save(path +'G_losses.npy', G_losses)
                np.save(path +'D_losses.npy', D_losses)
                np.save(path +'G_losses_val.npy', G_losses_val)
                np.save(path +'D_losses_val.npy', D_losses_val)

            # Set the mode of the discriminator to training
            discriminator = discriminator.train()


            # Set the mode of the generator to training
            generator = generator.train()


    return 0


# def validate_generator(path,device,epoch, current_batch, bw_im, generator, padding_sz=2,  norm=True):
#     with torch.no_grad():
#         fake_im = generator(bw_im.float()).detach().cpu()
#
#
#
#     bw_im = bw_im.squeeze(0)
#     fake_im = change_range_ab(fake_im)
#     #fake_im = fake_im.squeeze()
#     img_rgb = convert_lab_to_rgb(fake_im.to(device))
#     plt.axis("off")
#     img = vutils.make_grid(img_rgb, padding = 0, normalize = norm )
#     plt.imshow(np.transpose(img,(1,2,0)), animated=True)
#
#     plt.savefig(str(epoch) + "_" + str(current_batch) + "_color_generated.png")
#     plt.close()
#
#
#     return img


# def validate_generator(path,device,epoch, current_batch, bw_im, generator, padding_sz=2,  norm=True):
#
#     with torch.no_grad():
#         fake_im = generator(bw_im).detach().cpu()
#
#
#     #plt.savefig(path + "result_pics/"+ str(epoch) + "_" + str(current_batch) + "_color.png")
#     # plt.savefig("/home/projektet/network_v8/result_pics/" +str(epoch) + "_" + str(current_batch) + "_color.png")
#    # print(fake_im.shape)
#
#     img = Image.fromarray(np.transpose(np.array(fake_im.squeeze(0).detach()),(1,2,0)), 'RGB')
#     img.save(path +"/"+ str(epoch) + "_" + str(current_batch) + "_Color_generated.png")
#
#
#     return img

def validate_generator(save_path, epoch, current_batch, bw_im, generator):
    """ Generates a colored image and saves it
    """
    with torch.no_grad():
        fake_im = generator(bw_im).detach().cpu()

        img = (fake_im +1)*255/2
        img = img.squeeze(0)
        img = np.transpose(img, (1,2,0))
        img = np.array(img)
        img = img.astype(np.uint8)

        image = Image.fromarray(img, 'RGB')
        #image.save(save_path +"Color_generated_epoch_" + str(epoch)+ "_batch_" +str(current_batch) + ".png")
        image.save(save_path + str(epoch) + "_" + str(current_batch) + "_generated.png")





def change_range_ab(im_lab):
    #https://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space

    #real range
    l_lower = 0
    l_upper = 100
    a_lower = -86.125
    a_upper = 98.254
    b_lower = -107.863
    b_upper = 94.482


    #from training set
    # a_upper = 93.2470357189119
    # a_lower = -86.18302974439501
    # #---
    # b_upper = 94.47812227647825
    # b_lower = -106.24797040432395

    generated = im_lab.data.numpy()
    #NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    #print(generated.shape)

    l = l_lower +  (generated[0,0,:,:] - (-1)) * (l_upper-l_lower) / (1 - (-1)) #).astype(int)
    a = a_lower +  (generated[0,1,:,:] - (-1)) * (a_upper-a_lower) / (1 - (-1)) #).astype(int)
    b = b_lower +  (generated[0,2,:,:] - (-1)) * (b_upper-b_lower) / (1 - (-1)) #).astype(int)
    #b = generated[0,1,:,:] * (b_upper-b_lower) + b_lower#.astype(int)
    #image_ab = generated[0,:,:,:] + 1) * 255 / 2
    ##print("a size", a.shape)


    image_ab = torch.tensor([l, a, b])
    #image_ab = torch.cat((a, b), 0)

    return image_ab


def change_range_lab_batch(im_lab):
    #https://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space

    #real range
    l_lower = 0
    l_upper = 100
    a_lower = -86.125
    a_upper = 98.254
    b_lower = -107.863
    b_upper = 94.482


    #from training set
    #
    # a_upper = 93.2470357189119
    # a_lower = -86.18302974439501
    # #---
    # b_upper = 94.47812227647825
    # b_lower = -106.24797040432395

    generated = im_lab.data.cpu().numpy()
    #NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin


    l = torch.tensor(l_lower +  (generated[:,0,:,:] - (-1)) * (l_upper-l_lower) / (1 - (-1))).unsqueeze(1) #).astype(int)
    a = torch.tensor(a_lower +  (generated[:,1,:,:] - (-1)) * (a_upper-a_lower) / (1 - (-1))).unsqueeze(1) #).astype(int)
    b = torch.tensor(b_lower +  (generated[:,2,:,:] - (-1)) * (b_upper-b_lower) / (1 - (-1))).unsqueeze(1) #).astype(int)
    #b = generated[0,1,:,:] * (b_upper-b_lower) + b_lower#.astype(int)
    #image_ab = generated[0,:,:,:] + 1) * 255 / 2


    img_concat = torch.cat((l, a), 1).cpu()


    image_lab = torch.cat((img_concat, b), 1).cpu()

    return image_lab


def change_range_lab_to_tanh_batch(im_lab):
    #https://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space

    #real range
    l_lower = 0
    l_upper = 100
    a_lower = -86.125
    a_upper = 98.254
    b_lower = -107.863
    b_upper = 94.482

    tanh_lower = -1
    tanh_upper = 1


    #from training set
    # a_upper = 93.2470357189119
    # a_lower = -86.18302974439501
    # #---
    # b_upper = 94.47812227647825
    # b_lower = -106.24797040432395

    generated = im_lab.data.numpy()
    #NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin



    l = torch.tensor(l_lower +  (generated[:,0,:,:] - (-1)) * (l_upper-l_lower) / (1 - (-1))).unsqueeze(1) #).astype(int)
    a = torch.tensor(a_lower +  (generated[:,1,:,:] - (-1)) * (a_upper-a_lower) / (1 - (-1))).unsqueeze(1) #).astype(int)
    b = torch.tensor(b_lower +  (generated[:,2,:,:] - (-1)) * (b_upper-b_lower) / (1 - (-1))).unsqueeze(1) #).astype(int)
    #b = generated[0,1,:,:] * (b_upper-b_lower) + b_lower#.astype(int)
    #image_ab = generated[0,:,:,:] + 1) * 255 / 2


    img_concat = torch.cat((l, a), 1).cpu()


    image_lab = torch.cat((img_concat, b), 1).cpu()

    return image_lab


def read_gray_img(image_name_gray):
    #to be able to save a gray image, you first have to read itself.
    # hashtag logic life

    transform = transforms.ToTensor()
    image_gray = io.imread(str(image_name_gray))
    image_gray = transform(image_gray)
    return image_gray

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

def convert_lab_to_rgb(image_ab):


    lab = image_ab.cpu()
    l = lab[0,:,:].unsqueeze(2)
    a = lab[1,:,:].unsqueeze(2)
    b = lab[2,:,:].unsqueeze(2)

    lab_t1 = torch.cat((l,a), 2)
    lab_t = np.array(torch.cat((lab_t1, b), 2))




    # img_jacob = color.lab2xyz(np.transpose(np.array(lab)))
    # print(img_jacob.shape)
    #
    #
    # z = img_jacob[:,:,2]
    #
    # #if z<o, set z to 0
    # if np.any(z < 0):
    #     invalid = np.nonzero(z < 0)
    #     z[invalid] = 0
    #
    # img_jacob[:,:,2] = z


    img_concat = torch.tensor(color.lab2rgb(lab_t))


    r = img_concat[:,:,0].unsqueeze(0)
    g = img_concat[:,:,1].unsqueeze(0)
    b = img_concat[:,:,2].unsqueeze(0)


    lab_t1 = torch.cat((r,g), 0)
    lab_t = torch.cat((lab_t1, b), 0)

    #img_concat = color.xyz2rgb(img_jacob)

    #convert back to tensor to plot
    img_rgb = lab_t#torch.tensor(lab_t)

    return img_rgb

def convert_lab_to_rgb_batch(image_lab_batch):


    lab = np.array(image_lab_batch)
    l = lab[:,0,:,:]
    a = lab[:,1,:,:]
    b = lab[:,2,:,:]


    img_jacob = color.lab2xyz(np.transpose(np.array(lab)))


    z = img_jacob[:,:,:,2]

    #if z<o, set z to 0
    if np.any(z < 0):
        invalid = np.nonzero(z < 0)
        z[invalid] = 0

    img_jacob[:,:,:,2] = z


    #img_concat = color.lab2rgb(np.transpose(np.array(img_concat)))

    img_concat = color.xyz2rgb(img_jacob)

    #convert back to tensor to plot
    img_rgb_batch = torch.tensor(np.transpose(img_concat))

    return img_rgb_batch



# def save_image(path, img, epoch, current_batch, device, color):
#     """ Saves a black and white image
#     """
#
#     plt.figure()
#     plt.axis("off")
#     plt.imshow(np.transpose(vutils.make_grid(img.to(device)[:64], padding=0, normalize=True).cpu(),(1,2,0)))
#     #path = "/home/jacob/Documents/DD2424 projekt/lab/l_to_lab/"
#     if(color):
#         plt.savefig(path + "result_pics/"+ str(epoch) + "_" + str(current_batch) + "_color.png")
#         # plt.savefig("/home/projektet/network_v8/result_pics/" +str(epoch) + "_" + str(current_batch) + "_color.png")
#     else:
#         plt.savefig(path + "result_pics/"+ str(epoch) + "_" + str(current_batch) + "_BW.png")
#         # plt.savefig("/home/projektet/network_v8/result_pics/" +str(epoch) + "_" + str(current_batch) + "_BW.png")
#     #plt.clf()
#     plt.close()
    #plt.show()





# def save_image(path, img, epoch, current_batch, device, color):
#     """ Saves a black and white image
#     """
#     if(color):
#         #img = Image.fromarray(img)
#         img = Image.fromarray(np.array(img.detach().cpu()))
#         img.save(path + "result_pics/"+ str(epoch) + "_" + str(current_batch) + "_color.png")
#     else:
#         #img = Image.fromarray(np.array(img.detach()), 'RGB')
#         img = Image.fromarray(np.array(img.detach().cpu()).astype(np.uint8))
#         img.save(path + "result_pics/"+ str(epoch) + "_" + str(current_batch) + "_BW.png")

def save_image(path, img, epoch, current_batch, device, color):
    """ Saves a black and white image
    """
    if(color):
        #img = Image.fromarray(img)
        img = Image.fromarray(np.array(img.detach().cpu()))
        img.save(path + "result_pics/"+ str(epoch) + "_" + str(current_batch) + "_color.png")
        #image.save(save_path +"Color_generated_epoch_" + str(epoch)+ "_batch_" +str(current_batch) + ".png")
    else:
        #img = Image.fromarray(np.array(img.detach()), 'RGB')
        img = Image.fromarray(np.array(img.detach().cpu()).astype(np.uint8))
        img.save(path + "result_pics/"+ str(epoch) + "_" + str(current_batch) + "_BW.png")




def plot_losses(path, G_losses, D_losses, G_losses_val, D_losses_val, epoch, current_batch):

    """ creates two plots. One for the Generator loss and one for the Discriminator loss and saves these figures
    """
    D_loss_fig = plt.figure('D_cost' + str(epoch) + '_' + str(current_batch))
    plt.plot(D_losses, color='b', linewidth=1.5, label='D cost training')  # axis=0
    plt.plot(D_losses_val, color='purple', linewidth=1.5, label='D cost validation')  # axis=0
    plt.legend(loc='upper left')
    #D_loss_fig.savefig('/home/projektet/network_v8/plots/D_cost' + str(epoch) + '_' + str(current_batch) +'.png', dpi=D_loss_fig.dpi)
    #path = "/home/jacob/Documents/DD2424 projekt/lab/l_to_lab/"
    D_loss_fig.savefig(path + 'plots/'+ 'D_cost.png', dpi=D_loss_fig.dpi)
    #plt.clf()
    plt.close(D_loss_fig)



    G_loss_fig = plt.figure('G cost' + str(epoch) + '_' + str(current_batch))
    plt.plot(G_losses, color='b', linewidth=1.5, label='G cost training')  # axis=0
    plt.plot(G_losses_val, color='purple', linewidth=1.5, label='G cost validation')  # axis=0
    plt.legend(loc='upper left')
    # G_loss_fig.savefig('/home/projektet/network_v8/plots/G_cost' + str(epoch) + '_' + str(current_batch) +'.png', dpi=G_loss_fig.dpi)
    G_loss_fig.savefig(path + 'plots/'+ 'G_cost.png', dpi=G_loss_fig.dpi)
    #plt.clf()
    plt.close(G_loss_fig)



def tensor_format_labels(b_size, label_vec, device):

    label = torch.full((b_size,8,8), label_vec, device=device)


    return label


def reshape_to_vector(output):
    return torch.squeeze(output) #output.view(-1)

def update_patch_function( epoch, min = False ):
    """ Initially we want to use the mean when we calculate the patch loss. But as the model stabalizes we want
        start focusing on the bad areas created by the generator. Thus we instead switch to a mean function.
    """
    if epoch > 13:
        min_func = True
        return min_func

    else:
        return False



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
