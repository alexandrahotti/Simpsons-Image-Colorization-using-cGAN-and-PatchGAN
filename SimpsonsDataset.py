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






if __name__ == "__main__":
    #imports
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from skimage import io
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    import os
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    #set seed if you want
    torch.manual_seed(12)
    #point to dataroot_train to your training dataset folder. it can contain subfolders
    #point to dataroot_validation to your validation dataset folder. it can contain subfolders
    dataroot_train = "/home/jacob/Documents/DD2424 dataset/trainset/"

    trainset = SimpsonsDataset(datafolder = dataroot_train, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    #one dataloader each for tarin/validation/testing

    batch_size =10
    workers=2

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(len(trainset))

    #these should show the same imgs, just color and gray versions
    num_epochs = 1
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            print(data[1].shape)
            plt.subplot(2, 1, 1)
            #plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Color Images")
            plt.imshow(np.transpose(vutils.make_grid(data[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            #plt.show()
            plt.subplot(2, 1, 2)
            #plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Gray Images")
            plt.imshow(np.transpose(vutils.make_grid(data[1].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()
