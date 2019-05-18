# An Average PatchGAN Network
This branch contains a network which was selected as one of the best performin networks.

## Architecture
Two different versions of PatchGAN networks were constructed.
They share discriminator architecture, which outputs
1x32x32 ’patches’. This discriminator architecture is
displayed on the master branch. The size of the output patches was 
chosen so that each patch in the output corresponds to a 8x8
patch in the input image. The generator for the PatchGAN
networks are identical to the the one used in the previous
section and is displayed on the master branch. The learning 
rate forthese networks was set to 4e-4 and learning rate scheduling
was used to decay the learning rate by 0.5 every 5 epochs.

The first network is Avg-PatchGAN. In this network, to
receive a final likelihood score for an image the mean of
the patches is computed, as in (Isola et al., 2017), and as
described in section 4.3.1 in our paper linked in the master branch. 
This network is used as a baseline implementation.
