# A Softmin, Average PatchGAN Network
This branch contains a network which was selected as one of the best performin networks.

## Architecture
Two different versions of PatchGAN networks were constructed.
They share discriminator architecture, which outputs
1x32x32 ’patches’. This discriminator architecture is
displayed on the master branch. The size of the output patches was
chosen so that each patch in the output corresponds to a 8x8
patch in the input image. The generator for the PatchGAN
networks are identical to the the one used in the previous
section and is displayed on the master branch. The learning rate for
these networks was set to 4e-4 and learning rate scheduling
was used to decay the learning rate by 0.5 every 5 epochs.

This network, Min-PatchGAN, has the same generator
and discriminator as Avg-PatchGAN (branch: network_v10) but the final
likelihood is computed differently. From the first epoch up
to a certain point when the image quality plateaus, the likelihood
will be computed using the mean of the patches, just as
in Avg-PatchGAN. After this point and until the last epoch,
the likelihood will be computed using the method outlined
in section 4.3.2 in the paper linked on the master branch. 
The reasoning behind the switch is that the mean processing 
outlined in section 4.3.1 should achieve a likelihood score 
which is approximately correct, and the min based approach 
should achieve a likelihood score which is more fine tuned. 
To find the point when the switch from mean to weighted should 
be made, an experiment was conducted. This experiment is outlined
in the section 6.5 in our paper linked on the master branch. 
