# A Conditional GAN - Network V1
Three conditional GANs were trained. The following branch contains a network that was not selected as one of the best performing networks.

## Architecture
Network V1, has a structure similar to the one used in (Nazeri et al., 2018),
except that it does not have skip connections. The filter size in this generator 
is consistently 3x3, irrespective of layer. It has 10 layers in the generator. 
The discriminator has 8 filters, where the first 5 filters are 3x3, the sixth is 
5x5, the seventh is 1x1, and the eighth is 4x4.
