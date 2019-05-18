# A Conditional GAN - Network V2
Three conditional GANs were trained. The following branch contains a network that was selected as one of the better performing networks.

## Architecture
Network V2, has the same amount of layers as network V1 but uses skip connections. The generator
architectures filter size is not constant but instead has a size corresponding to the sizeof the 
input image to the layer. The first layer has a filter size of 6x6 which tapers off to 2x2 in the
encoder in the generator. In the decoder part of the generator the filter sizes gradually decrease
from 2x2 to 6x6 in the last layer. The discriminator similarly has a 6x6 filter size in the first 
layer and tapers off to 1x1 in the last layer. The reasoning behind this "pyramid" design is that 
larger  images shoulduse larger filters since the features, such as Homer’s head, are large. As the 
image size is decreased, thesize of the features are decreased and the filter size should thus be 
decreased. L2 regularization in the discriminator was used to decrease overfitting, with the penalty 
set to 1e-4.
