# The Simpsons Image Colorization using a Conditional Generative Adversarial Networks (cGANs) and a Markovian discriminator (PatchGAN)


## Prerequisites
- Pytorch 1.1

### Dataset
The data consisted of 36 000 images split into a training, a validation and a test set. The test set contains 1 000 images,
the validation set contains 7 000 images, and the training set contains 28 000 images. The images were taken from season
10-27 of The Simpsons. From these episodes, one image was taken every 15 seconds, excluding the first and last minute to
avoid capturing the intro and outro multiple times. The images were down-sampled to a resolution of 256x256 and transformed
to gray scale using the imagemagick and ffmpeg software suites.

## Networks Architecture
The network architecture of the final best performing networks are depicted below.

Below is the architecture used for the generator in Average-PatchGAN (branch: network_v10), MIN-PatchGAN (branch: network_v9), cGAN (branch: network_v10).

The generator architecture is inspired by  [U-Net](https://arxiv.org/abs/1505.04597):  The architecture of the model is symmetric, with `n` encoding units and `n` decoding units. The contracting path consists of 4x4 convolution layers with stride 2 for downsampling, each followed by batch normalization and Leaky-ReLU activation function with the slope of 0.2. The number of channels are doubled after each step. Each unit in the expansive path consists of a 4x4 transposed convolutional layer with stride 2 for upsampling, concatenation with the activation map of the mirroring layer in the contracting path, followed by batch normalization and ReLU activation function. The last layer of the network is a 1x1 convolution which is equivalent to cross-channel parametric pooling layer. We use `tanh` function for the last layer.

<p align='center'>  
  <img src='architecture/Generator.JPG' width="60%" height="60%"
   />
  
</p>
<p align='left'>  
  <img src='architecture/Discriminator_PatchGAN.JPG' width="30%" height="30%" />
</p>
<p align='left'>  
  <img src='architecture/Discriminator_Network_2.JPG'  width="45%" height="45%"/>
</p>
