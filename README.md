# A Conditional GAN - Network V3
Three conditional GANs were trained. The following branch contains a network that was not selected as one of the better performing networks.

## Architecture
The third network, V3, has a pyramid architecture but with
different filter sizes compared to network 2. In the generator,
the filters are 8x8 in the input layer and taper off similarly
to network 2 to a size of 2x2 in the encoding part and in
the decoder part the filters grow from 2x2 to 8x8. Also,
Adam optimization was used in both the generator and the
discriminator, similarly to the architecture used in (Radford
et al., 2015)
