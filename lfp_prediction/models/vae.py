import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()


class VAE(nn.Module):
    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dims=latent_dims)
        self.decoder = Decoder(latent_dims=latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
