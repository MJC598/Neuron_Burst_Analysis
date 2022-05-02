import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, optim
from torch.fft import rfft, irfft

from .base_vae import BaseVAE
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Union, Any


class LSTMVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self, params: dict, **kwargs):
        super(LSTMVAE, self).__init__()
        self.save_hyperparameters(params)

        self.params = params
        try:
            self.latent_dim = self.params['latent_dim']
            self.in_channels = self.params['in_channels']
            self.in_length = self.params['in_length']
        except KeyError as e:
            raise ValueError('Parameter {} not specified'.format(e.args[0]))

        self.example_input_array = torch.empty((1, 1, self.in_length))  # Literally just to graph the plot

        self.beta = 4 if 'beta' not in self.params else self.params['beta']
        self.gamma = 1500. if 'gamma' not in self.params else self.params['gamma']
        self.loss_type = 'H' if 'loss_type' not in self.params else self.params['loss_type']
        self.C_max = (torch.Tensor([50])) \
            if 'max_capacity' not in self.params else \
            (torch.Tensor(self.params['max_capacity']))

        self.C_stop_iter = 1e5 if 'capacity_max_iter' not in self.params else self.params['capacity_max_iter']
        self.hidden_dims = None if 'hidden_dims' not in self.params else self.params['hidden_dims']

        self.curr_device = None

        self.enc_lstm = nn.LSTM(input_size=1,
                                hidden_size=self.in_length,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5,
                                bidirectional=False)

        self.enc_linear = nn.Linear(in_features=self.in_length, out_features=64)
        self.fc_mu = nn.Linear(in_features=64, out_features=self.latent_dim)
        self.fc_var = nn.Linear(in_features=64, out_features=self.latent_dim)

        self.dec_linear = nn.Linear(in_features=self.latent_dim, out_features=self.in_length)

        self.dec_lstm = nn.LSTM(input_size=1,
                                hidden_size=1,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.5,
                                bidirectional=False)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encode the input and return a mean and log variance
        :param input: Input Tensor of shape (N, H_in, L)
        :return: List of Tensors, first is mean second is log variance. Both shapes (latent_dim, 1)
        """
        input = input.view(-1, self.in_length, 1)  # Transform to fit to LSTM (N, L, H_in)
        result, (_, _) = self.enc_lstm(input)
        result = self.enc_linear(result[:, -1, :])
        result = torch.squeeze(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.dec_linear(z)
        result = result.view(-1, self.in_length, 1)
        result, (_, _) = self.dec_lstm(result)
        result = result.view(-1, 1, self.in_length)  # transform to match label (N, H_out, L)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Union[Tensor, Any]]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    @staticmethod
    def fft_loss(output: torch.Tensor, target: torch.Tensor,
                 norm_factor: Optional[torch.Tensor] = None) -> torch.Tensor:
        # fft_data = irfft(rfft(output) - rfft(target))
        fft_data = (rfft(output).real - rfft(target).real)
        loss = torch.mean(torch.matmul(torch.square(fft_data), norm_factor)) * 1e-8
        # loss = torch.sum(torch.matmul(torch.square(fft_data), norm_factor)) * 1e-8
        # loss = torch.mean(torch.square(fft_data)) * 1e-8
        return loss

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = torch.clamp(self.fft_loss(recons,
                                                input,
                                                norm_factor=self.trainer.datamodule.norm_factor.to(self.curr_device)),
                                  min=1e-8,
                                  max=1e8)
        # recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.clamp((torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)),
                               min=1e-8,
                               max=1e8)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        if (torch.isnan(recons).any() or
                torch.isnan(input).any() or
                torch.isnan(mu).any() or
                torch.isnan(log_var).any() or
                torch.isnan(recons_loss).any() or
                torch.isnan(kld_loss).any()):
            raise ValueError("NaN occurred during loss calcuation. Parameters are: "
                             "\nrecons: {} "
                             "\ninput: {} "
                             "\nmu: {} "
                             "\nlog_var: {} "
                             "\nrecon_loss: {} "
                             "\nkld_loss: {}".format(recons, input, mu, log_var, recons_loss, kld_loss))

        return {'loss': loss,
                'recon_loss': recons_loss,
                'kld_loss': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs) -> STEP_OUTPUT:
        raw_signal, labels = batch
        self.curr_device = raw_signal.device

        results = self.forward(raw_signal, labels=labels)
        train_loss = self.loss_function(*results,
                                        M_N=self.params['kld_weight'],
                                        optimizer_idx=optimizer_idx,
                                        batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raw_signal, labels = batch
        self.curr_device = raw_signal.device

        results = self.forward(raw_signal, labels=labels)
        val_loss = self.loss_function(*results,
                                      M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                      optimizer_idx=optimizer_idx,
                                      batch_idx=batch_idx)

        self.log_dict({"val_{}".format(key): val.item() for key, val in val_loss.items()}, sync_dist=True)
        return val_loss['loss']

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                     gamma=self.params['scheduler_gamma'])
        scheds.append(scheduler)
        return optims, scheds

    def on_validation_end(self) -> None:
        self.sample(1, self.curr_device)
