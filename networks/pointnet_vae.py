import torch
import torch.nn as nn

from .pointnet_encoder import PointNetEncoder, ResnetPointnet
from .pointnet2_encoder import PointNet2Encoder
from .residual_mlp_vae import ResidualMLPDecoder


class PointNetLatentVAE(nn.Module):
    def __init__(
        self,
        latent_dim=16,
        output_dim=256,
        encoder_type="pointnet2",
        decoder_hidden_dims=(128, 256, 256),
        decoder_blocks=1,
        decoder_activation="gelu",
        decoder_dropout=0.0,
        decoder_layernorm=True,
        use_kl=True,
    ):
        super().__init__()
        self.use_kl = bool(use_kl)

        encoder_type = encoder_type.lower()
        if encoder_type in ("resnet_pointnet", "pointnet"):
            self.encoder = ResnetPointnet(latent_size=latent_dim, kl_div_loss=self.use_kl)
        elif encoder_type in ("pointnet2", "pointnet++"):
            self.encoder = PointNet2Encoder(latent_size=latent_dim, kl_div_loss=self.use_kl)
        elif encoder_type == "pointnet_encoder":
            self.encoder = PointNetEncoder(latent_size=latent_dim, kl_div_loss=self.use_kl)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.decoder = ResidualMLPDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=decoder_hidden_dims,
            num_blocks=decoder_blocks,
            activation=decoder_activation,
            dropout=decoder_dropout,
            use_layernorm=decoder_layernorm,
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, points):
        if self.use_kl:
            mu, logvar = self.encoder(points)
            z = self.reparameterize(mu, logvar)
        else:
            mu = self.encoder(points)
            logvar = torch.zeros_like(mu)
            z = mu
        z_hat = self.decoder(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_hat": z_hat,
        }
