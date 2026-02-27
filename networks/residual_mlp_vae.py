import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported activation: {name}")


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, activation="gelu", dropout=0.0, use_layernorm=True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = _get_activation(activation)()
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(dim) if use_layernorm else None

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual
        if self.norm is not None:
            x = self.norm(x)
        return x


class ResidualMLPStage(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_blocks=1,
        activation="gelu",
        dropout=0.0,
        use_layernorm=True,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act = _get_activation(activation)()
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    out_dim,
                    activation=activation,
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResidualMLPStack(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        num_blocks=1,
        activation="gelu",
        dropout=0.0,
        use_layernorm=True,
    ):
        super().__init__()
        dims = list(hidden_dims)
        if not dims:
            raise ValueError("hidden_dims must be non-empty")
        stages = []
        prev_dim = in_dim
        for dim in dims:
            stages.append(
                ResidualMLPStage(
                    prev_dim,
                    dim,
                    num_blocks=num_blocks,
                    activation=activation,
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                )
            )
            prev_dim = dim
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class ResidualMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim=256,
        latent_dim=16,
        hidden_dims=(256, 128),
        num_blocks=1,
        activation="gelu",
        dropout=0.0,
        use_layernorm=True,
    ):
        super().__init__()
        self.backbone = ResidualMLPStack(
            input_dim,
            hidden_dims,
            num_blocks=num_blocks,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        final_dim = hidden_dims[-1]
        self.mu_head = nn.Linear(final_dim, latent_dim)
        self.logvar_head = nn.Linear(final_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class ResidualMLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=16,
        output_dim=256,
        hidden_dims=(128, 256, 256),
        num_blocks=1,
        activation="gelu",
        dropout=0.0,
        use_layernorm=True,
    ):
        super().__init__()
        self.backbone = ResidualMLPStack(
            latent_dim,
            hidden_dims,
            num_blocks=num_blocks,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        h = self.backbone(z)
        return self.out(h)


class ResidualMLPVAE(nn.Module):
    def __init__(
        self,
        input_dim=256,
        latent_dim=16,
        encoder_hidden_dims=(256, 128),
        decoder_hidden_dims=(128, 256, 256),
        num_blocks=1,
        activation="gelu",
        dropout=0.0,
        use_layernorm=True,
        use_kl=True,
    ):
        super().__init__()
        self.encoder = ResidualMLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            num_blocks=num_blocks,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.use_kl = bool(use_kl)
        self.decoder = ResidualMLPDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden_dims,
            num_blocks=num_blocks,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        if self.use_kl:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        z_hat = self.decoder(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_hat": z_hat,
        }


def expand_latent_to_points(latent, xyz):
    if xyz.dim() == 3 and latent.dim() == 2:
        batch_size, num_points, _ = xyz.shape
        latent = latent[:, None, :].expand(batch_size, num_points, latent.shape[-1])
        latent = latent.reshape(-1, latent.shape[-1])
        xyz = xyz.reshape(-1, 3)
    elif xyz.dim() == 2 and latent.dim() == 2 and latent.shape[0] == 1:
        latent = latent.expand(xyz.shape[0], latent.shape[-1])
    return latent, xyz


class ResidualMLPVAEWithDeepSDF(nn.Module):
    def __init__(
        self,
        input_dim=256,
        latent_dim=16,
        encoder_hidden_dims=(256, 128),
        decoder_hidden_dims=(128, 256, 256),
        num_blocks=1,
        activation="gelu",
        dropout=0.0,
        use_layernorm=True,
        use_kl=True,
        sdf_decoder=None,
        sdf_decoder_kwargs=None,
    ):
        super().__init__()
        self.vae = ResidualMLPVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            num_blocks=num_blocks,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
            use_kl=use_kl,
        )
        if sdf_decoder is None:
            if sdf_decoder_kwargs is None:
                raise ValueError("Provide sdf_decoder or sdf_decoder_kwargs")
            from networks.deep_sdf_decoder import Decoder

            self.sdf_decoder = Decoder(**sdf_decoder_kwargs)
        else:
            self.sdf_decoder = sdf_decoder

    def forward(self, x, xyz):
        vae_out = self.vae(x)
        latent, xyz = expand_latent_to_points(vae_out["z_hat"], xyz)
        sdf_in = torch.cat([latent, xyz], dim=-1)
        pred_sdf = self.sdf_decoder(sdf_in)
        vae_out["pred_sdf"] = pred_sdf
        return vae_out


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def log_density_gaussian(z, mu, logvar):
    log2pi = math.log(2.0 * math.pi)
    return -0.5 * (log2pi + logvar + (z - mu).pow(2) / logvar.exp())


def vae_loss(
    z_hat,
    z_target,
    mu,
    logvar,
    recon_weight=1.0,
    kl_weight=1.0,
    recon_loss="mse",
):
    if recon_loss == "l1":
        recon = F.l1_loss(z_hat, z_target, reduction="mean")
    elif recon_loss == "mse":
        recon = F.mse_loss(z_hat, z_target, reduction="mean")
    else:
        raise ValueError(f"Unsupported recon_loss: {recon_loss}")
    kl = kl_divergence(mu, logvar)
    total = recon_weight * recon + kl_weight * kl
    return total, recon, kl


def _covariance_matrix(x):
    if x.dim() != 2:
        raise ValueError("covariance expects a 2D tensor [N, D]")
    n = x.shape[0]
    if n <= 1:
        return torch.zeros((x.shape[1], x.shape[1]), device=x.device, dtype=x.dtype)
    x_centered = x - x.mean(dim=0, keepdim=True)
    return (x_centered.t() @ x_centered) / float(n - 1)


def dip_vae_loss(
    z_hat,
    z_target,
    mu,
    logvar,
    recon_weight=1.0,
    kl_weight=1.0,
    dip_lambda_od=1.0,
    dip_lambda_d=1.0,
    dip_type="ii",
    recon_loss="mse",
):
    if recon_loss == "l1":
        recon = F.l1_loss(z_hat, z_target, reduction="mean")
    elif recon_loss == "mse":
        recon = F.mse_loss(z_hat, z_target, reduction="mean")
    else:
        raise ValueError(f"Unsupported recon_loss: {recon_loss}")

    kl = kl_divergence(mu, logvar)

    dip_type = str(dip_type).lower()
    cov_mu = _covariance_matrix(mu)
    if dip_type in ("ii", "2", "dip_vae_ii", "dip_vae2", "dip_ii", "dip2"):
        var = torch.exp(logvar)
        cov_z = cov_mu + torch.diag(var.mean(dim=0))
    else:
        cov_z = cov_mu

    diag = torch.diag(cov_z)
    off_diag = cov_z - torch.diag(diag)
    off_loss = torch.sum(off_diag.pow(2))
    diag_loss = torch.sum((diag - 1.0).pow(2))
    dip_loss = dip_lambda_od * off_loss + dip_lambda_d * diag_loss

    total = recon_weight * recon + kl_weight * kl + dip_loss
    return total, recon, kl, dip_loss, off_loss, diag_loss


def beta_tcvae_loss(
    z_hat,
    z_target,
    z,
    mu,
    logvar,
    recon_weight=1.0,
    kl_weight=1.0,
    tc_alpha=1.0,
    tc_beta=6.0,
    tc_gamma=1.0,
    recon_loss="mse",
    dataset_size=None,
):
    if recon_loss == "l1":
        recon = F.l1_loss(z_hat, z_target, reduction="mean")
    elif recon_loss == "mse":
        recon = F.mse_loss(z_hat, z_target, reduction="mean")
    else:
        raise ValueError(f"Unsupported recon_loss: {recon_loss}")

    batch_size = z.shape[0]
    if dataset_size is None:
        dataset_size = batch_size
    dataset_size = max(int(dataset_size), 1)

    log_qz_condx = log_density_gaussian(z, mu, logvar).sum(dim=1)

    z_expand = z.unsqueeze(1)
    mu_expand = mu.unsqueeze(0)
    logvar_expand = logvar.unsqueeze(0)
    log_qz_x = log_density_gaussian(z_expand, mu_expand, logvar_expand)

    log_qz = torch.logsumexp(log_qz_x.sum(2), dim=1) - math.log(dataset_size)
    log_qz_prod = torch.logsumexp(log_qz_x, dim=1) - math.log(dataset_size)
    log_prod_qz = log_qz_prod.sum(dim=1)

    log_pz = log_density_gaussian(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=1)

    mi = (log_qz_condx - log_qz).mean()
    tc = (log_qz - log_prod_qz).mean()
    dwkl = (log_prod_qz - log_pz).mean()
    total_kl = mi + tc + dwkl

    weighted_kl = tc_alpha * mi + tc_beta * tc + tc_gamma * dwkl
    total = recon_weight * recon + kl_weight * weighted_kl
    return total, recon, total_kl, mi, tc, dwkl


def deep_sdf_loss(
    pred_sdf,
    sdf_gt,
    latent_vecs,
    code_reg_lambda=1e-4,
    code_reg_weight=1.0,
):
    num_sdf_samples = float(pred_sdf.shape[0])
    sdf_loss = F.l1_loss(pred_sdf, sdf_gt, reduction="sum") / num_sdf_samples
    l2_size_loss = torch.sum(torch.norm(latent_vecs, dim=1))
    reg_loss = code_reg_lambda * code_reg_weight * l2_size_loss / num_sdf_samples
    total = sdf_loss + reg_loss
    return total, sdf_loss, reg_loss


def linear_warmup(step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step) / float(warmup_steps))
