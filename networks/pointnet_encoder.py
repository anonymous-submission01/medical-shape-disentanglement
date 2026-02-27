import torch
import torch.nn as nn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class PointNetEncoder(nn.Module):
    def __init__(self, latent_size, input_channels=3, kl_div_loss=False):
        super(PointNetEncoder, self).__init__()
        self.kl_div_loss = kl_div_loss

        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.max_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_mu = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
        )

        self.fc_z = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
        )

    def forward(self, x):
        x = x.float()
        x = x.transpose(2, 1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        if self.kl_div_loss:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            return mu, logvar
        z = self.fc_z(x)
        return z


class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


class ResnetPointnet(nn.Module):
    def __init__(self, latent_size=16, kl_div_loss=False, dim=3, hidden_dim=128):
        super().__init__()
        self.latent_size = latent_size
        self.kl_div_loss = kl_div_loss

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, latent_size)

        self.fc_mu = nn.Linear(hidden_dim, latent_size)
        self.fc_logvar = nn.Linear(hidden_dim, latent_size)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        nn.init.normal_(self.fc_c.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.fc_c.bias, 0.0)
        nn.init.normal_(self.fc_mu.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_logvar.bias, 0.0)

    def forward(self, p):
        p = p.float()

        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)
        net = self.pool(net, dim=1)

        if self.kl_div_loss:
            mu = self.fc_mu(self.actvn(net))
            logvar = self.fc_logvar(self.actvn(net))
            return mu, logvar
        z = self.fc_c(self.actvn(net))
        return z
