import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class DecoderGeneratorModelConfig:
    batch_size: int = 32
    sequence_length: int = 200
    channels: int = 3
    kernel_size_1: int = 5
    stride_1: int = 5
    latent_dim: int = 32
    padding: int = 0
    cond_dim: int = 7


@dataclass
class CriticConfig:
    batch_size: int = 32


class ResBlock_1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + res)


class SelfAttention(nn.Module):
    def __init__(self, embd_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embd_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embd_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        attn_output, _ = self.mha(x, x, x)
        x = self.norm(attn_output + x)
        return x.transpose(1, 2)


class DecoderGeneratorModel(nn.Module):
    def __init__(self, config: DecoderGeneratorModelConfig):
        super().__init__()
        self.config = config
        in_features = self.config.latent_dim + self.config.cond_dim

        self.initial_dense = nn.Sequential(
            nn.Linear(in_features, 256 * 10),
            nn.LeakyReLU(0.2)
        )
        self.attention = SelfAttention(256, 4)

        # Upsample 1: Length 10 -> 50 | Channels 256 -> 128
        self.upsample_1 = nn.Sequential(
            nn.Upsample(scale_factor=5, mode='nearest'),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(0.2)
        )

        # Upsample 2: Length 50 -> 100 | Channels 128 -> 64
        self.upsample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.LeakyReLU(0.2)
        )

        # Upsample 3: Length 100 -> 200 | Channels 64 -> 32
        self.upsample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.LeakyReLU(0.2)
        )

        self.channel_reduction = nn.Conv1d(
            in_channels=32,
            out_channels=self.config.channels,
            kernel_size=3,
            padding=1
        )

        self.final_smoothing = nn.Conv1d(
            in_channels=self.config.channels,
            out_channels=self.config.channels,
            kernel_size=5,
            padding='same'
        )

    def forward(self, z, c):
        x_cond = torch.cat((z, c), dim=1)
        x = self.initial_dense(x_cond)
        x = x.view(-1, 256, 10)

        x = self.attention(x)

        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = self.upsample_3(x)

        x = self.channel_reduction(x)
        x = self.final_smoothing(x)
        return x


class CriticModel(nn.Module):
    def __init__(self, config: CriticConfig, channels: int = 3, sequence_length: int = 200, cond_dim=7):
        super().__init__()
        self.config = config
        self.sequence_length = sequence_length
        in_channels = channels + cond_dim

        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_2 = ResBlock_1d(64, 128, stride=2)
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=5, padding=0),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(0.2)
        )
        self.attention = SelfAttention(embd_dim=256, num_heads=4)

        # Sequence math: 200 -> 100 -> 50 -> 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 10, 1)
        )

    def forward(self, x, c):
        c_expanded = c.unsqueeze(-1).expand(-1, -1, self.sequence_length)
        x_cond = torch.cat((x, c_expanded), dim=1)

        x = self.conv_block_1(x_cond)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = self.attention(x)
        x = self.classifier(x)
        return x


class EncoderModel(nn.Module):
    def __init__(self, sequence_length: int = 200, channels: int = 3, latent_dim: int = 32, cond_dim: int = 7):
        super().__init__()
        self.seq_length = sequence_length
        self.channels = channels
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        in_channels = self.channels + self.cond_dim

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(0.2),

            ResBlock_1d(128, 128),

            nn.Conv1d(128, 256, kernel_size=7, stride=5, padding=1),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),

            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 10, self.latent_dim)
        self.fc_logvar = nn.Linear(256 * 10, self.latent_dim)

    def forward(self, x, cond):
        cond_expanded = cond.unsqueeze(-1).expand(-1, -1, self.seq_length)
        x_cond = torch.cat((x, cond_expanded), dim=1)

        x_encoded = self.conv_block(x_cond)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        return mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
