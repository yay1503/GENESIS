import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from model import (
    DecoderGeneratorModel, DecoderGeneratorModelConfig,
    CriticModel, CriticConfig,
    EncoderModel, reparameterize
)
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 200
batch_size = 64
learning_rate = 3e-4
beta1 = 0.5
channels = 3
sequence_length = 200
latent_dim = 32
cond_dim = 7
n_critic = 5
# patience_limit = 30  # Early stopping patience

# Loss Weights
lambda_gp = 10.0
lambda_recon = 10.0
lambda_spectral = 1.0
kl_anneal_epochs = 100
target_lambda_kl = 0.01  # got from the best KL sweep in test.py


def plot_all_losses(history, filename="loss_history_final.png"):
    epochs_range = range(1, len(history['train_comp']) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)

    axes[0, 0].plot(epochs_range, history['train_loss_C'],
                    color='purple', label='Train Loss C')
    axes[0, 0].set_title('Critic Total Loss (Wasserstein + GP)')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_range, history['train_adv'],
                    color='red', label='Train Adv')
    axes[0, 1].set_title('Generator Adversarial Loss')
    axes[0, 1].legend()

    axes[1, 0].plot(epochs_range, history['train_recon'],
                    label='Train Recon (L1)')
    axes[1, 0].plot(epochs_range, history['val_recon'],
                    label='Val Recon (L1)', linestyle='--')
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].legend()

    axes[1, 1].plot(epochs_range, history['train_spectral'],
                    label='Train Spectral')
    axes[1, 1].plot(epochs_range, history['val_spectral'],
                    label='Val Spectral', linestyle='--')
    axes[1, 1].set_title('Spectral Loss')
    axes[1, 1].legend()

    axes[2, 0].plot(epochs_range, history['train_kl'], label='Train KL')
    axes[2, 0].plot(epochs_range, history['val_kl'],
                    label='Val KL', linestyle='--')
    axes[2, 0].set_title('KL Divergence')
    axes[2, 0].legend()

    axes[2, 1].plot(epochs_range, history['train_comp'],
                    label='Train Composite')
    axes[2, 1].plot(epochs_range, history['val_comp'],
                    label='Val Composite', linestyle='--')
    axes[2, 1].set_title('Total Composite Score')
    axes[2, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=300)
    plt.close()


class SpectralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, fake_waveforms, real_waveforms):
        fft_fake = torch.fft.rfft(fake_waveforms, dim=-1)
        fft_real = torch.fft.rfft(real_waveforms, dim=-1)

        mag_fake = torch.sqrt(fft_fake.real**2 + fft_fake.imag**2 + 1e-8)
        mag_real = torch.sqrt(fft_real.real**2 + fft_real.imag**2 + 1e-8)

        loss = self.l1_loss(mag_fake, mag_real)
        return loss


class EarthquakeNumpyDataset(Dataset):
    def __init__(self, metadata_csv, waveform_npy, sequence_length=200, split="train"):
        self.seq_length = sequence_length
        self.split = split
        self.df = pd.read_csv(metadata_csv)

        if 'Epicentral_Distance_km' not in self.df.columns:
            lat1 = np.radians(self.df['Epicenter_Lat'])
            lon1 = np.radians(self.df['Epicenter_Lon'])
            lat2 = np.radians(self.df['Station_Lat'])
            lon2 = np.radians(self.df['Station_Lon'])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            self.df['Epicentral_Distance_km'] = 6371.0 * c

        cond_cols = [
            'Magnitude', 'Epicenter_Lat', 'Epicenter_Lon',
            'EQ_Depth(km)', 'Station_Lat', 'Station_Lon', 'Epicentral_Distance_km'
        ]
        self.conditions = self.df[cond_cols].values.astype(np.float32)

        self.cond_mean = self.conditions.mean(axis=0)
        self.cond_std = self.conditions.std(axis=0) + 1e-8
        self.conditions = (self.conditions - self.cond_mean) / self.cond_std

        raw_waveforms = np.load(waveform_npy).astype(np.float32)
        N = len(self.df)
        raw_waveforms = raw_waveforms[:N]
        raw_waveforms = np.transpose(raw_waveforms, (0, 2, 1))

        medians = np.median(raw_waveforms, axis=2, keepdims=True)
        q75, q25 = np.percentile(raw_waveforms, [75, 25], axis=2)
        iqr = (q75 - q25)[..., np.newaxis] + 1e-8
        self.waveforms = (raw_waveforms - medians) / iqr

        self.waveforms = torch.tensor(self.waveforms, dtype=torch.float32)
        self.conditions = torch.tensor(self.conditions, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx].clone()
        condition = self.conditions[idx]

        # Apply augmentation only during training
        if self.split == "train":
            # 1. Amplitude Scaling (±10%)
            amp_scale = torch.empty(1).uniform_(0.9, 1.1)
            waveform = waveform * amp_scale

            # 2. Random Time-Shifting (roll between -10 and +10 time steps)
            shift = torch.randint(-10, 11, (1,)).item()
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

            # Optional: Add faint Gaussian noise to prevent discriminator overfitting
            noise = torch.randn_like(waveform) * 0.05
            waveform = waveform + noise

        return waveform, condition


def compute_gradient_penalty(critic, real_samples, fake_samples, conditions, device):
    alpha = torch.rand((real_samples.size(0), 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, conditions)
    fake_grad_outputs = torch.ones(d_interpolates.size(), device=device)

    gradients = grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake_grad_outputs, create_graph=True,
        retain_graph=True, only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# Datasets
train_dataset = EarthquakeNumpyDataset(
    metadata_csv="Train_tab_MAG_data_3_SEC.csv",
    waveform_npy="Train_waveform_MAG_data_3C.npy",
    sequence_length=sequence_length,
    split="train"
)
val_dataset = EarthquakeNumpyDataset(
    metadata_csv="Val_tab_MAG_data_3_SEC.csv",
    waveform_npy="Val_waveform_MAG_data_3C.npy",
    sequence_length=sequence_length,
    split="val"
)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

criterion_recon = nn.L1Loss()
criterion_spectral = SpectralLoss().to(device)

# SWEEP CONFIGURATION
# kl_candidates = [0.01, 0.1, 0.5, 1.0]

# Global trackers based on Composite Loss
# overall_best_val_comp = float('inf')
# overall_best_kl = None
# overall_best_epoch = None


print(f"\n{'='*50}")
print(f"STARTING 200-EPOCH TRAINING (KL: {target_lambda_kl})")
print(f"{'='*50}")

encoder = EncoderModel(sequence_length=sequence_length,
                       latent_dim=latent_dim, cond_dim=cond_dim).to(device)
decoder = DecoderGeneratorModel(DecoderGeneratorModelConfig(
    sequence_length=sequence_length, latent_dim=latent_dim, cond_dim=cond_dim)).to(device)
critic = CriticModel(
    CriticConfig(), sequence_length=sequence_length).to(device)

optimizer_E = optim.AdamW(encoder.parameters(),
                          lr=learning_rate, betas=(beta1, 0.999))
optimizer_G = optim.AdamW(decoder.parameters(),
                          lr=learning_rate, betas=(beta1, 0.999))
optimizer_C = optim.AdamW(
    critic.parameters(), lr=learning_rate, betas=(beta1, 0.999))

scheduler_E = optim.lr_scheduler.CosineAnnealingLR(optimizer_E, T_max=epochs)
scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs)
scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=epochs)

best_val_comp = float('inf')
best_epoch = 0

loss_history = {
    'train_loss_C': [], 'train_adv': [], 'train_recon': [],
    'train_kl': [], 'train_spectral': [], 'train_comp': [],
    'val_recon': [], 'val_kl': [], 'val_spectral': [], 'val_comp': []
}

# TRAINING LOOP
for epoch in range(epochs):
    encoder.train()
    decoder.train()
    critic.train()

    current_lambda_kl = target_lambda_kl * min(1.0, epoch / kl_anneal_epochs)

    ep_loss_C = ep_adv = ep_recon = ep_kl = ep_spectral = ep_train_comp = 0.0
    c_steps = g_steps = 0

    for i, (real_waveforms, conditions) in enumerate(train_dataloader):
        real_waveforms, conditions = real_waveforms.to(
            device), conditions.to(device)

        # --- Train Critic ---
        optimizer_C.zero_grad()
        with torch.no_grad():
            mu, logvar = encoder(real_waveforms, conditions)
            z = reparameterize(mu, logvar)
            fake_waveforms = decoder(z, conditions)

        critic_real = critic(real_waveforms, conditions)
        critic_fake = critic(fake_waveforms.detach(), conditions)
        w_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))
        gp = compute_gradient_penalty(
            critic, real_waveforms, fake_waveforms, conditions, device)

        loss_C = w_loss + (lambda_gp * gp)
        loss_C.backward()
        optimizer_C.step()

        ep_loss_C += loss_C.item()
        c_steps += 1

        # --- Train Generator & Encoder ---
        if i % n_critic == 0:
            optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            mu, logvar = encoder(real_waveforms, conditions)
            z = reparameterize(mu, logvar)
            fake_waveforms = decoder(z, conditions)

            fake_waveforms_adv = decoder(z.detach(), conditions)
            critic_fake_G = critic(fake_waveforms_adv, conditions)

            loss_adv = -torch.mean(critic_fake_G)
            loss_recon = criterion_recon(fake_waveforms, real_waveforms)
            loss_kl = -0.5 * \
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss_spectral = criterion_spectral(fake_waveforms, real_waveforms)

            loss_G_total = loss_adv + (lambda_recon * loss_recon) + (
                current_lambda_kl * loss_kl) + (lambda_spectral * loss_spectral)

            loss_G_total.backward()
            optimizer_E.step()
            optimizer_G.step()

            ep_adv += loss_adv.item()
            ep_recon += loss_recon.item()
            ep_kl += loss_kl.item()
            ep_spectral += loss_spectral.item()
            ep_train_comp += loss_G_total.item()
            g_steps += 1

    scheduler_E.step()
    scheduler_G.step()
    scheduler_C.step()

    loss_history['train_loss_C'].append(ep_loss_C / c_steps)
    loss_history['train_adv'].append(ep_adv / g_steps)
    loss_history['train_recon'].append(ep_recon / g_steps)
    loss_history['train_kl'].append(ep_kl / g_steps)
    loss_history['train_spectral'].append(ep_spectral / g_steps)
    loss_history['train_comp'].append(ep_train_comp / g_steps)

    # --- VALIDATION PHASE ---
    encoder.eval()
    decoder.eval()
    critic.eval()

    val_recon_total = val_spectral_total = val_kl_total = val_comp_error = 0.0

    with torch.inference_mode():
        for val_real, val_c in val_dataloader:
            val_real, val_c = val_real.to(device), val_c.to(device)

            mu, logvar = encoder(val_real, val_c)
            z = reparameterize(mu, logvar)
            val_fake = decoder(z, val_c)

            v_recon = criterion_recon(val_fake, val_real)
            v_spectral = criterion_spectral(val_fake, val_real)
            v_kl = -0.5 * torch.sum(1 + logvar -
                                    mu.pow(2) - logvar.exp(), dim=1).mean()

            v_composite = (lambda_recon * v_recon) + \
                (lambda_spectral * v_spectral) + (current_lambda_kl * v_kl)

            val_recon_total += v_recon.item()
            val_spectral_total += v_spectral.item()
            val_kl_total += v_kl.item()
            val_comp_error += v_composite.item()

    num_val_batches = len(val_dataloader)
    loss_history['val_recon'].append(val_recon_total / num_val_batches)
    loss_history['val_spectral'].append(val_spectral_total / num_val_batches)
    loss_history['val_kl'].append(val_kl_total / num_val_batches)

    avg_val_comp = val_comp_error / num_val_batches
    loss_history['val_comp'].append(avg_val_comp)

    # Save Best Checkpoint
    save_status = ""
    if avg_val_comp < best_val_comp:
        best_val_comp = avg_val_comp
        best_epoch = epoch + 1

        torch.save(encoder.state_dict(),
                   f"encoder_best_kl_{target_lambda_kl}.pth")
        torch.save(decoder.state_dict(),
                   f"decoder_best_kl_{target_lambda_kl}.pth")
        torch.save(critic.state_dict(),
                   f"critic_best_kl_{target_lambda_kl}.pth")

        save_status = f"-> NEW BEST COMPOSITE (Epoch {best_epoch})"

    if (epoch + 1) % 5 == 0 or "NEW BEST" in save_status:
        print(f"Epoch [{epoch+1}/{epochs}] | KL: {current_lambda_kl:.4f} | Train Comp: {loss_history['train_comp'][-1]:.4f} | Val Comp: {avg_val_comp:.4f} {save_status}")

# Generate loss plot after completing all 200 epochs
plot_all_losses(loss_history, filename=f"loss_history_final.png")
print(
    f"\nTraining Complete. Best Validation Loss: {best_val_comp:.4f} at Epoch {best_epoch}.")
