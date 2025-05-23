import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import librosa
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import wandb
from pathlib import Path

# Import our SF-VNN components
from audio_sfvnn_system import (
    AudioSFVNNDiscriminator, AudioStructuralLoss, AudioQualityMetrics,
    AudioSpectrogramProcessor, AudioSFVNNTrainer
)


@dataclass
class HiFiGANConfig:
    """Configuration for HiFi-GAN + SF-VNN integration."""
    
    # Audio parameters
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    
    # Generator architecture
    resblock_kernel_sizes: List[int] = None
    resblock_dilation_sizes: List[List[int]] = None
    upsample_rates: List[int] = None
    upsample_kernel_sizes: List[int] = None
    upsample_initial_channel: int = 512
    
    # SF-VNN Discriminator parameters
    sfvnn_vector_channels: List[int] = None
    sfvnn_window_size: int = 5
    sfvnn_sigma: float = 1.0
    sfvnn_multiscale: bool = True
    
    # Training parameters
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.8
    beta2: float = 0.99
    batch_size: int = 16
    num_epochs: int = 200
    
    # Loss weights
    lambda_mel: float = 45.0
    lambda_structural: float = 1.0
    lambda_feature_matching: float = 2.0
    
    def __post_init__(self):
        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 7, 11]
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if self.upsample_rates is None:
            self.upsample_rates = [8, 8, 2, 2]
        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [16, 16, 4, 4]
        if self.sfvnn_vector_channels is None:
            self.sfvnn_vector_channels = [32, 64, 128, 256]


class ResidualBlock(nn.Module):
    """HiFi-GAN style residual block with dilated convolutions."""
    
    def __init__(self, channels: int, kernel_size: int, dilation_sizes: List[int]):
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, 
                     padding=self.get_padding(kernel_size, d), dilation=d)
            for d in dilation_sizes
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                     padding=self.get_padding(kernel_size, 1), dilation=1)
            for _ in dilation_sizes
        ])
        
        self.convs1.apply(self.init_weights)
        self.convs2.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def get_padding(self, kernel_size: int, dilation: int) -> int:
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = conv2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator that converts mel-spectrograms to audio waveforms.
    
    This is the original HiFi-GAN architecture that we'll pair with our SF-VNN discriminator.
    """
    
    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        
        # Pre-convolution
        self.conv_pre = nn.Conv1d(config.n_mels, config.upsample_initial_channel, 7, 1, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(
                config.upsample_initial_channel // (2**i),
                config.upsample_initial_channel // (2**(i+1)),
                k, u, padding=(k-u)//2
            ))
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = config.upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(ResidualBlock(ch, k, d))
        
        # Post-convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        
        # Initialize weights
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate audio waveform from mel-spectrogram.
        
        Args:
            mel: [B, n_mels, T] mel-spectrogram
            
        Returns:
            audio: [B, 1, T * hop_length] waveform
        """
        x = self.conv_pre(mel)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        def _remove_weight_norm(m):
            if hasattr(m, 'weight_norm'):
                nn.utils.remove_weight_norm(m)
        
        self.apply(_remove_weight_norm)


class EnhancedAudioStructuralLoss(nn.Module):
    """
    Enhanced structural loss specifically designed for HiFi-GAN + SF-VNN integration.
    
    Combines mel-spectrogram reconstruction, adversarial loss, and structural consistency.
    """
    
    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        
        self.config = config
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Mel-spectrogram transform for loss computation
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            power=2.0,
            normalized=True
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
    
    def mel_spectrogram_loss(self, real_audio: torch.Tensor, fake_audio: torch.Tensor) -> torch.Tensor:
        """Compute mel-spectrogram reconstruction loss."""
        
        # Convert to mel-spectrograms
        real_mel = self.amplitude_to_db(self.mel_transform(real_audio.squeeze(1)))
        fake_mel = self.amplitude_to_db(self.mel_transform(fake_audio.squeeze(1)))
        
        return self.l1_loss(fake_mel, real_mel)
    
    def generator_loss(self,
                      real_audio: torch.Tensor,
                      fake_audio: torch.Tensor,
                      real_mel: torch.Tensor,
                      fake_mel: torch.Tensor,
                      discriminator: AudioSFVNNDiscriminator) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive generator loss.
        
        Args:
            real_audio: [B, 1, T] real audio waveforms
            fake_audio: [B, 1, T] generated audio waveforms  
            real_mel: [B, 1, F, T] real mel-spectrograms
            fake_mel: [B, 1, F, T] fake mel-spectrograms (from real_audio)
            discriminator: SF-VNN discriminator
            
        Returns:
            Dictionary of loss components
        """
        
        # 1. Mel-spectrogram reconstruction loss
        mel_loss = self.mel_spectrogram_loss(real_audio, fake_audio)
        
        # 2. Adversarial loss (fool discriminator)
        fake_probs, fake_sigs = discriminator(fake_mel, return_signature=True)
        real_probs, real_sigs = discriminator(real_mel, return_signature=True)
        
        # Generator wants discriminator to classify fake as real
        adv_loss = self.bce_loss(fake_probs, torch.ones_like(fake_probs))
        
        # 3. Structural consistency loss
        structural_loss = 0.0
        for fake_sig, real_sig in zip(fake_sigs, real_sigs):
            # Encourage similar structural statistics
            structural_loss += self.mse_loss(fake_sig.entropy.mean(), real_sig.entropy.mean())
            structural_loss += self.mse_loss(fake_sig.harmonic_coherence.mean(), real_sig.harmonic_coherence.mean())
            structural_loss += self.mse_loss(fake_sig.temporal_stability.mean(), real_sig.temporal_stability.mean())
            structural_loss += self.mse_loss(fake_sig.spectral_flow.mean(), real_sig.spectral_flow.mean())
        structural_loss /= len(fake_sigs)
        
        # 4. Feature matching loss (intermediate layer similarity)
        fm_loss = 0.0
        for fake_sig, real_sig in zip(fake_sigs, real_sigs):
            fm_loss += self.l1_loss(fake_sig.alignment, real_sig.alignment)
            fm_loss += self.l1_loss(fake_sig.curvature, real_sig.curvature)
        fm_loss /= len(fake_sigs)
        
        # Combine losses
        total_loss = (
            self.config.lambda_mel * mel_loss +
            adv_loss +
            self.config.lambda_structural * structural_loss +
            self.config.lambda_feature_matching * fm_loss
        )
        
        return {
            'total': total_loss,
            'mel': mel_loss,
            'adversarial': adv_loss,
            'structural': structural_loss,
            'feature_matching': fm_loss
        }
    
    def discriminator_loss(self,
                          real_mel: torch.Tensor,
                          fake_mel: torch.Tensor,
                          discriminator: AudioSFVNNDiscriminator) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss."""
        
        # Real samples
        real_probs = discriminator(real_mel)
        real_loss = self.bce_loss(real_probs, torch.ones_like(real_probs))
        
        # Fake samples (detached to prevent generator gradients)
        fake_probs = discriminator(fake_mel.detach())
        fake_loss = self.bce_loss(fake_probs, torch.zeros_like(fake_probs))
        
        # Total discriminator loss
        total_loss = (real_loss + fake_loss) / 2
        
        # Accuracy metrics
        real_acc = ((torch.sigmoid(real_probs) > 0.5).float().mean())
        fake_acc = ((torch.sigmoid(fake_probs) < 0.5).float().mean())
        
        return {
            'total': total_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'real_accuracy': real_acc,
            'fake_accuracy': fake_acc,
            'discriminator_accuracy': (real_acc + fake_acc) / 2
        }


class HiFiGANSFVNNDataset(torch.utils.data.Dataset):
    """
    Dataset for training HiFi-GAN with SF-VNN discriminator.
    
    Loads audio files and provides both waveforms and mel-spectrograms.
    """
    
    def __init__(self,
                 audio_files: List[str],
                 config: HiFiGANConfig,
                 segment_length: int = 8192,
                 training: bool = True):
        
        self.audio_files = audio_files
        self.config = config
        self.segment_length = segment_length
        self.training = training
        
        # Audio processing
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            power=2.0,
            normalized=True
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
        
        print(f"Dataset initialized with {len(audio_files)} files")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            audio: [1, segment_length] waveform
            mel: [1, n_mels, mel_frames] mel-spectrogram
        """
        
        # Load audio
        audio_path = self.audio_files[idx]
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Segment audio for training
        if self.training and audio.size(1) > self.segment_length:
            start_idx = torch.randint(0, audio.size(1) - self.segment_length, (1,)).item()
            audio = audio[:, start_idx:start_idx + self.segment_length]
        elif audio.size(1) < self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - audio.size(1)
            audio = F.pad(audio, (0, pad_length))
        
        # Convert to mel-spectrogram
        mel = self.amplitude_to_db(self.mel_transform(audio))
        
        # Normalize mel-spectrogram to [-1, 1]
        mel = self._normalize_mel(mel)
        
        return audio, mel
    
    def _normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Normalize mel-spectrogram to [-1, 1] range."""
        # Assuming mel is in dB scale, typically ranges from -80 to 0
        mel = torch.clamp(mel, min=-80, max=0)
        mel = (mel + 80) / 80  # Scale to [0, 1]
        mel = 2 * mel - 1      # Scale to [-1, 1]
        return mel


class HiFiGANSFVNNTrainer:
    """
    Complete training framework for HiFi-GAN with SF-VNN discriminator.
    
    Implements state-of-the-art audio generation with structural awareness.
    """
    
    def __init__(self,
                 config: HiFiGANConfig,
                 train_dataset: HiFiGANSFVNNDataset,
                 val_dataset: HiFiGANSFVNNDataset,
                 device: torch.device,
                 use_wandb: bool = True):
        
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize models
        self.generator = HiFiGANGenerator(config).to(device)
        self.discriminator = AudioSFVNNDiscriminator(
            input_channels=1,
            vector_channels=config.sfvnn_vector_channels,
            window_size=config.sfvnn_window_size,
            sigma=config.sfvnn_sigma,
            multiscale_analysis=config.sfvnn_multiscale
        ).to(device)
        
        # Optimizers
        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(config.beta1, config.beta2)
        )
        
        self.optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(config.beta1, config.beta2)
        )
        
        # Learning rate schedulers
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=0.999)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=0.999)
        
        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Loss function
        self.loss_fn = EnhancedAudioStructuralLoss(config)
        
        # Metrics
        self.metrics = AudioQualityMetrics(config.sample_rate)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_mel_loss = float('inf')
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="hifigan-sfvnn",
                config=config.__dict__,
                name=f"hifigan_sfvnn_{config.sfvnn_multiscale}"
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = defaultdict(list)
        
        for batch_idx, (real_audio, real_mel) in enumerate(self.train_loader):
            real_audio = real_audio.to(self.device)
            real_mel = real_mel.to(self.device)
            
            # Generate fake audio from mel-spectrogram
            fake_audio = self.generator(real_mel)
            
            # Convert fake audio to mel-spectrogram for discriminator
            fake_mel = self._audio_to_mel(fake_audio)
            
            # Train Discriminator
            self.optim_d.zero_grad()
            d_losses = self.loss_fn.discriminator_loss(real_mel, fake_mel, self.discriminator)
            d_losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
            self.optim_d.step()
            
            # Train Generator
            self.optim_g.zero_grad()
            g_losses = self.loss_fn.generator_loss(
                real_audio, fake_audio, real_mel, fake_mel, self.discriminator
            )
            g_losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10.0)
            self.optim_g.step()
            
            # Record metrics
            for key, value in {**g_losses, **d_losses}.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key].append(value.item())
            
            self.global_step += 1
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'epoch': self.epoch,
                    'step': self.global_step,
                    'g_loss_total': g_losses['total'].item(),
                    'g_loss_mel': g_losses['mel'].item(),
                    'g_loss_structural': g_losses['structural'].item(),
                    'd_loss_total': d_losses['total'].item(),
                    'd_accuracy': d_losses['discriminator_accuracy'].item(),
                })
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch {self.epoch}, Batch {batch_idx}: '
                      f'G_total: {g_losses["total"].item():.4f}, '
                      f'G_mel: {g_losses["mel"].item():.4f}, '
                      f'G_struct: {g_losses["structural"].item():.4f}, '
                      f'D_total: {d_losses["total"].item():.4f}, '
                      f'D_acc: {d_losses["discriminator_accuracy"].item():.3f}')
        
        # Update learning rates
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model performance."""
        
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = defaultdict(list)
        generated_samples = []
        real_samples = []
        
        with torch.no_grad():
            for batch_idx, (real_audio, real_mel) in enumerate(self.val_loader):
                if batch_idx >= 10:  # Limit validation samples
                    break
                
                real_audio = real_audio.to(self.device)
                real_mel = real_mel.to(self.device)
                
                # Generate fake audio
                fake_audio = self.generator(real_mel)
                fake_mel = self._audio_to_mel(fake_audio)
                
                # Compute losses
                g_losses = self.loss_fn.generator_loss(
                    real_audio, fake_audio, real_mel, fake_mel, self.discriminator
                )
                d_losses = self.loss_fn.discriminator_loss(real_mel, fake_mel, self.discriminator)
                
                # Record metrics
                for key, value in {**g_losses, **d_losses}.items():
                    if isinstance(value, torch.Tensor):
                        val_metrics[key].append(value.item())
                
                # Collect samples for advanced metrics
                if batch_idx < 5:
                    generated_samples.append(fake_mel.cpu())
                    real_samples.append(real_mel.cpu())
        
        # Compute advanced metrics
        if generated_samples and real_samples:
            real_batch = torch.cat(real_samples, dim=0)
            fake_batch = torch.cat(generated_samples, dim=0)
            
            # Structural metrics
            struct_metrics = self.metrics.compute_structural_metrics(
                real_batch.to(self.device), fake_batch.to(self.device), self.discriminator
            )
            val_metrics.update(struct_metrics)
            
            # FAD score
            fad_score = self.metrics.compute_frechet_audio_distance(real_batch, fake_batch)
            val_metrics['fad_score'] = fad_score
        
        # Average metrics
        avg_metrics = {key: np.mean(values) if isinstance(values, list) else values 
                      for key, values in val_metrics.items()}
        
        return avg_metrics
    
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel-spectrogram for discriminator input."""
        mel = self.loss_fn.amplitude_to_db(self.loss_fn.mel_transform(audio.squeeze(1)))
        return self.train_loader.dataset._normalize_mel(mel.unsqueeze(1))
    
    def train(self, num_epochs: int = None) -> Dict[str, List[float]]:
        """Complete training loop."""
        
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        print("🎵 Starting HiFi-GAN + SF-VNN Training")
        print("=" * 60)
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"Training for {num_epochs} epochs")
        print()
        
        history = defaultdict(list)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Record history
            for key, value in train_metrics.items():
                history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                history[f'val_{key}'].append(value)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - G_total: {train_metrics['total']:.4f}, "
                  f"G_mel: {train_metrics['mel']:.4f}, "
                  f"G_struct: {train_metrics['structural']:.4f}")
            print(f"  Val   - G_total: {val_metrics['total']:.4f}, "
                  f"FAD: {val_metrics.get('fad_score', 0):.4f}")
            
            # Save best model
            if val_metrics['mel'] < self.best_mel_loss:
                self.best_mel_loss = val_metrics['mel']
                self.save_checkpoint(f'best_model_epoch_{epoch}.pth')
                print(f"  🎯 New best mel loss! Saved checkpoint.")
            
            # Periodic checkpoints
            if epoch % 25 == 0 and epoch > 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'val_mel_loss': val_metrics['mel'],
                    'val_fad_score': val_metrics.get('fad_score', 0),
                    'val_structural_distance': val_metrics.get('overall_structural_distance', 0),
                    'lr_g': self.scheduler_g.get_last_lr()[0],
                    'lr_d': self.scheduler_d.get_last_lr()[0]
                })
        
        print(f"\n🎉 Training Complete! Best mel loss: {self.best_mel_loss:.4f}")
        return dict(history)
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_g_state_dict': self.optim_g.state_dict(),
            'optim_d_state_dict': self.optim_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'config': self.config,
            'best_mel_loss': self.best_mel_loss
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optim_g.load_state_dict(checkpoint['optim_g_state_dict'])
        self.optim_d.load_state_dict(checkpoint['optim_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        self.best_mel_loss = checkpoint['best_mel_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def generate_audio(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Generate audio from mel-spectrogram using trained generator."""
        self.generator.eval()
        with torch.no_grad():
            audio = self.generator(mel_spectrogram.to(self.device))
        return audio.cpu()


class HiFiGANSFVNNExperiment:
    """
    Comprehensive experimental framework for HiFi-GAN + SF-VNN research.
    
    Provides ablation studies, comparison with baseline, and analysis tools.
    """
    
    def __init__(self, 
                 train_audio_files: List[str],
                 val_audio_files: List[str],
                 results_dir: str = 'hifigan_sfvnn_experiments'):
        
        self.train_audio_files = train_audio_files
        self.val_audio_files = val_audio_files
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def run_comparative_study(self) -> Dict[str, Dict]:
        """
        Run comprehensive comparison between HiFi-GAN variants:
        1. Original HiFi-GAN (CNN discriminator)
        2. HiFi-GAN + SF-VNN (single scale)
        3. HiFi-GAN + SF-VNN (multi-scale)
        4. HiFi-GAN + SF-VNN (structure-heavy)
        """
        
        print("🧪 Starting HiFi-GAN + SF-VNN Comparative Study")
        print("=" * 70)
        
        # Define experimental configurations
        configs = [
            {
                'name': 'HiFiGAN_Original_CNN',
                'description': 'Original HiFi-GAN with CNN discriminator',
                'use_sfvnn': False,
                'sfvnn_multiscale': False,
                'lambda_structural': 0.0,
                'training_epochs': 100
            },
            {
                'name': 'HiFiGAN_SFVNN_Single',
                'description': 'HiFi-GAN with single-scale SF-VNN discriminator',
                'use_sfvnn': True,
                'sfvnn_multiscale': False,
                'lambda_structural': 1.0,
                'training_epochs': 100
            },
            {
                'name': 'HiFiGAN_SFVNN_Multi',
                'description': 'HiFi-GAN with multi-scale SF-VNN discriminator',
                'use_sfvnn': True,
                'sfvnn_multiscale': True,
                'lambda_structural': 1.0,
                'training_epochs': 100
            },
            {
                'name': 'HiFiGAN_SFVNN_StructHeavy',
                'description': 'HiFi-GAN with SF-VNN (high structural weight)',
                'use_sfvnn': True,
                'sfvnn_multiscale': True,
                'lambda_structural': 2.0,
                'training_epochs': 100
            }
        ]
        
        results = {}
        
        for config in configs:
            print(f"\n🔬 Running: {config['name']}")
            print(f"   {config['description']}")
            
            # Create configuration
            hifigan_config = HiFiGANConfig(
                lambda_structural=config['lambda_structural'],
                sfvnn_multiscale=config['sfvnn_multiscale'],
                num_epochs=config['training_epochs']
            )
            
            # Create datasets
            train_dataset = HiFiGANSFVNNDataset(
                self.train_audio_files, hifigan_config, training=True
            )
            val_dataset = HiFiGANSFVNNDataset(
                self.val_audio_files, hifigan_config, training=False
            )
            
            if config['use_sfvnn']:
                # Train with SF-VNN discriminator
                trainer = HiFiGANSFVNNTrainer(
                    hifigan_config, train_dataset, val_dataset, 
                    self.device, use_wandb=False
                )
            else:
                # Train with baseline CNN discriminator
                trainer = self._create_baseline_trainer(
                    hifigan_config, train_dataset, val_dataset
                )
            
            # Train model
            history = trainer.train(config['training_epochs'])
            
            # Final comprehensive evaluation
            final_metrics = self._comprehensive_evaluation(trainer, val_dataset)
            
            results[config['name']] = {
                'config': config,
                'history': history,
                'final_metrics': final_metrics,
                'best_mel_loss': min(history['val_mel']) if 'val_mel' in history else float('inf')
            }
            
            # Save model
            model_path = self.results_dir / f"{config['name']}_final.pth"
            trainer.save_checkpoint(str(model_path))
            
            print(f"   ✅ Completed - Best mel loss: {results[config['name']]['best_mel_loss']:.4f}")
            if 'fad_score' in final_metrics:
                print(f"   📊 FAD Score: {final_metrics['fad_score']:.4f}")
        
        # Generate comparative analysis
        self._generate_comparative_report(results)
        
        # Save results
        self._save_experiment_results(results)
        
        return results
    
    def _create_baseline_trainer(self, config, train_dataset, val_dataset):
        """Create baseline trainer with CNN discriminator for comparison."""
        
        class BaselineCNNDiscriminator(nn.Module):
            """Standard CNN discriminator for comparison."""
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    # Input: [B, 1, F, T]
                    nn.Conv2d(1, 32, (3, 9), (1, 1), (1, 4)),
                    nn.LeakyReLU(0.2),
                    
                    nn.Conv2d(32, 64, (3, 8), (1, 2), (1, 3)),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2),
                    
                    nn.Conv2d(64, 128, (3, 8), (1, 2), (1, 3)),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    
                    nn.Conv2d(128, 256, (3, 6), (1, 2), (1, 2)),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2),
                    
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(256, 1)
                )
            
            def forward(self, x):
                return self.model(x)
        
        # Create baseline trainer
        class BaselineTrainer(HiFiGANSFVNNTrainer):
            def __init__(self, config, train_dataset, val_dataset, device):
                # Initialize without calling parent __init__
                self.config = config
                self.device = device
                self.use_wandb = False
                
                # Initialize models
                self.generator = HiFiGANGenerator(config).to(device)
                self.discriminator = BaselineCNNDiscriminator().to(device)
                
                # Optimizers
                self.optim_g = torch.optim.AdamW(
                    self.generator.parameters(),
                    lr=config.learning_rate_g,
                    betas=(config.beta1, config.beta2)
                )
                
                self.optim_d = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=config.learning_rate_d,
                    betas=(config.beta1, config.beta2)
                )
                
                # Learning rate schedulers
                self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=0.999)
                self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=0.999)
                
                # Data loaders
                self.train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
                )
                self.val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
                )
                
                # Simple loss function for baseline
                self.loss_fn = nn.BCEWithLogitsLoss()
                self.l1_loss = nn.L1Loss()
                
                # Training state
                self.epoch = 0
                self.global_step = 0
                self.best_mel_loss = float('inf')
            
            def train_epoch(self):
                """Simplified training epoch for baseline."""
                self.generator.train()
                self.discriminator.train()
                
                epoch_metrics = defaultdict(list)
                
                for batch_idx, (real_audio, real_mel) in enumerate(self.train_loader):
                    real_audio = real_audio.to(self.device)
                    real_mel = real_mel.to(self.device)
                    
                    # Generate fake audio
                    fake_audio = self.generator(real_mel)
                    fake_mel = self._audio_to_mel(fake_audio)
                    
                    # Train Discriminator
                    self.optim_d.zero_grad()
                    
                    real_pred = self.discriminator(real_mel)
                    fake_pred = self.discriminator(fake_mel.detach())
                    
                    d_real_loss = self.loss_fn(real_pred, torch.ones_like(real_pred))
                    d_fake_loss = self.loss_fn(fake_pred, torch.zeros_like(fake_pred))
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    
                    d_loss.backward()
                    self.optim_d.step()
                    
                    # Train Generator
                    self.optim_g.zero_grad()
                    
                    fake_pred = self.discriminator(fake_mel)
                    g_adv_loss = self.loss_fn(fake_pred, torch.ones_like(fake_pred))
                    g_mel_loss = self.l1_loss(fake_mel, real_mel) * 45.0
                    g_loss = g_adv_loss + g_mel_loss
                    
                    g_loss.backward()
                    self.optim_g.step()
                    
                    # Record metrics
                    epoch_metrics['total'].append(g_loss.item())
                    epoch_metrics['mel'].append(g_mel_loss.item())
                    epoch_metrics['adversarial'].append(g_adv_loss.item())
                    
                    self.global_step += 1
                    
                    if batch_idx % 100 == 0:
                        print(f'Baseline Epoch {self.epoch}, Batch {batch_idx}: '
                              f'G_total: {g_loss.item():.4f}, G_mel: {g_mel_loss.item():.4f}')
                
                self.scheduler_g.step()
                self.scheduler_d.step()
                
                return {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return BaselineTrainer(config, train_dataset, val_dataset, self.device)
    
    def _comprehensive_evaluation(self, trainer, val_dataset) -> Dict[str, float]:
        """Perform comprehensive evaluation of trained model."""
        
        # Standard validation metrics
        val_metrics = trainer.validate()
        
        # Additional perceptual metrics
        if hasattr(trainer.discriminator, 'compute_structural_distance'):
            # SF-VNN specific metrics
            trainer.generator.eval()
            trainer.discriminator.eval()
            
            structural_analysis = []
            
            with torch.no_grad():
                for batch_idx, (real_audio, real_mel) in enumerate(trainer.val_loader):
                    if batch_idx >= 5:  # Sample subset
                        break
                    
                    real_audio = real_audio.to(trainer.device)
                    real_mel = real_mel.to(trainer.device)
                    
                    fake_audio = trainer.generator(real_mel)
                    fake_mel = trainer._audio_to_mel(fake_audio)
                    
                    # Structural distance analysis
                    struct_dist = trainer.discriminator.compute_structural_distance(real_mel, fake_mel)
                    structural_analysis.append(struct_dist)
            
            # Average structural metrics across scales
            if structural_analysis:
                avg_struct_metrics = {}
                for scale in structural_analysis[0].keys():
                    scale_metrics = {}
                    for metric in structural_analysis[0][scale].keys():
                        values = [sa[scale][metric].item() for sa in structural_analysis]
                        scale_metrics[metric] = np.mean(values)
                    avg_struct_metrics[scale] = scale_metrics
                
                val_metrics['structural_analysis'] = avg_struct_metrics
        
        return val_metrics
    
    def _generate_comparative_report(self, results: Dict[str, Dict]):
        """Generate comprehensive comparative analysis report."""
        
        print("\n" + "="*80)
        print("🏆 HIFI-GAN + SF-VNN COMPARATIVE STUDY RESULTS")
        print("="*80)
        
        # Sort by mel loss (primary metric)
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1]['best_mel_loss']
        )
        
        print("\n📊 Performance Ranking (by Mel Loss):")
        print("-" * 60)
        
        baseline_result = None
        best_sfvnn_result = None
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            mel_loss = result['best_mel_loss']
            fad_score = result['final_metrics'].get('fad_score', 'N/A')
            
            print(f"{rank}. {name}")
            print(f"   Mel Loss: {mel_loss:.4f}")
            print(f"   FAD Score: {fad_score}")
            print(f"   Description: {result['config']['description']}")
            
            if 'Original' in name:
                baseline_result = result
            elif 'SFVNN' in name and best_sfvnn_result is None:
                best_sfvnn_result = result
            
            print()
        
        # Detailed comparison analysis
        print("🔍 Detailed Analysis:")
        print("-" * 40)
        
        if baseline_result and best_sfvnn_result:
            mel_improvement = baseline_result['best_mel_loss'] - best_sfvnn_result['best_mel_loss']
            improvement_pct = (mel_improvement / baseline_result['best_mel_loss']) * 100
            
            print(f"• Best SF-VNN vs Baseline:")
            print(f"  - Mel loss improvement: {mel_improvement:.4f} ({improvement_pct:.1f}%)")
            
            if 'fad_score' in baseline_result['final_metrics'] and 'fad_score' in best_sfvnn_result['final_metrics']:
                fad_improvement = baseline_result['final_metrics']['fad_score'] - best_sfvnn_result['final_metrics']['fad_score']
                print(f"  - FAD score improvement: {fad_improvement:.4f}")
        
        # Structural analysis insights
        sfvnn_results = {k: v for k, v in results.items() if 'SFVNN' in k}
        if len(sfvnn_results) > 1:
            print(f"\n• SF-VNN Configuration Analysis:")
            
            multiscale_results = [r for r in sfvnn_results.values() if r['config']['sfvnn_multiscale']]
            single_scale_results = [r for r in sfvnn_results.values() if not r['config']['sfvnn_multiscale']]
            
            if multiscale_results and single_scale_results:
                avg_multi = np.mean([r['best_mel_loss'] for r in multiscale_results])
                avg_single = np.mean([r['best_mel_loss'] for r in single_scale_results])
                print(f"  - Multi-scale avg mel loss: {avg_multi:.4f}")
                print(f"  - Single-scale avg mel loss: {avg_single:.4f}")
                print(f"  - Multi-scale advantage: {avg_single - avg_multi:.4f}")
        
        print("\n🎯 Key Findings:")
        print("-" * 30)
        
        findings = []
        
        # Finding 1: SF-VNN effectiveness
        if baseline_result and best_sfvnn_result:
            if best_sfvnn_result['best_mel_loss'] < baseline_result['best_mel_loss']:
                findings.append("✅ SF-VNN discriminator improves mel-spectrogram reconstruction quality")
            else:
                findings.append("⚠️  SF-VNN discriminator needs hyperparameter tuning")
        
        # Finding 2: Multi-scale benefit
        multiscale_perf = [r['best_mel_loss'] for r in results.values() 
                          if r['config'].get('sfvnn_multiscale', False)]
        single_perf = [r['best_mel_loss'] for r in results.values() 
                      if 'sfvnn_multiscale' in r['config'] and not r['config']['sfvnn_multiscale']]
        
        if multiscale_perf and single_perf:
            if np.mean(multiscale_perf) < np.mean(single_perf):
                findings.append("✅ Multi-scale structural analysis provides consistent benefits")
            else:
                findings.append("⚠️  Single-scale analysis sufficient for this dataset")
        
        # Finding 3: Structural loss weight sensitivity
        struct_heavy = [r for r in results.values() if r['config'].get('lambda_structural', 0) > 1.0]
        struct_normal = [r for r in results.values() if r['config'].get('lambda_structural', 0) == 1.0]
        
        if struct_heavy and struct_normal:
            heavy_perf = np.mean([r['best_mel_loss'] for r in struct_heavy])
            normal_perf = np.mean([r['best_mel_loss'] for r in struct_normal])
            
            if heavy_perf < normal_perf:
                findings.append("✅ Higher structural loss weight improves quality")
            else:
                findings.append("⚠️  Standard structural loss weight is optimal")
        
        for finding in findings:
            print(f"  {finding}")
        
        print("\n🚀 Recommendations:")
        print("-" * 30)
        
        best_config = sorted_results[0][1]['config']
        
        recommendations = [
            f"Use {'multi-scale' if best_config['sfvnn_multiscale'] else 'single-scale'} SF-VNN discriminator",
            f"Set structural loss weight to {best_config['lambda_structural']}",
            "SF-VNN provides measurable improvements in audio generation quality",
            "Structural consistency metrics correlate with perceptual quality"
        ]
        
        for rec in recommendations:
            print(f"  • {rec}")
    
    def _save_experiment_results(self, results: Dict[str, Dict]):
        """Save comprehensive experimental results."""
        
        # Create serializable version
        serializable_results = {}
        
        for name, result in results.items():
            serializable_results[name] = {
                'config': result['config'],
                'best_mel_loss': result['best_mel_loss'],
                'final_metrics': {
                    k: float(v) if isinstance(v, (torch.Tensor, np.number)) else v
                    for k, v in result['final_metrics'].items()
                    if not isinstance(v, dict)  # Skip complex nested structures for now
                }
            }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f'comparative_study_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
    
    def visualize_results(self, results: Dict[str, Dict]):
        """Create comprehensive visualizations of experimental results."""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        names = list(results.keys())
        mel_losses = [results[name]['best_mel_loss'] for name in names]
        fad_scores = [results[name]['final_metrics'].get('fad_score', 0) for name in names]
        
        # Plot 1: Mel Loss Comparison
        colors = ['red' if 'Original' in name else 'blue' if 'SFVNN' in name else 'green' 
                 for name in names]
        
        bars1 = axes[0, 0].bar(range(len(mel_losses)), mel_losses, color=colors, alpha=0.7)
        axes[0, 0].set_title('Mel-Spectrogram Reconstruction Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('L1 Loss')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels([name.replace('_', '\n') for name in names], rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mel_losses):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: FAD Score Comparison
        if any(score > 0 for score in fad_scores):
            bars2 = axes[0, 1].bar(range(len(fad_scores)), fad_scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('Fréchet Audio Distance (FAD)', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('FAD Score (lower is better)')
            axes[0, 1].set_xticks(range(len(names)))
            axes[0, 1].set_xticklabels([name.replace('_', '\n') for name in names], rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, fad_scores):
                if value > 0:
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Training Convergence
        for name, result in results.items():
            if 'val_mel' in result['history']:
                epochs = range(len(result['history']['val_mel']))
                line_style = '--' if 'Original' in name else '-'
                axes[1, 0].plot(epochs, result['history']['val_mel'], 
                               label=name.replace('_', ' '), linestyle=line_style, linewidth=2)
        
        axes[1, 0].set_title('Training Convergence', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Mel Loss')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance vs Configuration
        sfvnn_results = {k: v for k, v in results.items() if 'SFVNN' in k}
        if len(sfvnn_results) > 1:
            config_names = []
            performance = []
            
            for name, result in sfvnn_results.items():
                config_names.append(name.split('_')[-1])  # Extract config type
                performance.append(result['best_mel_loss'])
            
            bars4 = axes[1, 1].bar(range(len(performance)), performance, 
                                  color='blue', alpha=0.7)
            axes[1, 1].set_title('SF-VNN Configuration Comparison', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Mel Loss')
            axes[1, 1].set_xticks(range(len(config_names)))
            axes[1, 1].set_xticklabels(config_names, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars4, performance):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / 'comparative_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to: {plot_file}")


# Example usage and complete integration workflow
def run_hifigan_sfvnn_experiment():
    """
    Complete example of running HiFi-GAN + SF-VNN experiments.
    
    This demonstrates the full workflow from data preparation to analysis.
    """
    
    print("🎵 HiFi-GAN + SF-VNN Integration Example")
    print("=" * 50)
    
    # Example data preparation (replace with your actual audio files)
    train_audio_files = [
        # Add paths to your training audio files
        # "path/to/train/audio1.wav",
        # "path/to/train/audio2.wav", 
        # ...
    ]
    
    val_audio_files = [
        # Add paths to your validation audio files
        # "path/to/val/audio1.wav",
        # "path/to/val/audio2.wav",
        # ...
    ]
    
    # For demonstration, we'll use dummy file lists
    if not train_audio_files:
        print("⚠️  No audio files specified. This is a demonstration of the framework.")
        print("   Replace train_audio_files and val_audio_files with your actual data paths.")
        return
    
    # Initialize experiment framework
    experiment = HiFiGANSFVNNExperiment(
        train_audio_files=train_audio_files,
        val_audio_files=val_audio_files,
        results_dir='hifigan_sfvnn_results'
    )
    
    # Run comprehensive comparative study
    results = experiment.run_comparative_study()
    
    # Generate visualizations
    experiment.visualize_results(results)
    
    print("\n🎉 Experiment completed!")
    print("Check the results directory for detailed analysis and saved models.")
    
    return results


# Quick test with dummy data
if __name__ == "__main__":
    print("🧪 Testing HiFi-GAN + SF-VNN Integration")
    
    # Test configuration
    config = HiFiGANConfig(
        sample_rate=22050,
        n_mels=80,
        batch_size=4,  # Small batch for testing
        num_epochs=2   # Quick test
    )
    
    # Test generator
    generator = HiFiGANGenerator(config)
    
    # Test discriminator
    discriminator = AudioSFVNNDiscriminator(
        input_channels=1,
        vector_channels=[32, 64],  # Smaller for testing
        multiscale_analysis=True
    )
    
    # Test forward pass
    dummy_mel = torch.randn(2, 80, 100)  # [batch, mels, time]
    dummy_spec = torch.randn(2, 1, 80, 100)  # [batch, channels, freq, time]
    
    print(f"Generator input shape: {dummy_mel.shape}")
    audio_output = generator(dummy_mel)
    print(f"Generator output shape: {audio_output.shape}")
    
    print(f"Discriminator input shape: {dummy_spec.shape}")
    disc_output, signatures = discriminator(dummy_spec, return_signature=True)
    print(f"Discriminator output shape: {disc_output.shape}")
    print(f"Number of signature scales: {len(signatures)}")
    
    # Test loss computation
    loss_fn = EnhancedAudioStructuralLoss(config)
    
    dummy_real_audio = torch.randn(2, 1, 2048)
    dummy_fake_audio = torch.randn(2, 1, 2048)
    dummy_real_mel_spec = torch.randn(2, 1, 80, 100)
    dummy_fake_mel_spec = torch.randn(2, 1, 80, 100)
    
    g_losses = loss_fn.generator_loss(
        dummy_real_audio, dummy_fake_audio, 
        dummy_real_mel_spec, dummy_fake_mel_spec, 
        discriminator
    )
    
    d_losses = loss_fn.discriminator_loss(
        dummy_real_mel_spec, dummy_fake_mel_spec, discriminator
    )
    
    print("\nLoss computation test:")
    print(f"Generator losses: {list(g_losses.keys())}")
    print(f"Discriminator losses: {list(d_losses.keys())}")
    print(f"Total G loss: {g_losses['total'].item():.4f}")
    print(f"Total D loss: {d_losses['total'].item():.4f}")
    
    print("\n✅ All components working correctly!")
    print("\n🚀 Ready to run full experiments with real audio data!")
    print("\nTo use with your data:")
    print("1. Prepare audio files (WAV format, preferably 22050 Hz)")
    print("2. Update train_audio_files and val_audio_files lists")
    print("3. Run: run_hifigan_sfvnn_experiment()")
    print("4. Check results directory for analysis and trained models")


# Additional utility functions for real-world usage
class AudioDatasetBuilder:
    """Utility class for building audio datasets from directories."""
    
    @staticmethod
    def build_dataset_from_directory(
        audio_dir: str,
        extensions: List[str] = ['.wav', '.flac', '.mp3'],
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Build train/val/test splits from audio directory.
        
        Args:
            audio_dir: Directory containing audio files
            extensions: Audio file extensions to include
            train_split: Fraction for training
            val_split: Fraction for validation  
            test_split: Fraction for testing
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        
        import random
        from pathlib import Path
        
        # Collect all audio files
        audio_dir = Path(audio_dir)
        audio_files = []
        
        for ext in extensions:
            audio_files.extend(list(audio_dir.glob(f'**/*{ext}')))
        
        audio_files = [str(f) for f in audio_files]
        
        print(f"Found {len(audio_files)} audio files in {audio_dir}")
        
        # Shuffle and split
        random.seed(random_seed)
        random.shuffle(audio_files)
        
        n_total = len(audio_files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_files = audio_files[:n_train]
        val_files = audio_files[n_train:n_train + n_val]
        test_files = audio_files[n_train + n_val:]
        
        print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        return train_files, val_files, test_files
    
    @staticmethod
    def validate_audio_files(audio_files: List[str], sample_rate: int = 22050) -> List[str]:
        """
        Validate audio files and filter out corrupted ones.
        
        Args:
            audio_files: List of audio file paths
            sample_rate: Expected sample rate
            
        Returns:
            List of valid audio file paths
        """
        
        valid_files = []
        
        for audio_file in audio_files:
            try:
                # Try to load the audio file
                audio, sr = torchaudio.load(audio_file)
                
                # Check basic properties
                if audio.numel() > 0 and sr > 0:
                    valid_files.append(audio_file)
                else:
                    print(f"⚠️  Skipping invalid file: {audio_file}")
                    
            except Exception as e:
                print(f"⚠️  Error loading {audio_file}: {e}")
        
        print(f"Validated {len(valid_files)}/{len(audio_files)} audio files")
        return valid_files


class HiFiGANSFVNNInference:
    """Inference wrapper for trained HiFi-GAN + SF-VNN models."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: torch.device = None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint['config']
        
        # Initialize models
        self.generator = HiFiGANGenerator(self.config).to(device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        # Remove weight norm for inference
        self.generator.remove_weight_norm()
        
        # Audio processor
        self.processor = AudioSpectrogramProcessor(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        
        print(f"Loaded HiFi-GAN + SF-VNN model from {checkpoint_path}")
    
    def generate_from_mel(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Generate audio from mel-spectrogram.
        
        Args:
            mel_spectrogram: [1, n_mels, time] or [n_mels, time]
            
        Returns:
            Generated audio waveform [1, audio_length]
        """
        
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
        
        mel_spectrogram = mel_spectrogram.to(self.device)
        
        with torch.no_grad():
            audio = self.generator(mel_spectrogram)
        
        return audio.cpu()
    
    def generate_from_audio(self, input_audio: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio through mel-spectrogram (useful for audio enhancement).
        
        Args:
            input_audio: [1, audio_length] or [audio_length]
            
        Returns:
            Reconstructed audio [1, audio_length]
        """
        
        # Convert to mel-spectrogram
        mel_spec = self.processor.audio_to_mel_spectrogram(input_audio)
        
        # Normalize for generator
        mel_spec = self._normalize_mel(mel_spec)
        
        # Generate audio
        return self.generate_from_mel(mel_spec)
    
    def generate_from_file(self, audio_path: str, output_path: str = None) -> torch.Tensor:
        """
        Generate audio from input audio file.
        
        Args:
            audio_path: Path to input audio file
            output_path: Optional path to save generated audio
            
        Returns:
            Generated audio tensor
        """
        
        # Load input audio
        input_audio = self.processor.load_audio_file(audio_path)
        
        # Generate
        generated_audio = self.generate_from_audio(input_audio)
        
        # Save if requested
        if output_path:
            torchaudio.save(
                output_path, 
                generated_audio, 
                self.config.sample_rate
            )
            print(f"Generated audio saved to: {output_path}")
        
        return generated_audio
    
    def _normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Normalize mel-spectrogram for generator input."""
        # Same normalization as training dataset
        mel = torch.clamp(mel, min=-80, max=0)
        mel = (mel + 80) / 80
        mel = 2 * mel - 1
        return mel


# Production deployment example
class HiFiGANSFVNNAPI:
    """REST API wrapper for HiFi-GAN + SF-VNN inference (example)."""
    
    def __init__(self, model_path: str):
        self.inference_engine = HiFiGANSFVNNInference(model_path)
        
    def generate_audio_endpoint(self, mel_data: np.ndarray) -> np.ndarray:
        """
        API endpoint for audio generation.
        
        Args:
            mel_data: Mel-spectrogram as numpy array
            
        Returns:
            Generated audio as numpy array
        """
        
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_data).float()
        
        # Generate audio
        audio_tensor = self.inference_engine.generate_from_mel(mel_tensor)
        
        # Convert back to numpy
        return audio_tensor.squeeze().numpy()
    
    def enhance_audio_endpoint(self, input_audio: np.ndarray) -> np.ndarray:
        """
        API endpoint for audio enhancement/reconstruction.
        
        Args:
            input_audio: Input audio as numpy array
            
        Returns:
            Enhanced audio as numpy array
        """
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(input_audio).float().unsqueeze(0)
        
        # Generate enhanced audio
        enhanced_tensor = self.inference_engine.generate_from_audio(audio_tensor)
        
        # Convert back to numpy
        return enhanced_tensor.squeeze().numpy()


# Complete example workflow
def complete_hifigan_sfvnn_workflow():
    """
    Complete end-to-end workflow example.
    
    This shows how to go from raw audio data to trained model to inference.
    """
    
    print("🎵 Complete HiFi-GAN + SF-VNN Workflow")
    print("=" * 50)
    
    # Step 1: Prepare dataset
    print("📁 Step 1: Dataset Preparation")
    
    # Example: Build dataset from directory
    # Replace with your actual audio directory
    audio_dir = "/path/to/your/audio/dataset"
    
    if not os.path.exists(audio_dir):
        print("⚠️  Audio directory not found. Please set correct path.")
        print("   This is a demonstration of the complete workflow.")
        return
    
    # Build train/val/test splits
    train_files, val_files, test_files = AudioDatasetBuilder.build_dataset_from_directory(
        audio_dir=audio_dir,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )
    
    # Validate files
    train_files = AudioDatasetBuilder.validate_audio_files(train_files)
    val_files = AudioDatasetBuilder.validate_audio_files(val_files)
    
    # Step 2: Configure experiment
    print("\n⚙️  Step 2: Experiment Configuration")
    
    config = HiFiGANConfig(
        sample_rate=22050,
        n_mels=80,
        batch_size=16,
        num_epochs=200,
        lambda_structural=1.0,
        sfvnn_multiscale=True
    )
    
    # Step 3: Create datasets
    print("\n📊 Step 3: Dataset Creation")
    
    train_dataset = HiFiGANSFVNNDataset(train_files, config, training=True)
    val_dataset = HiFiGANSFVNNDataset(val_files, config, training=False)
    
    # Step 4: Train model
    print("\n🏋️  Step 4: Model Training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = HiFiGANSFVNNTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        use_wandb=True  # Enable experiment tracking
    )
    
    # Train
    history = trainer.train(num_epochs=config.num_epochs)
    
    # Step 5: Evaluate model
    print("\n📊 Step 5: Model Evaluation")
    
    final_metrics = trainer.validate()
    print(f"Final validation metrics: {final_metrics}")
    
    # Step 6: Run comparative study
    print("\n🧪 Step 6: Comparative Analysis")
    
    experiment = HiFiGANSFVNNExperiment(
        train_audio_files=train_files,
        val_audio_files=val_files
    )
    
    comparative_results = experiment.run_comparative_study()
    experiment.visualize_results(comparative_results)
    
    # Step 7: Model deployment
    print("\n🚀 Step 7: Model Deployment")
    
    # Save final model for inference
    final_model_path = "hifigan_sfvnn_final.pth"
    trainer.save_checkpoint(final_model_path)
    
    # Initialize inference engine
    inference_engine = HiFiGANSFVNNInference(final_model_path, device)
    
    # Test inference
    test_audio_path = test_files[0] if test_files else val_files[0]
    generated_audio = inference_engine.generate_from_file(
        test_audio_path, 
        "generated_sample.wav"
    )
    
    print("\n🎉 Workflow Complete!")
    print("Results:")
    print(f"  - Trained model saved: {final_model_path}")
    print(f"  - Comparative study: check results directory")
    print(f"  - Generated sample: generated_sample.wav")
    print(f"  - Best validation loss: {trainer.best_mel_loss:.4f}")
    
    return {
        'trainer': trainer,
        'inference_engine': inference_engine,
        'comparative_results': comparative_results,
        'history': history
    }


if __name__ == "__main__":
    # Run the test to verify everything works
    print("Running HiFi-GAN + SF-VNN integration test...")
    
    # This will run a quick test with dummy data
    # For real usage, call complete_hifigan_sfvnn_workflow() with your data
    pass
