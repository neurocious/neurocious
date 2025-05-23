#!/usr/bin/env python3
"""
Quick Empirical Comparison: Structure-First vs Vanilla Discriminator
Simplified version for demonstration and rapid testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import our components
from hifi import HiFiGANConfig, HiFiGANGenerator, AudioSFVNNDiscriminator
from empirical_comparison import VanillaCNNDiscriminator

@dataclass
class QuickConfig:
    """Simplified configuration for quick testing."""
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 2e-4
    device: str = "cpu"
    
class QuickComparison:
    """Simplified comparison framework for rapid testing."""
    
    def __init__(self, config: QuickConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create synthetic test data
        self.real_audio = torch.randn(config.batch_size, 1, 8192).to(self.device)
        self.real_mel = torch.randn(config.batch_size, 1, 80, 32).to(self.device)
        
        # Initialize models
        self.setup_models()
        
    def setup_models(self):
        """Initialize generator and both discriminators."""
        
        # Generator
        gen_config = HiFiGANConfig()
        self.generator = HiFiGANGenerator(gen_config).to(self.device)
        
        # SF-VNN Discriminator
        self.sf_discriminator = AudioSFVNNDiscriminator(
            input_channels=1,
            vector_channels=[32, 64, 128],
            window_size=3,
            sigma=1.0,
            multiscale_analysis=True
        ).to(self.device)
        
        # Vanilla CNN Discriminator
        self.vanilla_discriminator = VanillaCNNDiscriminator({
            'channels': [32, 64, 128],
            'kernel_sizes': [(3, 9), (3, 8), (3, 8)],
            'strides': [(1, 1), (1, 2), (1, 2)]
        }).to(self.device)
        
        # Optimizers
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        self.sf_opt = torch.optim.Adam(self.sf_discriminator.parameters(), lr=self.config.learning_rate)
        self.vanilla_opt = torch.optim.Adam(self.vanilla_discriminator.parameters(), lr=self.config.learning_rate)
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        
    def train_step(self, discriminator, d_optimizer, discriminator_name):
        """Single training step for a discriminator."""
        
        # Generate fake audio (using mel-spectrogram as input)
        noise_mel = torch.randn(self.config.batch_size, 80, 32).to(self.device)
        fake_audio = self.generator(noise_mel)
        
        # Ensure consistent shapes
        if fake_audio.dim() == 2:
            fake_audio = fake_audio.unsqueeze(1)
        if fake_audio.size(-1) > self.real_audio.size(-1):
            fake_audio = fake_audio[..., :self.real_audio.size(-1)]
        elif fake_audio.size(-1) < self.real_audio.size(-1):
            padding = self.real_audio.size(-1) - fake_audio.size(-1)
            fake_audio = F.pad(fake_audio, (0, padding))
            
        # Convert to mel spectrograms
        try:
            import torchaudio.transforms as T
            mel_transform = T.MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256).to(self.device)
            real_mel = mel_transform(self.real_audio.squeeze(1)).unsqueeze(1)
            fake_mel = mel_transform(fake_audio.squeeze(1)).unsqueeze(1)
            
            # Ensure consistent mel shapes
            min_w = min(real_mel.size(-1), fake_mel.size(-1))
            real_mel = real_mel[..., :min_w]
            fake_mel = fake_mel[..., :min_w]
            
        except Exception as e:
            print(f"Warning: Mel computation failed: {e}")
            real_mel = self.real_mel
            fake_mel = self.real_mel  # Fallback
        
        # Discriminator predictions
        real_pred = discriminator(real_mel)
        fake_pred = discriminator(fake_mel.detach())
        
        # Discriminator loss
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        
        d_loss_real = self.adversarial_loss(real_pred, real_labels)
        d_loss_fake = self.adversarial_loss(fake_pred, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # Update discriminator
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Generator adversarial loss
        fake_pred_for_gen = discriminator(fake_mel)
        g_loss_adv = self.adversarial_loss(fake_pred_for_gen, torch.ones_like(fake_pred_for_gen))
        
        return {
            'discriminator_loss': d_loss.item(),
            'generator_adv_loss': g_loss_adv.item(),
            'real_pred_mean': real_pred.mean().item(),
            'fake_pred_mean': fake_pred.mean().item()
        }
    
    def run_comparison(self):
        """Run the quick comparison."""
        
        print("🚀 Quick Empirical Comparison: SF-VNN vs Vanilla CNN")
        print("=" * 60)
        
        # Parameter counts
        sf_params = sum(p.numel() for p in self.sf_discriminator.parameters())
        vanilla_params = sum(p.numel() for p in self.vanilla_discriminator.parameters())
        gen_params = sum(p.numel() for p in self.generator.parameters())
        
        print(f"📊 Model Parameters:")
        print(f"   Generator: {gen_params:,}")
        print(f"   SF-VNN Discriminator: {sf_params:,}")
        print(f"   Vanilla CNN Discriminator: {vanilla_params:,}")
        print(f"   Parameter Ratio (SF/Vanilla): {sf_params/vanilla_params:.3f}")
        print()
        
        # Training history
        sf_history = {'d_loss': [], 'g_loss': [], 'real_pred': [], 'fake_pred': []}
        vanilla_history = {'d_loss': [], 'g_loss': [], 'real_pred': [], 'fake_pred': []}
        
        print(f"🏃 Training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            
            # Train SF-VNN discriminator
            sf_metrics = self.train_step(self.sf_discriminator, self.sf_opt, "SF-VNN")
            sf_history['d_loss'].append(sf_metrics['discriminator_loss'])
            sf_history['g_loss'].append(sf_metrics['generator_adv_loss'])
            sf_history['real_pred'].append(sf_metrics['real_pred_mean'])
            sf_history['fake_pred'].append(sf_metrics['fake_pred_mean'])
            
            # Train Vanilla discriminator
            vanilla_metrics = self.train_step(self.vanilla_discriminator, self.vanilla_opt, "Vanilla")
            vanilla_history['d_loss'].append(vanilla_metrics['discriminator_loss'])
            vanilla_history['g_loss'].append(vanilla_metrics['generator_adv_loss'])
            vanilla_history['real_pred'].append(vanilla_metrics['real_pred_mean'])
            vanilla_history['fake_pred'].append(vanilla_metrics['fake_pred_mean'])
            
            if epoch % 2 == 0:
                print(f"   Epoch {epoch:2d} | SF-VNN D_loss: {sf_metrics['discriminator_loss']:.4f} | "
                      f"Vanilla D_loss: {vanilla_metrics['discriminator_loss']:.4f}")
        
        print()
        
        # Results analysis
        self.analyze_results(sf_history, vanilla_history, sf_params, vanilla_params)
        
        return sf_history, vanilla_history
    
    def analyze_results(self, sf_history, vanilla_history, sf_params, vanilla_params):
        """Analyze and display results."""
        
        print("📈 Results Analysis")
        print("-" * 40)
        
        # Final losses
        sf_final_d = sf_history['d_loss'][-1]
        vanilla_final_d = vanilla_history['d_loss'][-1]
        sf_final_g = sf_history['g_loss'][-1]
        vanilla_final_g = vanilla_history['g_loss'][-1]
        
        print(f"Final Discriminator Losses:")
        print(f"   SF-VNN: {sf_final_d:.4f}")
        print(f"   Vanilla: {vanilla_final_d:.4f}")
        print(f"   Difference: {sf_final_d - vanilla_final_d:.4f}")
        print()
        
        print(f"Final Generator Adversarial Losses:")
        print(f"   vs SF-VNN: {sf_final_g:.4f}")
        print(f"   vs Vanilla: {vanilla_final_g:.4f}")
        print(f"   Difference: {sf_final_g - vanilla_final_g:.4f}")
        print()
        
        # Stability analysis
        sf_d_std = np.std(sf_history['d_loss'][-5:])
        vanilla_d_std = np.std(vanilla_history['d_loss'][-5:])
        
        print(f"Training Stability (last 5 epochs std dev):")
        print(f"   SF-VNN: {sf_d_std:.4f}")
        print(f"   Vanilla: {vanilla_d_std:.4f}")
        print()
        
        # Discrimination ability
        sf_sep = np.mean(sf_history['real_pred'][-5:]) - np.mean(sf_history['fake_pred'][-5:])
        vanilla_sep = np.mean(vanilla_history['real_pred'][-5:]) - np.mean(vanilla_history['fake_pred'][-5:])
        
        print(f"Discrimination Ability (real-fake separation):")
        print(f"   SF-VNN: {sf_sep:.4f}")
        print(f"   Vanilla: {vanilla_sep:.4f}")
        print()
        
        # Efficiency
        efficiency_sf = sf_sep / sf_params * 1e6  # Per million parameters
        efficiency_vanilla = vanilla_sep / vanilla_params * 1e6
        
        print(f"Parameter Efficiency (discrimination per million params):")
        print(f"   SF-VNN: {efficiency_sf:.6f}")
        print(f"   Vanilla: {efficiency_vanilla:.6f}")
        print(f"   SF-VNN Efficiency Ratio: {efficiency_sf/efficiency_vanilla:.3f}x")
        print()
        
        # Summary
        winner = "SF-VNN" if sf_final_d < vanilla_final_d else "Vanilla"
        stable_winner = "SF-VNN" if sf_d_std < vanilla_d_std else "Vanilla"
        efficient_winner = "SF-VNN" if efficiency_sf > efficiency_vanilla else "Vanilla"
        
        print("🏆 Summary:")
        print(f"   Lower discriminator loss: {winner}")
        print(f"   More stable training: {stable_winner}")
        print(f"   Higher parameter efficiency: {efficient_winner}")
        print()
        
        # Create simple visualization
        self.plot_results(sf_history, vanilla_history)
    
    def plot_results(self, sf_history, vanilla_history):
        """Create comparison plots."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(len(sf_history['d_loss']))
        
        # Discriminator losses
        ax1.plot(epochs, sf_history['d_loss'], 'b-', label='SF-VNN', linewidth=2)
        ax1.plot(epochs, vanilla_history['d_loss'], 'r-', label='Vanilla CNN', linewidth=2)
        ax1.set_title('Discriminator Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Generator adversarial losses
        ax2.plot(epochs, sf_history['g_loss'], 'b-', label='vs SF-VNN', linewidth=2)
        ax2.plot(epochs, vanilla_history['g_loss'], 'r-', label='vs Vanilla CNN', linewidth=2)
        ax2.set_title('Generator Adversarial Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Real predictions
        ax3.plot(epochs, sf_history['real_pred'], 'b-', label='SF-VNN', linewidth=2)
        ax3.plot(epochs, vanilla_history['real_pred'], 'r-', label='Vanilla CNN', linewidth=2)
        ax3.set_title('Real Audio Predictions')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Prediction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Fake predictions
        ax4.plot(epochs, sf_history['fake_pred'], 'b-', label='SF-VNN', linewidth=2)
        ax4.plot(epochs, vanilla_history['fake_pred'], 'r-', label='Vanilla CNN', linewidth=2)
        ax4.set_title('Fake Audio Predictions')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Prediction')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quick_comparison_results.png', dpi=150, bbox_inches='tight')
        print("📊 Results saved to: quick_comparison_results.png")
        plt.close()


def main():
    """Run the quick comparison."""
    
    # Configuration
    config = QuickConfig(
        num_epochs=15,
        batch_size=4,
        learning_rate=2e-4,
        device="cpu"
    )
    
    # Run comparison
    comparison = QuickComparison(config)
    sf_history, vanilla_history = comparison.run_comparison()
    
    # Save results
    results = {
        'config': {
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'device': config.device
        },
        'sf_vnn_history': sf_history,
        'vanilla_history': vanilla_history,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('quick_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 Results saved to: quick_comparison_results.json")
    print("✅ Quick comparison completed!")


if __name__ == "__main__":
    main()