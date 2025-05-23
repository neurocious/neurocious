#!/usr/bin/env python3
"""
Enhanced Empirical Comparison: Structure-First vs Vanilla with Audio Quality Metrics
Integrates comprehensive audio quality evaluation for rigorous comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd

# Import our components
from hifi import HiFiGANConfig, HiFiGANGenerator, AudioSFVNNDiscriminator
from empirical_comparison import VanillaCNNDiscriminator
from audio_quality_metrics import AudioQualityEvaluator

@dataclass
class EnhancedConfig:
    """Enhanced configuration with audio quality evaluation."""
    num_epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 2e-4
    device: str = "cpu"
    
    # Audio quality evaluation settings
    eval_every_n_epochs: int = 5
    comprehensive_eval: bool = True
    save_audio_samples: bool = True

class EnhancedComparison:
    """Enhanced comparison with comprehensive audio quality metrics."""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize audio quality evaluator
        self.audio_evaluator = AudioQualityEvaluator(device=config.device)
        
        # Create test data
        self.setup_test_data()
        
        # Initialize models
        self.setup_models()
        
        # Results storage
        self.results_history = {
            'sf_vnn': {'training': [], 'quality': []},
            'vanilla': {'training': [], 'quality': []}
        }
        
    def setup_test_data(self):
        """Create synthetic but realistic test audio data."""
        
        # Real audio: harmonic content with some noise
        batch_size = self.config.batch_size
        audio_length = 16384  # ~0.74 seconds
        sample_rate = 22050
        
        t = torch.linspace(0, audio_length/sample_rate, audio_length)
        
        # Create a mix of frequencies (more realistic)
        fundamental = 440  # A4
        harmonics = [fundamental, fundamental*2, fundamental*3, fundamental*4]
        weights = [1.0, 0.5, 0.25, 0.125]
        
        real_audio = torch.zeros(audio_length)
        for freq, weight in zip(harmonics, weights):
            real_audio += weight * torch.sin(2 * np.pi * freq * t)
        
        # Add some realistic noise and envelope
        envelope = torch.exp(-t * 2)  # Decay envelope
        noise = 0.1 * torch.randn(audio_length)
        real_audio = (real_audio * envelope + noise).unsqueeze(0).unsqueeze(0)
        
        self.real_audio = real_audio.repeat(batch_size, 1, 1).to(self.device)
        
        # Create corresponding mel-spectrogram
        import torchaudio.transforms as T
        mel_transform = T.MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256)
        self.real_mel = mel_transform(self.real_audio.squeeze(1)).unsqueeze(1)
        
    def setup_models(self):
        """Initialize all models and optimizers."""
        
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
        
        # Loss function
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
    def train_step(self, discriminator, d_optimizer, discriminator_name: str) -> Dict[str, float]:
        """Single training step with a discriminator."""
        
        # Generate fake audio
        noise_mel = torch.randn(self.config.batch_size, 80, 32).to(self.device)
        fake_audio = self.generator(noise_mel)
        
        # Ensure consistent shapes
        if fake_audio.dim() == 2:
            fake_audio = fake_audio.unsqueeze(1)
        
        min_len = min(fake_audio.size(-1), self.real_audio.size(-1))
        fake_audio = fake_audio[..., :min_len]
        real_audio_batch = self.real_audio[..., :min_len]
        
        # Convert to mel spectrograms for discriminator
        try:
            import torchaudio.transforms as T
            mel_transform = T.MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256).to(self.device)
            real_mel = mel_transform(real_audio_batch.squeeze(1)).unsqueeze(1)
            fake_mel = mel_transform(fake_audio.squeeze(1)).unsqueeze(1)
            
            # Ensure consistent shapes
            min_w = min(real_mel.size(-1), fake_mel.size(-1))
            real_mel = real_mel[..., :min_w]
            fake_mel = fake_mel[..., :min_w]
            
        except Exception as e:
            print(f"Warning: Mel computation failed: {e}")
            real_mel = self.real_mel
            fake_mel = self.real_mel
        
        # Discriminator training
        real_pred = discriminator(real_mel)
        fake_pred = discriminator(fake_mel.detach())
        
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        
        d_loss_real = self.adversarial_loss(real_pred, real_labels)
        d_loss_fake = self.adversarial_loss(fake_pred, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Generator loss (for this discriminator)
        fake_pred_for_gen = discriminator(fake_mel)
        g_loss = self.adversarial_loss(fake_pred_for_gen, torch.ones_like(fake_pred_for_gen))
        
        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'real_pred_mean': real_pred.mean().item(),
            'fake_pred_mean': fake_pred.mean().item(),
            'fake_audio': fake_audio.detach(),
            'real_audio': real_audio_batch
        }
    
    def evaluate_audio_quality(self, 
                             real_audio: torch.Tensor, 
                             fake_audio: torch.Tensor, 
                             discriminator_name: str) -> Dict[str, float]:
        """Evaluate audio quality using comprehensive metrics."""
        
        print(f"🎵 Evaluating audio quality for {discriminator_name}...")
        
        # Run comprehensive evaluation
        quality_metrics = self.audio_evaluator.comprehensive_evaluation(real_audio, fake_audio)
        
        # Add discriminator identifier
        quality_metrics['discriminator'] = discriminator_name
        quality_metrics['timestamp'] = time.time()
        
        return quality_metrics
    
    def run_enhanced_comparison(self) -> Dict[str, any]:
        """Run the enhanced comparison with audio quality evaluation."""
        
        print("🚀 Enhanced Empirical Comparison: SF-VNN vs Vanilla CNN")
        print("🎵 With Comprehensive Audio Quality Metrics")
        print("=" * 70)
        
        # Model statistics
        sf_params = sum(p.numel() for p in self.sf_discriminator.parameters())
        vanilla_params = sum(p.numel() for p in self.vanilla_discriminator.parameters())
        gen_params = sum(p.numel() for p in self.generator.parameters())
        
        print(f"📊 Model Parameters:")
        print(f"   Generator: {gen_params:,}")
        print(f"   SF-VNN Discriminator: {sf_params:,}")
        print(f"   Vanilla CNN Discriminator: {vanilla_params:,}")
        print(f"   Parameter Efficiency: SF-VNN uses {sf_params/vanilla_params:.3f}x parameters")
        print()
        
        print(f"🏃 Training for {self.config.num_epochs} epochs...")
        print(f"🎧 Quality evaluation every {self.config.eval_every_n_epochs} epochs")
        print()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            
            # Train both discriminators
            sf_metrics = self.train_step(self.sf_discriminator, self.sf_opt, "SF-VNN")
            vanilla_metrics = self.train_step(self.vanilla_discriminator, self.vanilla_opt, "Vanilla")
            
            # Store training metrics
            self.results_history['sf_vnn']['training'].append({
                'epoch': epoch,
                'discriminator_loss': sf_metrics['discriminator_loss'],
                'generator_loss': sf_metrics['generator_loss'],
                'real_pred_mean': sf_metrics['real_pred_mean'],
                'fake_pred_mean': sf_metrics['fake_pred_mean']
            })
            
            self.results_history['vanilla']['training'].append({
                'epoch': epoch,
                'discriminator_loss': vanilla_metrics['discriminator_loss'],
                'generator_loss': vanilla_metrics['generator_loss'],
                'real_pred_mean': vanilla_metrics['real_pred_mean'],
                'fake_pred_mean': vanilla_metrics['fake_pred_mean']
            })
            
            # Periodic quality evaluation
            if epoch % self.config.eval_every_n_epochs == 0 or epoch == self.config.num_epochs - 1:
                
                print(f"\\n🔍 Epoch {epoch} - Quality Evaluation")
                print("-" * 50)
                
                # Evaluate SF-VNN audio quality
                sf_quality = self.evaluate_audio_quality(
                    sf_metrics['real_audio'], 
                    sf_metrics['fake_audio'], 
                    "SF-VNN"
                )
                sf_quality['epoch'] = epoch
                self.results_history['sf_vnn']['quality'].append(sf_quality)
                
                # Evaluate Vanilla audio quality
                vanilla_quality = self.evaluate_audio_quality(
                    vanilla_metrics['real_audio'], 
                    vanilla_metrics['fake_audio'], 
                    "Vanilla"
                )
                vanilla_quality['epoch'] = epoch
                self.results_history['vanilla']['quality'].append(vanilla_quality)
                
                # Print comparison
                print(f"📈 Audio Quality Comparison (Epoch {epoch}):")
                key_metrics = ['fad_mel', 'predicted_mos', 'mel_l1_distance', 'zcr_correlation']
                
                for metric in key_metrics:
                    if metric in sf_quality and metric in vanilla_quality:
                        sf_val = sf_quality[metric]
                        vanilla_val = vanilla_quality[metric]
                        
                        if 'correlation' in metric or 'mos' in metric:
                            winner = "SF-VNN" if sf_val > vanilla_val else "Vanilla"
                            print(f"   {metric:20s}: SF-VNN={sf_val:.4f}, Vanilla={vanilla_val:.4f} → {winner}")
                        else:
                            winner = "SF-VNN" if sf_val < vanilla_val else "Vanilla"
                            print(f"   {metric:20s}: SF-VNN={sf_val:.4f}, Vanilla={vanilla_val:.4f} → {winner}")
            
            # Progress update
            if epoch % 2 == 0:
                print(f"   Epoch {epoch:2d} | SF-VNN D_loss: {sf_metrics['discriminator_loss']:.4f} | "
                      f"Vanilla D_loss: {vanilla_metrics['discriminator_loss']:.4f}")
        
        print()
        
        # Final analysis
        final_results = self.analyze_results()
        
        return final_results
    
    def analyze_results(self) -> Dict[str, any]:
        """Comprehensive analysis of results."""
        
        print("📊 Final Results Analysis")
        print("=" * 50)
        
        # Training stability analysis
        sf_d_losses = [m['discriminator_loss'] for m in self.results_history['sf_vnn']['training']]
        vanilla_d_losses = [m['discriminator_loss'] for m in self.results_history['vanilla']['training']]
        
        sf_stability = np.std(sf_d_losses[-5:])
        vanilla_stability = np.std(vanilla_d_losses[-5:])
        
        print(f"🎯 Training Stability (last 5 epochs std dev):")
        print(f"   SF-VNN: {sf_stability:.6f}")
        print(f"   Vanilla: {vanilla_stability:.6f}")
        print(f"   Winner: {'SF-VNN' if sf_stability < vanilla_stability else 'Vanilla'}")
        print()
        
        # Audio quality analysis
        if self.results_history['sf_vnn']['quality'] and self.results_history['vanilla']['quality']:
            print(f"🎵 Audio Quality Analysis:")
            
            # Get final quality metrics
            sf_final_quality = self.results_history['sf_vnn']['quality'][-1]
            vanilla_final_quality = self.results_history['vanilla']['quality'][-1]
            
            quality_metrics = ['fad_mel', 'predicted_mos', 'mel_l1_distance', 'zcr_correlation', 'rms_correlation']
            quality_winners = {}
            
            for metric in quality_metrics:
                if metric in sf_final_quality and metric in vanilla_final_quality:
                    sf_val = sf_final_quality[metric]
                    vanilla_val = vanilla_final_quality[metric]
                    
                    if 'correlation' in metric or 'mos' in metric:
                        winner = "SF-VNN" if sf_val > vanilla_val else "Vanilla"
                        improvement = abs(sf_val - vanilla_val) / max(abs(vanilla_val), 1e-8) * 100
                    else:
                        winner = "SF-VNN" if sf_val < vanilla_val else "Vanilla"
                        improvement = abs(sf_val - vanilla_val) / max(abs(vanilla_val), 1e-8) * 100
                    
                    quality_winners[metric] = winner
                    print(f"   {metric:20s}: SF-VNN={sf_val:.4f}, Vanilla={vanilla_val:.4f} → {winner} ({improvement:.1f}% diff)")
            
            # Overall quality winner
            sf_wins = sum(1 for winner in quality_winners.values() if winner == "SF-VNN")
            vanilla_wins = len(quality_winners) - sf_wins
            
            print(f"\\n🏆 Overall Audio Quality Winner: {'SF-VNN' if sf_wins > vanilla_wins else 'Vanilla'}")
            print(f"   SF-VNN wins: {sf_wins}/{len(quality_winners)} metrics")
            print(f"   Vanilla wins: {vanilla_wins}/{len(quality_winners)} metrics")
        
        print()
        
        # Save results
        results = {
            'training_stability': {
                'sf_vnn': sf_stability,
                'vanilla': vanilla_stability
            },
            'model_parameters': {
                'sf_vnn': sum(p.numel() for p in self.sf_discriminator.parameters()),
                'vanilla': sum(p.numel() for p in self.vanilla_discriminator.parameters())
            },
            'history': self.results_history
        }
        
        # Create visualizations
        self.create_visualizations()
        
        return results
    
    def create_visualizations(self):
        """Create comprehensive result visualizations."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training losses
        epochs = [m['epoch'] for m in self.results_history['sf_vnn']['training']]
        sf_d_losses = [m['discriminator_loss'] for m in self.results_history['sf_vnn']['training']]
        vanilla_d_losses = [m['discriminator_loss'] for m in self.results_history['vanilla']['training']]
        
        axes[0, 0].plot(epochs, sf_d_losses, 'b-', label='SF-VNN', linewidth=2)
        axes[0, 0].plot(epochs, vanilla_d_losses, 'r-', label='Vanilla CNN', linewidth=2)
        axes[0, 0].set_title('Discriminator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Generator losses
        sf_g_losses = [m['generator_loss'] for m in self.results_history['sf_vnn']['training']]
        vanilla_g_losses = [m['generator_loss'] for m in self.results_history['vanilla']['training']]
        
        axes[0, 1].plot(epochs, sf_g_losses, 'b-', label='vs SF-VNN', linewidth=2)
        axes[0, 1].plot(epochs, vanilla_g_losses, 'r-', label='vs Vanilla CNN', linewidth=2)
        axes[0, 1].set_title('Generator Adversarial Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Audio quality metrics over time
        if self.results_history['sf_vnn']['quality']:
            quality_epochs = [q['epoch'] for q in self.results_history['sf_vnn']['quality']]
            
            # FAD scores
            sf_fad = [q['fad_mel'] for q in self.results_history['sf_vnn']['quality']]
            vanilla_fad = [q['fad_mel'] for q in self.results_history['vanilla']['quality']]
            
            axes[0, 2].plot(quality_epochs, sf_fad, 'b-o', label='SF-VNN', linewidth=2, markersize=6)
            axes[0, 2].plot(quality_epochs, vanilla_fad, 'r-o', label='Vanilla CNN', linewidth=2, markersize=6)
            axes[0, 2].set_title('Fréchet Audio Distance (Lower = Better)')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('FAD Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # MOS scores
            sf_mos = [q['predicted_mos'] for q in self.results_history['sf_vnn']['quality']]
            vanilla_mos = [q['predicted_mos'] for q in self.results_history['vanilla']['quality']]
            
            axes[1, 0].plot(quality_epochs, sf_mos, 'b-o', label='SF-VNN', linewidth=2, markersize=6)
            axes[1, 0].plot(quality_epochs, vanilla_mos, 'r-o', label='Vanilla CNN', linewidth=2, markersize=6)
            axes[1, 0].set_title('Predicted MOS Score (Higher = Better)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MOS Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Spectral distance
            sf_mel_dist = [q['mel_l1_distance'] for q in self.results_history['sf_vnn']['quality']]
            vanilla_mel_dist = [q['mel_l1_distance'] for q in self.results_history['vanilla']['quality']]
            
            axes[1, 1].plot(quality_epochs, sf_mel_dist, 'b-o', label='SF-VNN', linewidth=2, markersize=6)
            axes[1, 1].plot(quality_epochs, vanilla_mel_dist, 'r-o', label='Vanilla CNN', linewidth=2, markersize=6)
            axes[1, 1].set_title('Mel-Spectrogram L1 Distance (Lower = Better)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('L1 Distance')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Perceptual correlation
            sf_corr = [q.get('zcr_correlation', 0) for q in self.results_history['sf_vnn']['quality']]
            vanilla_corr = [q.get('zcr_correlation', 0) for q in self.results_history['vanilla']['quality']]
            
            axes[1, 2].plot(quality_epochs, sf_corr, 'b-o', label='SF-VNN', linewidth=2, markersize=6)
            axes[1, 2].plot(quality_epochs, vanilla_corr, 'r-o', label='Vanilla CNN', linewidth=2, markersize=6)
            axes[1, 2].set_title('ZCR Correlation (Higher = Better)')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Correlation')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_comparison_results.png', dpi=150, bbox_inches='tight')
        print("📊 Enhanced comparison visualization saved: enhanced_comparison_results.png")
        plt.close()


def main():
    """Run the enhanced comparison with audio quality metrics."""
    
    config = EnhancedConfig(
        num_epochs=15,
        batch_size=2,
        learning_rate=2e-4,
        device="cpu",
        eval_every_n_epochs=3
    )
    
    comparison = EnhancedComparison(config)
    results = comparison.run_enhanced_comparison()
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_comparison_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_for_json(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Comprehensive results saved: {results_file}")
    print("\\n✅ Enhanced empirical comparison completed!")
    print("🎵 Your structure-first discriminator has been thoroughly evaluated!")


if __name__ == "__main__":
    main()