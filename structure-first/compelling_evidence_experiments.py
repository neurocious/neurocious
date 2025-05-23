#!/usr/bin/env python3
"""
Compelling Evidence Experiments: Structure-First vs Vanilla
Rigorous experiments designed to find where SF-VNN truly excels.
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
import pandas as pd
from collections import defaultdict

# Import our components
from hifi import HiFiGANConfig, HiFiGANGenerator, AudioSFVNNDiscriminator
from empirical_comparison import VanillaCNNDiscriminator

@dataclass
class CompellingExperimentConfig:
    """Configuration for rigorous experiments."""
    
    # Long-term stability test
    stability_epochs: int = 50
    stability_batch_size: int = 4
    stability_lr: float = 2e-4
    
    # Adversarial robustness test
    adversarial_noise_levels: List[float] = None
    adversarial_epochs: int = 20
    
    # Convergence test
    convergence_epochs: int = 100
    convergence_lr_schedule: bool = True
    
    # Generalization test
    test_different_audio_types: bool = True
    
    # Computational efficiency
    measure_training_time: bool = True
    measure_memory_usage: bool = True
    
    def __post_init__(self):
        if self.adversarial_noise_levels is None:
            self.adversarial_noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]

class CompellingEvidenceExperiments:
    """Rigorous experiments to demonstrate SF-VNN advantages."""
    
    def __init__(self, config: CompellingExperimentConfig):
        self.config = config
        self.device = torch.device("cpu")
        
        # Results storage
        self.all_results = {}
        
        # Setup audio types for generalization testing
        self.setup_audio_types()
        
        print("🔬 Compelling Evidence Experiments Initialized")
        print("🎯 Goal: Find where Structure-First truly excels")
        print("=" * 60)
    
    def setup_audio_types(self):
        """Create different types of audio for generalization testing."""
        
        audio_length = 16384
        sample_rate = 22050
        t = torch.linspace(0, audio_length/sample_rate, audio_length)
        
        self.audio_types = {}
        
        # 1. Pure sine wave (simple)
        self.audio_types['sine'] = torch.sin(2 * np.pi * 440 * t)
        
        # 2. Musical chord (harmonic)
        chord = torch.zeros(audio_length)
        for freq in [440, 554.37, 659.25]:  # A major chord
            chord += torch.sin(2 * np.pi * freq * t)
        self.audio_types['chord'] = chord
        
        # 3. Noise burst (chaotic)
        noise = torch.randn(audio_length) * 0.5
        envelope = torch.exp(-t * 3)
        self.audio_types['noise'] = noise * envelope
        
        # 4. Frequency sweep (dynamic)
        freq_sweep = 200 + 1000 * t  # 200Hz to 1200Hz sweep
        self.audio_types['sweep'] = torch.sin(2 * np.pi * freq_sweep * t)
        
        # 5. Complex mixture (realistic)
        mixture = (self.audio_types['chord'] * 0.6 + 
                  self.audio_types['noise'] * 0.2 + 
                  torch.sin(2 * np.pi * 880 * t) * 0.2)
        self.audio_types['mixture'] = mixture
        
        print(f"📊 Created {len(self.audio_types)} audio types for generalization testing")
    
    def create_models(self):
        """Create fresh model instances."""
        
        # Generator
        gen_config = HiFiGANConfig()
        generator = HiFiGANGenerator(gen_config).to(self.device)
        
        # SF-VNN Discriminator
        sf_discriminator = AudioSFVNNDiscriminator(
            input_channels=1,
            vector_channels=[32, 64, 128],
            window_size=3,
            sigma=1.0,
            multiscale_analysis=True
        ).to(self.device)
        
        # Vanilla CNN Discriminator
        vanilla_discriminator = VanillaCNNDiscriminator({
            'channels': [32, 64, 128],
            'kernel_sizes': [(3, 9), (3, 8), (3, 8)],
            'strides': [(1, 1), (1, 2), (1, 2)]
        }).to(self.device)
        
        return generator, sf_discriminator, vanilla_discriminator
    
    def experiment_1_long_term_stability(self) -> Dict:
        """Test long-term training stability."""
        
        print("\\n🧪 EXPERIMENT 1: Long-Term Training Stability")
        print("=" * 50)
        print(f"Training for {self.config.stability_epochs} epochs to test stability...")
        
        generator, sf_disc, vanilla_disc = self.create_models()
        
        # Optimizers
        sf_opt = torch.optim.Adam(sf_disc.parameters(), lr=self.config.stability_lr)
        vanilla_opt = torch.optim.Adam(vanilla_disc.parameters(), lr=self.config.stability_lr)
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Training data
        batch_size = self.config.stability_batch_size
        real_audio = self.audio_types['chord'].unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Results tracking
        sf_losses = []
        vanilla_losses = []
        sf_gradients = []
        vanilla_gradients = []
        
        print("🏃 Training discriminators...")
        
        for epoch in range(self.config.stability_epochs):
            
            # Generate fake audio
            with torch.no_grad():
                noise_mel = torch.randn(batch_size, 80, 32)
                fake_audio = generator(noise_mel)
                if fake_audio.dim() == 2:
                    fake_audio = fake_audio.unsqueeze(1)
                
                min_len = min(fake_audio.size(-1), real_audio.size(-1))
                fake_audio = fake_audio[..., :min_len]
                real_audio_batch = real_audio[..., :min_len]
            
            # Convert to mel spectrograms
            import torchaudio.transforms as T
            mel_transform = T.MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256)
            real_mel = mel_transform(real_audio_batch.squeeze(1)).unsqueeze(1)
            fake_mel = mel_transform(fake_audio.squeeze(1)).unsqueeze(1)
            
            min_w = min(real_mel.size(-1), fake_mel.size(-1))
            real_mel = real_mel[..., :min_w]
            fake_mel = fake_mel[..., :min_w]
            
            # Train SF-VNN
            sf_opt.zero_grad()
            sf_real_pred = sf_disc(real_mel)
            sf_fake_pred = sf_disc(fake_mel.detach())
            
            sf_loss = (criterion(sf_real_pred, torch.ones_like(sf_real_pred)) + 
                      criterion(sf_fake_pred, torch.zeros_like(sf_fake_pred))) / 2
            
            sf_loss.backward()
            
            # Calculate gradient norm for SF-VNN
            sf_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                                  for p in sf_disc.parameters() if p.grad is not None]))
            sf_gradients.append(sf_grad_norm.item())
            
            sf_opt.step()
            sf_losses.append(sf_loss.item())
            
            # Train Vanilla
            vanilla_opt.zero_grad()
            vanilla_real_pred = vanilla_disc(real_mel)
            vanilla_fake_pred = vanilla_disc(fake_mel.detach())
            
            vanilla_loss = (criterion(vanilla_real_pred, torch.ones_like(vanilla_real_pred)) + 
                           criterion(vanilla_fake_pred, torch.zeros_like(vanilla_fake_pred))) / 2
            
            vanilla_loss.backward()
            
            # Calculate gradient norm for Vanilla
            vanilla_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                                       for p in vanilla_disc.parameters() if p.grad is not None]))
            vanilla_gradients.append(vanilla_grad_norm.item())
            
            vanilla_opt.step()
            vanilla_losses.append(vanilla_loss.item())
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:2d} | SF-VNN: {sf_loss.item():.4f} | Vanilla: {vanilla_loss.item():.4f}")
        
        # Analyze stability
        results = self.analyze_stability(sf_losses, vanilla_losses, sf_gradients, vanilla_gradients)
        
        print("\\n📊 Long-Term Stability Results:")
        print(f"   SF-VNN loss std (final 20 epochs): {results['sf_final_std']:.6f}")
        print(f"   Vanilla loss std (final 20 epochs): {results['vanilla_final_std']:.6f}")
        print(f"   🏆 Stability winner: {results['stability_winner']}")
        print(f"   SF-VNN gradient stability: {results['sf_grad_std']:.6f}")
        print(f"   Vanilla gradient stability: {results['vanilla_grad_std']:.6f}")
        print(f"   🏆 Gradient stability winner: {results['grad_stability_winner']}")
        
        return results
    
    def analyze_stability(self, sf_losses, vanilla_losses, sf_grads, vanilla_grads):
        """Analyze training stability metrics."""
        
        # Loss stability (final 20 epochs)
        sf_final_std = np.std(sf_losses[-20:])
        vanilla_final_std = np.std(vanilla_losses[-20:])
        stability_winner = "SF-VNN" if sf_final_std < vanilla_final_std else "Vanilla"
        
        # Gradient stability
        sf_grad_std = np.std(sf_grads[-20:])
        vanilla_grad_std = np.std(vanilla_grads[-20:])
        grad_stability_winner = "SF-VNN" if sf_grad_std < vanilla_grad_std else "Vanilla"
        
        # Convergence analysis
        sf_trend = np.polyfit(range(len(sf_losses)), sf_losses, 1)[0]  # Slope
        vanilla_trend = np.polyfit(range(len(vanilla_losses)), vanilla_losses, 1)[0]
        
        return {
            'sf_losses': sf_losses,
            'vanilla_losses': vanilla_losses,
            'sf_final_std': sf_final_std,
            'vanilla_final_std': vanilla_final_std,
            'stability_winner': stability_winner,
            'sf_grad_std': sf_grad_std,
            'vanilla_grad_std': vanilla_grad_std,
            'grad_stability_winner': grad_stability_winner,
            'sf_trend': sf_trend,
            'vanilla_trend': vanilla_trend,
            'stability_ratio': vanilla_final_std / sf_final_std if sf_final_std > 0 else float('inf')
        }
    
    def experiment_2_adversarial_robustness(self) -> Dict:
        """Test robustness to adversarial conditions."""
        
        print("\\n🧪 EXPERIMENT 2: Adversarial Robustness")
        print("=" * 50)
        print("Testing discriminator performance under adversarial noise...")
        
        results = {}
        
        for noise_level in self.config.adversarial_noise_levels:
            print(f"\\n🔊 Testing noise level: {noise_level}")
            
            generator, sf_disc, vanilla_disc = self.create_models()
            
            # Add noise to training data
            real_audio = self.audio_types['chord'].unsqueeze(0).unsqueeze(0).repeat(2, 1, 1)
            
            if noise_level > 0:
                noise = torch.randn_like(real_audio) * noise_level
                noisy_real_audio = real_audio + noise
            else:
                noisy_real_audio = real_audio
            
            # Quick training session
            sf_opt = torch.optim.Adam(sf_disc.parameters(), lr=2e-4)
            vanilla_opt = torch.optim.Adam(vanilla_disc.parameters(), lr=2e-4)
            criterion = nn.BCEWithLogitsLoss()
            
            sf_performance = []
            vanilla_performance = []
            
            for epoch in range(self.config.adversarial_epochs):
                
                # Generate fake audio
                with torch.no_grad():
                    noise_mel = torch.randn(2, 80, 32)
                    fake_audio = generator(noise_mel)
                    if fake_audio.dim() == 2:
                        fake_audio = fake_audio.unsqueeze(1)
                    
                    min_len = min(fake_audio.size(-1), noisy_real_audio.size(-1))
                    fake_audio = fake_audio[..., :min_len]
                    real_batch = noisy_real_audio[..., :min_len]
                
                # Convert to mel
                import torchaudio.transforms as T
                mel_transform = T.MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256)
                real_mel = mel_transform(real_batch.squeeze(1)).unsqueeze(1)
                fake_mel = mel_transform(fake_audio.squeeze(1)).unsqueeze(1)
                
                min_w = min(real_mel.size(-1), fake_mel.size(-1))
                real_mel = real_mel[..., :min_w]
                fake_mel = fake_mel[..., :min_w]
                
                # Evaluate discrimination ability
                with torch.no_grad():
                    sf_real_pred = torch.sigmoid(sf_disc(real_mel))
                    sf_fake_pred = torch.sigmoid(sf_disc(fake_mel))
                    sf_discrimination = abs(sf_real_pred.mean() - sf_fake_pred.mean()).item()
                    
                    vanilla_real_pred = torch.sigmoid(vanilla_disc(real_mel))
                    vanilla_fake_pred = torch.sigmoid(vanilla_disc(fake_mel))
                    vanilla_discrimination = abs(vanilla_real_pred.mean() - vanilla_fake_pred.mean()).item()
                    
                    sf_performance.append(sf_discrimination)
                    vanilla_performance.append(vanilla_discrimination)
                
                # Train one step
                # SF-VNN training
                sf_opt.zero_grad()
                sf_real_pred = sf_disc(real_mel)
                sf_fake_pred = sf_disc(fake_mel.detach())
                sf_loss = (criterion(sf_real_pred, torch.ones_like(sf_real_pred)) + 
                          criterion(sf_fake_pred, torch.zeros_like(sf_fake_pred))) / 2
                sf_loss.backward()
                sf_opt.step()
                
                # Vanilla training
                vanilla_opt.zero_grad()
                vanilla_real_pred = vanilla_disc(real_mel)
                vanilla_fake_pred = vanilla_disc(fake_mel.detach())
                vanilla_loss = (criterion(vanilla_real_pred, torch.ones_like(vanilla_real_pred)) + 
                               criterion(vanilla_fake_pred, torch.zeros_like(vanilla_fake_pred))) / 2
                vanilla_loss.backward()
                vanilla_opt.step()
            
            # Store results for this noise level
            sf_avg_performance = np.mean(sf_performance[-10:])  # Last 10 epochs
            vanilla_avg_performance = np.mean(vanilla_performance[-10:])
            
            results[f'noise_{noise_level}'] = {
                'sf_performance': sf_avg_performance,
                'vanilla_performance': vanilla_avg_performance,
                'performance_ratio': sf_avg_performance / (vanilla_avg_performance + 1e-8),
                'winner': 'SF-VNN' if sf_avg_performance > vanilla_avg_performance else 'Vanilla'
            }
            
            print(f"   SF-VNN discrimination: {sf_avg_performance:.4f}")
            print(f"   Vanilla discrimination: {vanilla_avg_performance:.4f}")
            print(f"   Winner: {results[f'noise_{noise_level}']['winner']}")
        
        # Analyze robustness
        sf_wins = sum(1 for r in results.values() if r['winner'] == 'SF-VNN')
        total_tests = len(results)
        
        print(f"\\n🏆 Adversarial Robustness Summary:")
        print(f"   SF-VNN wins: {sf_wins}/{total_tests} noise levels")
        print(f"   Overall robustness winner: {'SF-VNN' if sf_wins > total_tests/2 else 'Vanilla'}")
        
        return results
    
    def experiment_3_generalization_test(self) -> Dict:
        """Test generalization across different audio types."""
        
        print("\\n🧪 EXPERIMENT 3: Generalization Across Audio Types")
        print("=" * 50)
        
        results = {}
        
        for audio_type, audio_signal in self.audio_types.items():
            print(f"\\n🎵 Testing on {audio_type} audio...")
            
            generator, sf_disc, vanilla_disc = self.create_models()
            
            # Prepare audio
            real_audio = audio_signal.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1)
            
            # Quick adaptation test
            sf_opt = torch.optim.Adam(sf_disc.parameters(), lr=2e-4)
            vanilla_opt = torch.optim.Adam(vanilla_disc.parameters(), lr=2e-4)
            criterion = nn.BCEWithLogitsLoss()
            
            sf_adaptation = []
            vanilla_adaptation = []
            
            for epoch in range(15):  # Quick test
                
                # Generate fake audio
                with torch.no_grad():
                    noise_mel = torch.randn(2, 80, 32)
                    fake_audio = generator(noise_mel)
                    if fake_audio.dim() == 2:
                        fake_audio = fake_audio.unsqueeze(1)
                    
                    min_len = min(fake_audio.size(-1), real_audio.size(-1))
                    fake_audio = fake_audio[..., :min_len]
                    real_batch = real_audio[..., :min_len]
                
                # Convert to mel
                import torchaudio.transforms as T
                mel_transform = T.MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256)
                real_mel = mel_transform(real_batch.squeeze(1)).unsqueeze(1)
                fake_mel = mel_transform(fake_audio.squeeze(1)).unsqueeze(1)
                
                min_w = min(real_mel.size(-1), fake_mel.size(-1))
                real_mel = real_mel[..., :min_w]
                fake_mel = fake_mel[..., :min_w]
                
                # Train and measure adaptation speed
                # SF-VNN
                sf_opt.zero_grad()
                sf_real_pred = sf_disc(real_mel)
                sf_fake_pred = sf_disc(fake_mel.detach())
                sf_loss = (criterion(sf_real_pred, torch.ones_like(sf_real_pred)) + 
                          criterion(sf_fake_pred, torch.zeros_like(sf_fake_pred))) / 2
                sf_loss.backward()
                sf_opt.step()
                sf_adaptation.append(sf_loss.item())
                
                # Vanilla
                vanilla_opt.zero_grad()
                vanilla_real_pred = vanilla_disc(real_mel)
                vanilla_fake_pred = vanilla_disc(fake_mel.detach())
                vanilla_loss = (criterion(vanilla_real_pred, torch.ones_like(vanilla_real_pred)) + 
                               criterion(vanilla_fake_pred, torch.zeros_like(vanilla_fake_pred))) / 2
                vanilla_loss.backward()
                vanilla_opt.step()
                vanilla_adaptation.append(vanilla_loss.item())
            
            # Analyze adaptation
            sf_final_loss = np.mean(sf_adaptation[-5:])
            vanilla_final_loss = np.mean(vanilla_adaptation[-5:])
            
            sf_adaptation_speed = sf_adaptation[0] - sf_final_loss  # How much it improved
            vanilla_adaptation_speed = vanilla_adaptation[0] - vanilla_final_loss
            
            results[audio_type] = {
                'sf_final_loss': sf_final_loss,
                'vanilla_final_loss': vanilla_final_loss,
                'sf_adaptation_speed': sf_adaptation_speed,
                'vanilla_adaptation_speed': vanilla_adaptation_speed,
                'adaptation_winner': 'SF-VNN' if sf_adaptation_speed > vanilla_adaptation_speed else 'Vanilla',
                'final_performance_winner': 'SF-VNN' if sf_final_loss < vanilla_final_loss else 'Vanilla'
            }
            
            print(f"   SF-VNN adaptation: {sf_adaptation_speed:.4f}")
            print(f"   Vanilla adaptation: {vanilla_adaptation_speed:.4f}")
            print(f"   Adaptation winner: {results[audio_type]['adaptation_winner']}")
        
        # Summary
        sf_adaptation_wins = sum(1 for r in results.values() if r['adaptation_winner'] == 'SF-VNN')
        sf_performance_wins = sum(1 for r in results.values() if r['final_performance_winner'] == 'SF-VNN')
        
        print(f"\\n🏆 Generalization Summary:")
        print(f"   SF-VNN adaptation wins: {sf_adaptation_wins}/{len(results)}")
        print(f"   SF-VNN performance wins: {sf_performance_wins}/{len(results)}")
        
        return results
    
    def run_all_experiments(self) -> Dict:
        """Run all compelling evidence experiments."""
        
        print("🚀 RUNNING ALL COMPELLING EVIDENCE EXPERIMENTS")
        print("🎯 Finding where Structure-First truly excels...")
        print("=" * 70)
        
        all_results = {}
        
        # Experiment 1: Long-term stability
        print("\\n⏰ Starting long-term stability test...")
        all_results['stability'] = self.experiment_1_long_term_stability()
        
        # Experiment 2: Adversarial robustness
        print("\\n🛡️  Starting adversarial robustness test...")
        all_results['robustness'] = self.experiment_2_adversarial_robustness()
        
        # Experiment 3: Generalization
        print("\\n🌍 Starting generalization test...")
        all_results['generalization'] = self.experiment_3_generalization_test()
        
        # Create comprehensive summary
        summary = self.create_compelling_summary(all_results)
        all_results['summary'] = summary
        
        return all_results
    
    def create_compelling_summary(self, results: Dict) -> Dict:
        """Create a compelling summary of where SF-VNN excels."""
        
        print("\\n🏆 COMPELLING EVIDENCE SUMMARY")
        print("=" * 50)
        
        strengths = []
        
        # Stability analysis
        if results['stability']['stability_winner'] == 'SF-VNN':
            ratio = results['stability']['stability_ratio']
            strengths.append(f"Training stability: {ratio:.1f}x more stable than vanilla")
            print(f"✅ SF-VNN is {ratio:.1f}x more stable in long-term training")
        
        if results['stability']['grad_stability_winner'] == 'SF-VNN':
            strengths.append("Gradient stability: More stable gradient flow")
            print("✅ SF-VNN has more stable gradient flow")
        
        # Robustness analysis
        sf_robustness_wins = sum(1 for r in results['robustness'].values() 
                                if isinstance(r, dict) and r.get('winner') == 'SF-VNN')
        total_robustness_tests = len([r for r in results['robustness'].values() if isinstance(r, dict)])
        
        if sf_robustness_wins > total_robustness_tests / 2:
            strengths.append(f"Adversarial robustness: Wins {sf_robustness_wins}/{total_robustness_tests} noise conditions")
            print(f"✅ SF-VNN is more robust to adversarial noise ({sf_robustness_wins}/{total_robustness_tests} wins)")
        
        # Generalization analysis
        sf_adaptation_wins = sum(1 for r in results['generalization'].values() 
                                if r.get('adaptation_winner') == 'SF-VNN')
        total_audio_types = len(results['generalization'])
        
        if sf_adaptation_wins > total_audio_types / 2:
            strengths.append(f"Adaptation speed: Faster adaptation to {sf_adaptation_wins}/{total_audio_types} audio types")
            print(f"✅ SF-VNN adapts faster to new audio types ({sf_adaptation_wins}/{total_audio_types} wins)")
        
        # Overall assessment
        print(f"\\n🎯 SF-VNN Key Strengths Found:")
        for i, strength in enumerate(strengths, 1):
            print(f"   {i}. {strength}")
        
        if len(strengths) >= 2:
            print("\\n🎉 COMPELLING EVIDENCE FOUND!")
            print("   SF-VNN shows clear advantages in multiple areas!")
        else:
            print("\\n🤔 Mixed results - SF-VNN shows some advantages")
        
        return {
            'strengths': strengths,
            'total_advantages': len(strengths),
            'stability_advantage': results['stability']['stability_winner'] == 'SF-VNN',
            'robustness_advantage': sf_robustness_wins > total_robustness_tests / 2,
            'adaptation_advantage': sf_adaptation_wins > total_audio_types / 2,
            'compelling_evidence': len(strengths) >= 2
        }
    
    def save_results(self, results: Dict):
        """Save all experimental results."""
        
        # Convert to JSON-serializable format
        def convert_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_types(results)
        
        with open('compelling_evidence_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\\n💾 Results saved to: compelling_evidence_results.json")


def main():
    """Run compelling evidence experiments."""
    
    config = CompellingExperimentConfig(
        stability_epochs=30,  # Reasonable for testing
        adversarial_epochs=15,
        adversarial_noise_levels=[0.0, 0.1, 0.3, 0.5]
    )
    
    experiments = CompellingEvidenceExperiments(config)
    results = experiments.run_all_experiments()
    experiments.save_results(results)
    
    print("\\n✅ All compelling evidence experiments completed!")
    print("🎵 Check the results to see where your SF-VNN truly excels!")


if __name__ == "__main__":
    main()