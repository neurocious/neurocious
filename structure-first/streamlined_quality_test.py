#!/usr/bin/env python3
"""
Streamlined Audio Quality Test for Structure-First vs Vanilla
Focus on key metrics that are most relevant for the methods paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict

# Import our components
from hifi import HiFiGANConfig, HiFiGANGenerator, AudioSFVNNDiscriminator
from empirical_comparison import VanillaCNNDiscriminator

class StreamlinedQualityEvaluator:
    """Streamlined audio quality evaluator focusing on key metrics."""
    
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        
        # Setup mel-spectrogram transform
        import torchaudio.transforms as T
        self.mel_transform = T.MelSpectrogram(
            n_mels=80, 
            n_fft=1024, 
            hop_length=256
        ).to(self.device)
        
        self.amplitude_to_db = T.AmplitudeToDB().to(self.device)
    
    def evaluate_key_metrics(self, real_audio: torch.Tensor, fake_audio: torch.Tensor) -> Dict[str, float]:
        """Compute key audio quality metrics quickly."""
        
        metrics = {}
        
        # Ensure same lengths
        min_len = min(real_audio.size(-1), fake_audio.size(-1))
        real_audio = real_audio[..., :min_len]
        fake_audio = fake_audio[..., :min_len]
        
        # 1. Mel-spectrogram L1 distance (spectral similarity)
        real_mel = self.mel_transform(real_audio.squeeze(1))
        fake_mel = self.mel_transform(fake_audio.squeeze(1))
        
        # Ensure same mel dimensions
        min_mel_w = min(real_mel.size(-1), fake_mel.size(-1))
        real_mel = real_mel[..., :min_mel_w]
        fake_mel = fake_mel[..., :min_mel_w]
        
        mel_l1 = F.l1_loss(fake_mel, real_mel).item()
        metrics['mel_l1_distance'] = mel_l1
        
        # 2. Simplified FAD using mel features
        real_mel_flat = real_mel.reshape(real_mel.size(0), -1).cpu().numpy()
        fake_mel_flat = fake_mel.reshape(fake_mel.size(0), -1).cpu().numpy()
        
        # Simple distance between means (simplified FAD)
        real_mean = np.mean(real_mel_flat, axis=0)
        fake_mean = np.mean(fake_mel_flat, axis=0)
        simplified_fad = np.linalg.norm(real_mean - fake_mean)
        metrics['simplified_fad'] = float(simplified_fad)
        
        # 3. RMS energy similarity
        real_rms = torch.sqrt(torch.mean(real_audio.squeeze(1)**2, dim=-1))
        fake_rms = torch.sqrt(torch.mean(fake_audio.squeeze(1)**2, dim=-1))
        rms_diff = F.l1_loss(fake_rms, real_rms).item()
        metrics['rms_difference'] = rms_diff
        
        # 4. Spectral centroid difference
        real_centroid = self._compute_spectral_centroid(real_audio)
        fake_centroid = self._compute_spectral_centroid(fake_audio)
        centroid_diff = abs(real_centroid - fake_centroid).item()
        metrics['spectral_centroid_diff'] = centroid_diff
        
        # 5. Simple quality score (lower = better, combines key metrics)
        quality_score = mel_l1 * 0.4 + simplified_fad * 0.001 + rms_diff * 0.3 + centroid_diff * 0.0001
        metrics['overall_quality_score'] = quality_score
        
        return metrics
    
    def _compute_spectral_centroid(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid efficiently."""
        
        # Use mel-spectrogram for efficiency
        mel_spec = self.mel_transform(audio.squeeze(1))
        
        # Mel bin centers (approximate frequencies)
        mel_freqs = torch.linspace(0, 8000, mel_spec.size(1)).to(audio.device)
        mel_freqs = mel_freqs.unsqueeze(0).unsqueeze(-1)
        
        # Weighted average
        centroid = torch.sum(mel_freqs * mel_spec, dim=1) / (torch.sum(mel_spec, dim=1) + 1e-8)
        
        return centroid.mean()


def run_streamlined_comparison():
    """Run streamlined comparison focusing on audio quality."""
    
    print("🎵 Streamlined Audio Quality Comparison")
    print("🚀 Structure-First vs Vanilla CNN Discriminators")
    print("=" * 60)
    
    device = "cpu"
    
    # Setup models
    gen_config = HiFiGANConfig()
    generator = HiFiGANGenerator(gen_config).to(device)
    
    sf_discriminator = AudioSFVNNDiscriminator(
        input_channels=1,
        vector_channels=[32, 64, 128],
        window_size=3,
        sigma=1.0,
        multiscale_analysis=True
    ).to(device)
    
    vanilla_discriminator = VanillaCNNDiscriminator({
        'channels': [32, 64, 128],
        'kernel_sizes': [(3, 9), (3, 8), (3, 8)],
        'strides': [(1, 1), (1, 2), (1, 2)]
    }).to(device)
    
    # Create test audio (harmonic content)
    batch_size = 2
    audio_length = 16384
    t = torch.linspace(0, audio_length/22050, audio_length)
    
    # Real audio: Musical chord (multiple harmonics)
    frequencies = [440, 554.37, 659.25]  # A major chord
    real_audio = torch.zeros(audio_length)
    for freq in frequencies:
        real_audio += torch.sin(2 * np.pi * freq * t)
    
    # Add realistic envelope and slight noise
    envelope = torch.exp(-t * 1.5)
    real_audio = (real_audio * envelope * 0.3 + 0.05 * torch.randn(audio_length))
    real_audio = real_audio.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    # Quality evaluator
    evaluator = StreamlinedQualityEvaluator(device=device)
    
    # Model parameters
    sf_params = sum(p.numel() for p in sf_discriminator.parameters())
    vanilla_params = sum(p.numel() for p in vanilla_discriminator.parameters())
    gen_params = sum(p.numel() for p in generator.parameters())
    
    print(f"📊 Model Parameters:")
    print(f"   Generator: {gen_params:,}")
    print(f"   SF-VNN Discriminator: {sf_params:,}")
    print(f"   Vanilla CNN Discriminator: {vanilla_params:,}")
    print(f"   Parameter Efficiency: SF-VNN uses {sf_params/vanilla_params:.3f}x parameters")
    print()
    
    # Generate audio with both approaches
    print("🎵 Generating audio samples...")
    
    with torch.no_grad():
        # Generate audio (using random mel input)
        noise_mel = torch.randn(batch_size, 80, 32).to(device)
        fake_audio = generator(noise_mel)
        
        # Ensure proper shape
        if fake_audio.dim() == 2:
            fake_audio = fake_audio.unsqueeze(1)
        
        min_len = min(fake_audio.size(-1), real_audio.size(-1))
        fake_audio_1 = fake_audio[..., :min_len]
        real_audio_batch = real_audio[..., :min_len]
        
        # Create second variant (slightly different)
        noise_mel_2 = torch.randn(batch_size, 80, 32).to(device) * 0.8
        fake_audio_2 = generator(noise_mel_2)
        if fake_audio_2.dim() == 2:
            fake_audio_2 = fake_audio_2.unsqueeze(1)
        fake_audio_2 = fake_audio_2[..., :min_len]
    
    print("📈 Evaluating audio quality...")
    print()
    
    # Evaluate both fake audio samples
    results_1 = evaluator.evaluate_key_metrics(real_audio_batch, fake_audio_1)
    results_2 = evaluator.evaluate_key_metrics(real_audio_batch, fake_audio_2)
    
    print("🎯 Audio Quality Results:")
    print("-" * 40)
    
    metrics = ['mel_l1_distance', 'simplified_fad', 'rms_difference', 'spectral_centroid_diff', 'overall_quality_score']
    
    print(f"{'Metric':<25} {'Sample 1':<12} {'Sample 2':<12} {'Better':<10}")
    print("-" * 65)
    
    for metric in metrics:
        val1 = results_1[metric]
        val2 = results_2[metric]
        better = "Sample 1" if val1 < val2 else "Sample 2"
        
        print(f"{metric:<25} {val1:<12.6f} {val2:<12.6f} {better:<10}")
    
    print()
    
    # Discriminator evaluation
    print("🔍 Discriminator Behavior Analysis:")
    print("-" * 40)
    
    with torch.no_grad():
        # Convert to mel for discriminators
        import torchaudio.transforms as T
        mel_transform = T.MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256).to(device)
        
        real_mel = mel_transform(real_audio_batch.squeeze(1)).unsqueeze(1)
        fake_mel_1 = mel_transform(fake_audio_1.squeeze(1)).unsqueeze(1)
        fake_mel_2 = mel_transform(fake_audio_2.squeeze(1)).unsqueeze(1)
        
        # Ensure consistent shapes
        min_w = min(real_mel.size(-1), fake_mel_1.size(-1), fake_mel_2.size(-1))
        real_mel = real_mel[..., :min_w]
        fake_mel_1 = fake_mel_1[..., :min_w]
        fake_mel_2 = fake_mel_2[..., :min_w]
        
        # SF-VNN predictions
        sf_real_pred = torch.sigmoid(sf_discriminator(real_mel))
        sf_fake_pred_1 = torch.sigmoid(sf_discriminator(fake_mel_1))
        sf_fake_pred_2 = torch.sigmoid(sf_discriminator(fake_mel_2))
        
        # Vanilla predictions
        vanilla_real_pred = torch.sigmoid(vanilla_discriminator(real_mel))
        vanilla_fake_pred_1 = torch.sigmoid(vanilla_discriminator(fake_mel_1))
        vanilla_fake_pred_2 = torch.sigmoid(vanilla_discriminator(fake_mel_2))
        
        print(f"SF-VNN Discriminator:")
        print(f"   Real audio prediction: {sf_real_pred.mean():.4f}")
        print(f"   Fake audio 1 prediction: {sf_fake_pred_1.mean():.4f}")
        print(f"   Fake audio 2 prediction: {sf_fake_pred_2.mean():.4f}")
        print(f"   Discrimination ability: {(sf_real_pred.mean() - sf_fake_pred_1.mean()).abs():.4f}")
        print()
        
        print(f"Vanilla CNN Discriminator:")
        print(f"   Real audio prediction: {vanilla_real_pred.mean():.4f}")
        print(f"   Fake audio 1 prediction: {vanilla_fake_pred_1.mean():.4f}")
        print(f"   Fake audio 2 prediction: {vanilla_fake_pred_2.mean():.4f}")
        print(f"   Discrimination ability: {(vanilla_real_pred.mean() - vanilla_fake_pred_1.mean()).abs():.4f}")
        print()
        
        # Quality vs Discrimination correlation
        better_quality_sample = 1 if results_1['overall_quality_score'] < results_2['overall_quality_score'] else 2
        
        sf_prefers = 1 if sf_fake_pred_1.mean() > sf_fake_pred_2.mean() else 2
        vanilla_prefers = 1 if vanilla_fake_pred_1.mean() > vanilla_fake_pred_2.mean() else 2
        
        print(f"🏆 Quality-Discrimination Alignment:")
        print(f"   Better quality sample: Sample {better_quality_sample}")
        print(f"   SF-VNN prefers: Sample {sf_prefers} {'✓' if sf_prefers == better_quality_sample else '✗'}")
        print(f"   Vanilla prefers: Sample {vanilla_prefers} {'✓' if vanilla_prefers == better_quality_sample else '✗'}")
    
    # Save results
    final_results = {
        'model_parameters': {
            'sf_vnn': sf_params,
            'vanilla': vanilla_params,
            'generator': gen_params
        },
        'audio_quality_sample_1': results_1,
        'audio_quality_sample_2': results_2,
        'discriminator_predictions': {
            'sf_vnn': {
                'real': float(sf_real_pred.mean()),
                'fake_1': float(sf_fake_pred_1.mean()),
                'fake_2': float(sf_fake_pred_2.mean())
            },
            'vanilla': {
                'real': float(vanilla_real_pred.mean()),
                'fake_1': float(vanilla_fake_pred_1.mean()),
                'fake_2': float(vanilla_fake_pred_2.mean())
            }
        }
    }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        else:
            return obj
    
    final_results_serializable = convert_types(final_results)
    
    with open('streamlined_quality_results.json', 'w') as f:
        json.dump(final_results_serializable, f, indent=2)
    
    print()
    print("💾 Results saved to: streamlined_quality_results.json")
    print("✅ Streamlined audio quality evaluation completed!")
    
    return final_results


if __name__ == "__main__":
    run_streamlined_comparison()