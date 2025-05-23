#!/usr/bin/env python3
"""
Comprehensive Audio Quality Metrics for Structure-First vs Vanilla Discriminator
Implements state-of-the-art audio quality evaluation metrics for GAN comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
from scipy import linalg, signal
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class AudioQualityEvaluator:
    """Comprehensive audio quality evaluation suite."""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 80,
                 device: str = "cpu"):
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = torch.device(device)
        
        # Initialize transforms
        self._setup_transforms()
        
        # Precomputed statistics for FAD (would normally load from dataset)
        self._setup_reference_stats()
        
    def _setup_transforms(self):
        """Initialize audio transformation layers."""
        
        # STFT transform
        self.stft_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0
        ).to(self.device)
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)
        
        # MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels
            }
        ).to(self.device)
        
        # Amplitude to dB conversion
        self.amplitude_to_db = T.AmplitudeToDB().to(self.device)
        
    def _setup_reference_stats(self):
        """Setup reference statistics for FAD computation."""
        # In practice, these would be computed from your real dataset
        # For now, using synthetic reference stats
        self.reference_fad_stats = {
            'mu': np.random.randn(512),  # Feature mean
            'sigma': np.eye(512) + np.random.randn(512, 512) * 0.1  # Covariance
        }
        
    def frechet_audio_distance(self, 
                             real_audio: torch.Tensor, 
                             fake_audio: torch.Tensor,
                             use_mel_features: bool = True) -> float:
        """
        Compute Fréchet Audio Distance (FAD).
        Similar to FID but for audio using mel-spectrogram features.
        """
        
        def extract_features(audio_batch):
            """Extract features for FAD computation."""
            if use_mel_features:
                # Use mel-spectrogram features
                mel_specs = self.mel_transform(audio_batch.squeeze(1))
                mel_db = self.amplitude_to_db(mel_specs)
                # Flatten to feature vectors
                features = mel_db.reshape(mel_db.size(0), -1)
            else:
                # Use raw STFT features
                stft_specs = self.stft_transform(audio_batch.squeeze(1))
                features = stft_specs.reshape(stft_specs.size(0), -1)
            
            return features.cpu().numpy()
        
        # Extract features
        real_features = extract_features(real_audio)
        fake_features = extract_features(fake_audio)
        
        # Ensure same feature dimensions
        min_features = min(real_features.shape[1], fake_features.shape[1])
        real_features = real_features[:, :min_features]
        fake_features = fake_features[:, :min_features]
        
        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Fréchet distance computation
        diff = mu_real - mu_fake
        
        # Handle numerical stability
        try:
            covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
        except:
            # Fallback for numerical issues
            covmean = np.zeros_like(sigma_real)
        
        fad = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2*covmean)
        
        return float(fad)
    
    def spectral_distance_metrics(self, 
                                real_audio: torch.Tensor, 
                                fake_audio: torch.Tensor) -> Dict[str, float]:
        """Compute various spectral distance metrics."""
        
        metrics = {}
        
        # Ensure same lengths
        min_len = min(real_audio.size(-1), fake_audio.size(-1))
        real_audio = real_audio[..., :min_len]
        fake_audio = fake_audio[..., :min_len]
        
        # 1. Mel-spectrogram L1 distance
        real_mel = self.mel_transform(real_audio.squeeze(1))
        fake_mel = self.mel_transform(fake_audio.squeeze(1))
        mel_l1 = F.l1_loss(fake_mel, real_mel).item()
        metrics['mel_l1_distance'] = mel_l1
        
        # 2. STFT magnitude L1 distance
        real_stft = self.stft_transform(real_audio.squeeze(1))
        fake_stft = self.stft_transform(fake_audio.squeeze(1))
        stft_l1 = F.l1_loss(fake_stft, real_stft).item()
        metrics['stft_l1_distance'] = stft_l1
        
        # 3. MFCC distance
        real_mfcc = self.mfcc_transform(real_audio.squeeze(1))
        fake_mfcc = self.mfcc_transform(fake_audio.squeeze(1))
        mfcc_l1 = F.l1_loss(fake_mfcc, real_mfcc).item()
        metrics['mfcc_l1_distance'] = mfcc_l1
        
        # 4. Spectral centroid difference
        real_centroids = self._compute_spectral_centroids(real_audio)
        fake_centroids = self._compute_spectral_centroids(fake_audio)
        centroid_diff = abs(real_centroids.mean() - fake_centroids.mean()).item()
        metrics['spectral_centroid_diff'] = centroid_diff
        
        # 5. Spectral rolloff difference  
        real_rolloff = self._compute_spectral_rolloff(real_audio)
        fake_rolloff = self._compute_spectral_rolloff(fake_audio)
        rolloff_diff = abs(real_rolloff.mean() - fake_rolloff.mean()).item()
        metrics['spectral_rolloff_diff'] = rolloff_diff
        
        return metrics
    
    def _compute_spectral_centroids(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroids."""
        spectrogram = self.stft_transform(audio.squeeze(1))
        
        # Frequency bins
        freqs = torch.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2 + 1]
        freqs = freqs.to(audio.device).unsqueeze(0).unsqueeze(-1)
        
        # Weighted average of frequencies
        centroids = torch.sum(freqs * spectrogram, dim=1) / (torch.sum(spectrogram, dim=1) + 1e-8)
        
        return centroids.mean(dim=-1)  # Average over time
    
    def _compute_spectral_rolloff(self, audio: torch.Tensor, rolloff_percent: float = 0.85) -> torch.Tensor:
        """Compute spectral rolloff."""
        spectrogram = self.stft_transform(audio.squeeze(1))
        
        # Cumulative sum along frequency axis
        cumsum_spec = torch.cumsum(spectrogram, dim=1)
        total_energy = cumsum_spec[:, -1:, :]
        
        # Find rolloff point
        rolloff_threshold = rolloff_percent * total_energy
        rolloff_indices = torch.argmax((cumsum_spec >= rolloff_threshold).float(), dim=1)
        
        # Convert to frequency
        freqs = torch.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2 + 1]
        rolloff_freqs = freqs[rolloff_indices.flatten()].reshape(rolloff_indices.shape)
        
        return rolloff_freqs.mean(dim=-1)  # Average over time
    
    def perceptual_quality_metrics(self, 
                                 real_audio: torch.Tensor, 
                                 fake_audio: torch.Tensor) -> Dict[str, float]:
        """Compute perceptual audio quality metrics."""
        
        metrics = {}
        
        # Ensure same lengths
        min_len = min(real_audio.size(-1), fake_audio.size(-1))
        real_audio = real_audio[..., :min_len]
        fake_audio = fake_audio[..., :min_len]
        
        # Convert to numpy for librosa processing
        real_np = real_audio.squeeze().cpu().numpy()
        fake_np = fake_audio.squeeze().cpu().numpy()
        
        # Handle batch processing
        if real_np.ndim == 2:
            real_np = real_np[0]  # Take first sample
            fake_np = fake_np[0]
        
        # 1. Zero Crossing Rate similarity
        real_zcr = librosa.feature.zero_crossing_rate(real_np)[0]
        fake_zcr = librosa.feature.zero_crossing_rate(fake_np)[0]
        zcr_correlation = pearsonr(real_zcr, fake_zcr[:len(real_zcr)])[0]
        metrics['zcr_correlation'] = float(zcr_correlation) if not np.isnan(zcr_correlation) else 0.0
        
        # 2. RMS Energy similarity
        real_rms = librosa.feature.rms(y=real_np)[0]
        fake_rms = librosa.feature.rms(y=fake_np)[0]
        rms_correlation = pearsonr(real_rms, fake_rms[:len(real_rms)])[0]
        metrics['rms_correlation'] = float(rms_correlation) if not np.isnan(rms_correlation) else 0.0
        
        # 3. Tempo similarity (if long enough)
        try:
            if len(real_np) > self.sample_rate * 2:  # At least 2 seconds
                real_tempo = librosa.beat.tempo(y=real_np, sr=self.sample_rate)[0]
                fake_tempo = librosa.beat.tempo(y=fake_np, sr=self.sample_rate)[0]
                tempo_diff = abs(real_tempo - fake_tempo) / max(real_tempo, fake_tempo)
                metrics['tempo_difference'] = float(tempo_diff)
            else:
                metrics['tempo_difference'] = 0.0
        except:
            metrics['tempo_difference'] = 0.0
        
        # 4. Harmonic-Percussive separation similarity
        try:
            real_harmonic, real_percussive = librosa.effects.hpss(real_np)
            fake_harmonic, fake_percussive = librosa.effects.hpss(fake_np)
            
            # Harmonic component correlation
            harm_correlation = pearsonr(real_harmonic, fake_harmonic[:len(real_harmonic)])[0]
            metrics['harmonic_correlation'] = float(harm_correlation) if not np.isnan(harm_correlation) else 0.0
            
            # Percussive component correlation
            perc_correlation = pearsonr(real_percussive, fake_percussive[:len(real_percussive)])[0]
            metrics['percussive_correlation'] = float(perc_correlation) if not np.isnan(perc_correlation) else 0.0
            
        except:
            metrics['harmonic_correlation'] = 0.0
            metrics['percussive_correlation'] = 0.0
        
        return metrics
    
    def mos_prediction(self, 
                      real_audio: torch.Tensor, 
                      fake_audio: torch.Tensor) -> Dict[str, float]:
        """
        Predict Mean Opinion Score (MOS) using objective metrics.
        This is a simplified model - in practice you'd use a trained neural network.
        """
        
        # Get spectral and perceptual metrics
        spectral_metrics = self.spectral_distance_metrics(real_audio, fake_audio)
        perceptual_metrics = self.perceptual_quality_metrics(real_audio, fake_audio)
        
        # Simple heuristic MOS prediction (you'd train this properly)
        # Lower distance = higher quality
        mel_quality = max(0, 5 - spectral_metrics['mel_l1_distance'] * 10)
        stft_quality = max(0, 5 - spectral_metrics['stft_l1_distance'])
        
        # Higher correlation = higher quality
        correlation_score = (
            perceptual_metrics.get('zcr_correlation', 0) +
            perceptual_metrics.get('rms_correlation', 0) +
            perceptual_metrics.get('harmonic_correlation', 0) +
            perceptual_metrics.get('percussive_correlation', 0)
        ) / 4 * 5  # Scale to 0-5
        
        # Weighted average (you'd optimize these weights)
        predicted_mos = (mel_quality * 0.3 + 
                        stft_quality * 0.3 + 
                        correlation_score * 0.4)
        
        # Clip to valid MOS range
        predicted_mos = np.clip(predicted_mos, 1.0, 5.0)
        
        return {
            'predicted_mos': float(predicted_mos),
            'mel_quality_component': float(mel_quality),
            'stft_quality_component': float(stft_quality),
            'correlation_component': float(correlation_score)
        }
    
    def comprehensive_evaluation(self, 
                               real_audio: torch.Tensor, 
                               fake_audio: torch.Tensor) -> Dict[str, float]:
        """Run comprehensive audio quality evaluation."""
        
        results = {}
        
        print("🎵 Computing Fréchet Audio Distance...")
        fad_mel = self.frechet_audio_distance(real_audio, fake_audio, use_mel_features=True)
        fad_stft = self.frechet_audio_distance(real_audio, fake_audio, use_mel_features=False)
        results['fad_mel'] = fad_mel
        results['fad_stft'] = fad_stft
        
        print("📊 Computing spectral distance metrics...")
        spectral_metrics = self.spectral_distance_metrics(real_audio, fake_audio)
        results.update(spectral_metrics)
        
        print("🎧 Computing perceptual quality metrics...")
        perceptual_metrics = self.perceptual_quality_metrics(real_audio, fake_audio)
        results.update(perceptual_metrics)
        
        print("🏅 Predicting MOS score...")
        mos_metrics = self.mos_prediction(real_audio, fake_audio)
        results.update(mos_metrics)
        
        return results


def demonstrate_quality_metrics():
    """Demonstrate the audio quality metrics on synthetic data."""
    
    print("🧪 Audio Quality Metrics Demonstration")
    print("=" * 50)
    
    # Create evaluator
    evaluator = AudioQualityEvaluator(device="cpu")
    
    # Create synthetic audio data
    batch_size = 2
    audio_length = 16384  # ~0.74 seconds at 22050 Hz
    
    # Real audio (sine wave with some noise)
    t = torch.linspace(0, audio_length/22050, audio_length)
    real_audio = torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn(audio_length)
    real_audio = real_audio.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Fake audio 1: Similar to real (high quality)
    fake_audio_good = real_audio + 0.05 * torch.randn_like(real_audio)
    
    # Fake audio 2: Very different (low quality)
    fake_audio_bad = torch.randn_like(real_audio) * 0.5
    
    print("\\n🎯 Evaluating HIGH QUALITY fake audio...")
    print("-" * 40)
    good_results = evaluator.comprehensive_evaluation(real_audio, fake_audio_good)
    
    print("\\n💀 Evaluating LOW QUALITY fake audio...")
    print("-" * 40)
    bad_results = evaluator.comprehensive_evaluation(real_audio, fake_audio_bad)
    
    print("\\n📈 Quality Comparison Summary")
    print("=" * 40)
    
    key_metrics = ['fad_mel', 'mel_l1_distance', 'predicted_mos', 'zcr_correlation']
    
    for metric in key_metrics:
        if metric in good_results and metric in bad_results:
            good_val = good_results[metric]
            bad_val = bad_results[metric]
            
            if 'correlation' in metric or 'mos' in metric:
                better = "HIGH" if good_val > bad_val else "LOW"
                print(f"{metric:20s}: HIGH={good_val:.4f}, LOW={bad_val:.4f} → {better} quality wins")
            else:
                better = "HIGH" if good_val < bad_val else "LOW"
                print(f"{metric:20s}: HIGH={good_val:.4f}, LOW={bad_val:.4f} → {better} quality wins")
    
    print("\\n✅ Audio quality metrics demonstration completed!")
    
    return good_results, bad_results


if __name__ == "__main__":
    # Run demonstration
    demonstrate_quality_metrics()