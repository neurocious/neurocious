#!/usr/bin/env python3
"""
Empirical Comparison: Structure-First vs Vanilla Discriminator

This script implements a rigorous experimental setup to compare:
1. Structure-First Vector Neuron Network (SF-VNN) Discriminator  
2. Vanilla CNN Discriminator (Baseline)

For a methods paper demonstrating the effectiveness of structure-first approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import our components
from hifi import (
    HiFiGANConfig, HiFiGANGenerator, AudioSFVNNDiscriminator, 
    EnhancedAudioStructuralLoss, HiFiGANSFVNNDataset,
    AudioQualityMetrics, AudioDatasetBuilder
)


@dataclass
class ExperimentConfig:
    """Configuration for empirical comparison experiments."""
    
    # Experiment metadata
    experiment_name: str = "sf_vnn_vs_vanilla_comparison"
    description: str = "Empirical comparison of structure-first vs vanilla discriminators"
    random_seed: int = 42
    
    # Model architecture
    generator_config: Dict = field(default_factory=lambda: {
        'sample_rate': 22050,
        'n_mels': 80,
        'upsample_rates': [8, 8, 2, 2],
        'upsample_kernel_sizes': [16, 16, 4, 4],
        'resblock_kernel_sizes': [3, 7, 11],
        'upsample_initial_channel': 512
    })
    
    # Training parameters
    training_config: Dict = field(default_factory=lambda: {
        'batch_size': 16,
        'learning_rate_g': 2e-4,
        'learning_rate_d': 2e-4,
        'beta1': 0.8,
        'beta2': 0.99,
        'num_epochs': 5,
        'segment_length': 8192,
        'lambda_mel': 45.0
    })
    
    # SF-VNN specific parameters
    sfvnn_config: Dict = field(default_factory=lambda: {
        'vector_channels': [32, 64, 128, 256],
        'window_size': 5,
        'sigma': 1.0,
        'multiscale_analysis': True,
        'lambda_structural': 1.0,
        'lambda_feature_matching': 2.0
    })
    
    # Baseline CNN discriminator parameters
    vanilla_config: Dict = field(default_factory=lambda: {
        'channels': [32, 64, 128, 256, 512],
        'kernel_sizes': [(3, 9), (3, 8), (3, 8), (3, 6), (3, 4)],
        'strides': [(1, 1), (1, 2), (1, 2), (1, 2), (1, 2)],
        'use_spectral_norm': False,
        'use_gradient_penalty': False
    })
    
    # Evaluation parameters
    evaluation_config: Dict = field(default_factory=lambda: {
        'eval_every_n_epochs': 5,
        'num_eval_samples': 100,
        'compute_fad': True,
        'compute_is_score': True,
        'compute_structural_metrics': True,
        'save_generated_samples': True
    })
    
    # Statistical testing
    statistical_config: Dict = field(default_factory=lambda: {
        'num_runs': 2,  # Number of independent runs for statistical significance
        'confidence_level': 0.95,
        'use_paired_tests': True,
        'bonferroni_correction': True
    })


class VanillaCNNDiscriminator(nn.Module):
    """
    Vanilla CNN discriminator baseline for comparison.
    
    This is a standard convolutional discriminator commonly used in audio GANs,
    similar to those in HiFi-GAN, MelGAN, etc.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        channels = config['channels']
        kernel_sizes = config['kernel_sizes']
        strides = config['strides']
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # Input layer
        in_channels = 1  # Mono spectrogram
        
        for i, (out_channels, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            # Convolutional layer
            conv = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size, stride, 
                padding=(kernel_size[0]//2, kernel_size[1]//2)
            )
            
            # Optional spectral normalization
            if config.get('use_spectral_norm', False):
                conv = nn.utils.spectral_norm(conv)
            
            # Layer with activation and normalization
            if i == 0:
                # First layer: no normalization
                layer = nn.Sequential(
                    conv,
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                # Other layers: with batch normalization
                layer = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            
            self.conv_layers.append(layer)
            in_channels = out_channels
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights following standard GAN practices."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vanilla CNN discriminator.
        
        Args:
            x: Input spectrogram [B, 1, F, T]
            
        Returns:
            Logits [B, 1] (not probabilities - raw logits for loss computation)
        """
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        logits = self.classifier(x)
        
        return logits
    
    def compute_feature_loss(self, real_spectrograms: torch.Tensor, 
                           fake_spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Compute feature matching loss using intermediate features.
        
        This provides a fairer comparison with SF-VNN's feature matching.
        """
        real_features = []
        fake_features = []
        
        # Extract features from intermediate layers
        real_x = real_spectrograms
        fake_x = fake_spectrograms
        
        for layer in self.conv_layers[:-1]:  # Exclude last layer
            real_x = layer(real_x)
            fake_x = layer(fake_x)
            
            # Store flattened features for comparison
            real_features.append(real_x.view(real_x.size(0), -1))
            fake_features.append(fake_x.view(fake_x.size(0), -1))
        
        # Compute L1 loss between features
        feature_loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            feature_loss += F.l1_loss(fake_feat, real_feat)
        
        return feature_loss / len(real_features)


class VanillaDiscriminatorLoss(nn.Module):
    """Loss function for vanilla CNN discriminator."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Standard GAN losses
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Mel-spectrogram transform for reconstruction loss
        from hifi import AudioSpectrogramProcessor
        self.audio_processor = AudioSpectrogramProcessor(
            sample_rate=config.generator_config['sample_rate'],
            n_mels=config.generator_config['n_mels']
        )
    
    def generator_loss(self, real_audio: torch.Tensor, fake_audio: torch.Tensor,
                      real_mel: torch.Tensor, fake_mel: torch.Tensor,
                      discriminator: VanillaCNNDiscriminator) -> Dict[str, torch.Tensor]:
        """Compute generator loss for vanilla discriminator."""
        
        # 1. Mel reconstruction loss
        mel_loss = self.l1_loss(fake_mel, real_mel)
        
        # 2. Adversarial loss
        fake_logits = discriminator(fake_mel)
        adv_loss = self.adversarial_loss(fake_logits, torch.ones_like(fake_logits))
        
        # 3. Feature matching loss (for fair comparison)
        fm_loss = discriminator.compute_feature_loss(real_mel, fake_mel)
        
        # Combine losses
        total_loss = (
            self.config.training_config['lambda_mel'] * mel_loss +
            adv_loss +
            2.0 * fm_loss  # Same weight as SF-VNN
        )
        
        return {
            'total': total_loss,
            'mel': mel_loss,
            'adversarial': adv_loss,
            'feature_matching': fm_loss
        }
    
    def discriminator_loss(self, real_mel: torch.Tensor, fake_mel: torch.Tensor,
                          discriminator: VanillaCNNDiscriminator) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss."""
        
        # Real samples
        real_logits = discriminator(real_mel)
        real_loss = self.adversarial_loss(real_logits, torch.ones_like(real_logits))
        
        # Fake samples (detached)
        fake_logits = discriminator(fake_mel.detach())
        fake_loss = self.adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
        
        # Total loss
        total_loss = (real_loss + fake_loss) / 2
        
        # Accuracy metrics
        real_acc = (torch.sigmoid(real_logits) > 0.5).float().mean()
        fake_acc = (torch.sigmoid(fake_logits) < 0.5).float().mean()
        
        return {
            'total': total_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'real_accuracy': real_acc,
            'fake_accuracy': fake_acc,
            'discriminator_accuracy': (real_acc + fake_acc) / 2
        }


class DiscriminatorTrainer:
    """Trainer for a single discriminator type (vanilla or SF-VNN)."""
    
    def __init__(self, 
                 discriminator: nn.Module,
                 loss_fn: nn.Module,
                 config: ExperimentConfig,
                 dataset: HiFiGANSFVNNDataset,
                 val_dataset: HiFiGANSFVNNDataset,
                 device: torch.device,
                 discriminator_type: str):
        
        self.discriminator = discriminator.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.discriminator_type = discriminator_type
        
        # Create generator (same for both discriminators for fair comparison)
        hifi_config = HiFiGANConfig()
        # Update with relevant parameters
        for key, value in config.generator_config.items():
            if hasattr(hifi_config, key):
                setattr(hifi_config, key, value)
        for key, value in config.training_config.items():
            if hasattr(hifi_config, key):
                setattr(hifi_config, key, value)
        self.generator = HiFiGANGenerator(hifi_config).to(device)
        
        # Optimizers
        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.training_config['learning_rate_g'],
            betas=(config.training_config['beta1'], config.training_config['beta2'])
        )
        
        self.optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.training_config['learning_rate_d'],
            betas=(config.training_config['beta1'], config.training_config['beta2'])
        )
        
        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.training_config['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.training_config['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Metrics
        self.metrics = AudioQualityMetrics(config.generator_config['sample_rate'])
        
        # Training state
        self.epoch = 0
        self.best_fad = float('inf')
        self.training_history = defaultdict(list)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = defaultdict(list)
        
        for batch_idx, (real_audio, real_mel) in enumerate(self.train_loader):
            real_audio = real_audio.to(self.device)
            real_mel = real_mel.to(self.device)
            
            # Generate fake audio - squeeze to get correct shape for generator
            # real_mel is [B, 1, F, T] but generator expects [B, F, T]
            real_mel_for_gen = real_mel.squeeze(1)  # Remove channel dimension
            fake_audio = self.generator(real_mel_for_gen)
            fake_mel = self._audio_to_mel(fake_audio)
            
            # Train Discriminator
            self.optim_d.zero_grad()
            d_losses = self.loss_fn.discriminator_loss(real_mel, fake_mel, self.discriminator)
            d_losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
            self.optim_d.step()
            
            # Train Generator  
            self.optim_g.zero_grad()
            g_losses = self.loss_fn.generator_loss(real_audio, fake_audio, real_mel, fake_mel, self.discriminator)
            g_losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10.0)
            self.optim_g.step()
            
            # Record metrics
            for key, value in {**g_losses, **d_losses}.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key].append(value.item())
        
        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance."""
        
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = defaultdict(list)
        real_samples = []
        fake_samples = []
        
        with torch.no_grad():
            for batch_idx, (real_audio, real_mel) in enumerate(self.val_loader):
                if batch_idx >= 10:  # Limit for faster evaluation
                    break
                
                real_audio = real_audio.to(self.device)
                real_mel = real_mel.to(self.device)
                
                # Generate samples
                real_mel_for_gen = real_mel.squeeze(1)  # Remove channel dimension
                fake_audio = self.generator(real_mel_for_gen)
                fake_mel = self._audio_to_mel(fake_audio)
                
                # Compute losses
                g_losses = self.loss_fn.generator_loss(real_audio, fake_audio, real_mel, fake_mel, self.discriminator)
                d_losses = self.loss_fn.discriminator_loss(real_mel, fake_mel, self.discriminator)
                
                for key, value in {**g_losses, **d_losses}.items():
                    if isinstance(value, torch.Tensor):
                        val_metrics[key].append(value.item())
                
                # Collect samples for advanced metrics
                if len(real_samples) < 5:
                    real_samples.append(fake_mel.cpu())
                    fake_samples.append(real_mel.cpu())
        
        # Compute advanced metrics
        if real_samples and fake_samples:
            real_batch = torch.cat(real_samples, dim=0)
            fake_batch = torch.cat(fake_samples, dim=0)
            
            # FAD score
            fad_score = self.metrics.compute_frechet_audio_distance(real_batch, fake_batch)
            val_metrics['fad_score'] = fad_score
            
            # Structural metrics (only for SF-VNN)
            if hasattr(self.discriminator, 'compute_structural_distance'):
                struct_metrics = self.metrics.compute_structural_metrics(
                    real_batch.to(self.device), fake_batch.to(self.device), self.discriminator
                )
                val_metrics.update(struct_metrics)
        
        return {key: np.mean(values) if isinstance(values, list) else values 
                for key, values in val_metrics.items()}
    
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel-spectrogram."""
        # Simplified conversion for evaluation - create basic mel-spectrogram
        if hasattr(self.loss_fn, 'audio_processor') and self.loss_fn.audio_processor is not None:
            processor = self.loss_fn.audio_processor
            mel = processor.audio_to_mel_spectrogram(audio.squeeze(1))
            return processor._normalize_mel(mel.unsqueeze(1)) if hasattr(processor, '_normalize_mel') else mel
        else:
            # Fallback: create simple mel-spectrogram using torch
            import torchaudio.transforms as T
            mel_transform = T.MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256).to(audio.device)
            mel = mel_transform(audio.squeeze(1))
            return mel.unsqueeze(1)
    
    def train(self) -> Dict[str, List[float]]:
        """Complete training loop."""
        
        print(f"🎵 Training {self.discriminator_type} discriminator")
        print(f"   Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"   Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        for epoch in range(self.config.training_config['num_epochs']):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            if epoch % self.config.evaluation_config['eval_every_n_epochs'] == 0:
                val_metrics = self.evaluate()
                
                # Record history
                for key, value in train_metrics.items():
                    self.training_history[f'train_{key}'].append(value)
                for key, value in val_metrics.items():
                    self.training_history[f'val_{key}'].append(value)
                
                # Track best model
                if 'fad_score' in val_metrics and val_metrics['fad_score'] < self.best_fad:
                    self.best_fad = val_metrics['fad_score']
                
                print(f"Epoch {epoch}: Val FAD: {val_metrics.get('fad_score', 0):.4f}, "
                      f"G_loss: {train_metrics['total']:.4f}")
        
        return dict(self.training_history)


class EmpiricalComparison:
    """Main class for conducting empirical comparison experiments."""
    
    def __init__(self, config: ExperimentConfig, results_dir: str = "comparison_results"):
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)
        
        print(f"🧪 Empirical Comparison: SF-VNN vs Vanilla Discriminator")
        print(f"   Device: {self.device}")
        print(f"   Results dir: {self.results_dir}")
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def create_discriminators(self) -> Tuple[nn.Module, nn.Module]:
        """Create SF-VNN and vanilla discriminators."""
        
        # SF-VNN discriminator
        sfvnn_disc = AudioSFVNNDiscriminator(
            input_channels=1,
            vector_channels=self.config.sfvnn_config['vector_channels'],
            window_size=self.config.sfvnn_config['window_size'],
            sigma=self.config.sfvnn_config['sigma'],
            multiscale_analysis=self.config.sfvnn_config['multiscale_analysis']
        )
        
        # Vanilla CNN discriminator
        vanilla_disc = VanillaCNNDiscriminator(self.config.vanilla_config)
        
        return sfvnn_disc, vanilla_disc
    
    def create_loss_functions(self) -> Tuple[nn.Module, nn.Module]:
        """Create loss functions for both discriminators."""
        
        # SF-VNN loss (enhanced structural loss)
        hifi_config = HiFiGANConfig()
        for key, value in self.config.generator_config.items():
            if hasattr(hifi_config, key):
                setattr(hifi_config, key, value)
        for key, value in self.config.training_config.items():
            if hasattr(hifi_config, key):
                setattr(hifi_config, key, value)
        sfvnn_loss = EnhancedAudioStructuralLoss(hifi_config)
        
        # Vanilla loss
        vanilla_loss = VanillaDiscriminatorLoss(self.config)
        
        return sfvnn_loss, vanilla_loss
    
    def run_single_comparison(self, train_dataset: HiFiGANSFVNNDataset, 
                            val_dataset: HiFiGANSFVNNDataset, 
                            run_id: int = 0) -> Dict[str, Any]:
        """Run a single comparison between SF-VNN and vanilla discriminators."""
        
        print(f"\n🔬 Running comparison {run_id + 1}")
        
        # Create discriminators and loss functions
        sfvnn_disc, vanilla_disc = self.create_discriminators()
        sfvnn_loss, vanilla_loss = self.create_loss_functions()
        
        # Create trainers
        sfvnn_trainer = DiscriminatorTrainer(
            sfvnn_disc, sfvnn_loss, self.config, train_dataset, val_dataset, 
            self.device, "SF-VNN"
        )
        
        vanilla_trainer = DiscriminatorTrainer(
            vanilla_disc, vanilla_loss, self.config, train_dataset, val_dataset,
            self.device, "Vanilla"
        )
        
        # Train both models
        print("   Training SF-VNN discriminator...")
        sfvnn_history = sfvnn_trainer.train()
        
        print("   Training Vanilla discriminator...")
        vanilla_history = vanilla_trainer.train()
        
        # Final evaluation
        print("   Final evaluation...")
        sfvnn_final = sfvnn_trainer.evaluate()
        vanilla_final = vanilla_trainer.evaluate()
        
        return {
            'run_id': run_id,
            'sfvnn': {
                'history': sfvnn_history,
                'final_metrics': sfvnn_final,
                'best_fad': sfvnn_trainer.best_fad,
                'num_parameters': sum(p.numel() for p in sfvnn_disc.parameters())
            },
            'vanilla': {
                'history': vanilla_history,
                'final_metrics': vanilla_final,
                'best_fad': vanilla_trainer.best_fad,
                'num_parameters': sum(p.numel() for p in vanilla_disc.parameters())
            }
        }
    
    def run_multiple_comparisons(self, train_dataset: HiFiGANSFVNNDataset,
                               val_dataset: HiFiGANSFVNNDataset) -> Dict[str, Any]:
        """Run multiple independent comparisons for statistical significance."""
        
        num_runs = self.config.statistical_config['num_runs']
        all_results = []
        
        print(f"🧪 Running {num_runs} independent comparisons")
        
        for run_id in range(num_runs):
            # Set different random seed for each run
            self._set_random_seeds(self.config.random_seed + run_id * 100)
            
            result = self.run_single_comparison(train_dataset, val_dataset, run_id)
            all_results.append(result)
            
            # Save intermediate results
            self._save_intermediate_results(all_results, run_id)
        
        # Compute statistical analysis
        statistical_analysis = self._compute_statistical_analysis(all_results)
        
        return {
            'config': self.config,
            'individual_runs': all_results,
            'statistical_analysis': statistical_analysis,
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'num_runs': num_runs
            }
        }
    
    def _compute_statistical_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute statistical significance tests and summary statistics."""
        
        # Extract key metrics for comparison
        sfvnn_fads = [r['sfvnn']['best_fad'] for r in results]
        vanilla_fads = [r['vanilla']['best_fad'] for r in results]
        
        sfvnn_final_losses = [r['sfvnn']['final_metrics'].get('total', 0) for r in results]
        vanilla_final_losses = [r['vanilla']['final_metrics'].get('total', 0) for r in results]
        
        # Descriptive statistics
        stats_summary = {
            'fad_scores': {
                'sfvnn': {
                    'mean': np.mean(sfvnn_fads),
                    'std': np.std(sfvnn_fads),
                    'median': np.median(sfvnn_fads),
                    'min': np.min(sfvnn_fads),
                    'max': np.max(sfvnn_fads)
                },
                'vanilla': {
                    'mean': np.mean(vanilla_fads),
                    'std': np.std(vanilla_fads),
                    'median': np.median(vanilla_fads),
                    'min': np.min(vanilla_fads),
                    'max': np.max(vanilla_fads)
                }
            },
            'final_losses': {
                'sfvnn': {
                    'mean': np.mean(sfvnn_final_losses),
                    'std': np.std(sfvnn_final_losses)
                },
                'vanilla': {
                    'mean': np.mean(vanilla_final_losses),
                    'std': np.std(vanilla_final_losses)
                }
            }
        }
        
        # Statistical significance tests
        significance_tests = {}
        
        # Paired t-test for FAD scores
        if len(sfvnn_fads) > 1:
            fad_tstat, fad_pvalue = stats.ttest_rel(sfvnn_fads, vanilla_fads)
            significance_tests['fad_paired_ttest'] = {
                'statistic': fad_tstat,
                'p_value': fad_pvalue,
                'significant': fad_pvalue < (1 - self.config.statistical_config['confidence_level']),
                'interpretation': 'SF-VNN significantly better' if fad_tstat < 0 and fad_pvalue < 0.05 else 'No significant difference'
            }
            
            # Wilcoxon signed-rank test (non-parametric)
            wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(sfvnn_fads, vanilla_fads)
            significance_tests['fad_wilcoxon'] = {
                'statistic': wilcoxon_stat,
                'p_value': wilcoxon_pvalue,
                'significant': wilcoxon_pvalue < 0.05
            }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(sfvnn_fads) + np.var(vanilla_fads)) / 2)
        cohens_d = (np.mean(vanilla_fads) - np.mean(sfvnn_fads)) / pooled_std if pooled_std > 0 else 0
        
        significance_tests['effect_size'] = {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_effect_size(cohens_d)
        }
        
        return {
            'descriptive_statistics': stats_summary,
            'significance_tests': significance_tests,
            'summary': self._generate_statistical_summary(stats_summary, significance_tests)
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_statistical_summary(self, stats_summary: Dict, significance_tests: Dict) -> str:
        """Generate human-readable summary of statistical results."""
        
        sfvnn_mean_fad = stats_summary['fad_scores']['sfvnn']['mean']
        vanilla_mean_fad = stats_summary['fad_scores']['vanilla']['mean']
        
        improvement = vanilla_mean_fad - sfvnn_mean_fad
        improvement_pct = (improvement / vanilla_mean_fad) * 100 if vanilla_mean_fad > 0 else 0
        
        summary = f"""
Statistical Analysis Summary:
• SF-VNN mean FAD: {sfvnn_mean_fad:.4f} ± {stats_summary['fad_scores']['sfvnn']['std']:.4f}
• Vanilla mean FAD: {vanilla_mean_fad:.4f} ± {stats_summary['fad_scores']['vanilla']['std']:.4f}
• Improvement: {improvement:.4f} ({improvement_pct:.1f}%)
• Effect size: {significance_tests.get('effect_size', {}).get('interpretation', 'unknown')}
• Statistical significance: {significance_tests.get('fad_paired_ttest', {}).get('interpretation', 'unknown')}
        """.strip()
        
        return summary
    
    def _save_intermediate_results(self, results: List[Dict], run_id: int):
        """Save intermediate results after each run."""
        filename = self.results_dir / f"intermediate_results_run_{run_id:03d}.json"
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    serializable_result[key] = {
                        k: float(v) if isinstance(v, (torch.Tensor, np.number)) else v
                        for k, v in value.items() if not isinstance(v, dict)
                    }
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def generate_paper_ready_results(self, results: Dict[str, Any]) -> str:
        """Generate publication-ready results summary."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"paper_results_{timestamp}.txt"
        
        stats = results['statistical_analysis']
        
        # Generate comprehensive report
        report = f"""
EMPIRICAL COMPARISON: Structure-First vs Vanilla Discriminator
==============================================================

Experiment Configuration:
• Number of runs: {len(results['individual_runs'])}
• Training epochs: {self.config.training_config['num_epochs']}
• Batch size: {self.config.training_config['batch_size']}
• Generator architecture: HiFi-GAN
• Dataset: Audio spectrograms

Results Summary:
{stats['summary']}

Detailed Statistics:
{json.dumps(stats['descriptive_statistics'], indent=2)}

Statistical Tests:
{json.dumps(stats['significance_tests'], indent=2)}

Conclusion:
{'SF-VNN discriminator shows statistically significant improvement' if stats['significance_tests'].get('fad_paired_ttest', {}).get('significant', False) else 'No statistically significant difference found'}

Generated on: {datetime.now().isoformat()}
        """.strip()
        
        with open(results_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Paper-ready results saved to: {results_file}")
        return report
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create publication-quality visualizations."""
        
        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data for plotting
        runs = results['individual_runs']
        sfvnn_fads = [r['sfvnn']['best_fad'] for r in runs]
        vanilla_fads = [r['vanilla']['best_fad'] for r in runs]
        
        # Plot 1: FAD Score Comparison
        x_pos = [0, 1]
        fad_means = [np.mean(sfvnn_fads), np.mean(vanilla_fads)]
        fad_stds = [np.std(sfvnn_fads), np.std(vanilla_fads)]
        
        bars = axes[0,0].bar(x_pos, fad_means, yerr=fad_stds, 
                           color=['#2E8B57', '#CD853F'], alpha=0.7,
                           capsize=5)
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(['SF-VNN', 'Vanilla CNN'])
        axes[0,0].set_ylabel('Fréchet Audio Distance (FAD)')
        axes[0,0].set_title('FAD Score Comparison\n(Lower is Better)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, fad_means, fad_stds):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std/2,
                         f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Individual Run Results
        run_ids = range(len(runs))
        axes[0,1].plot(run_ids, sfvnn_fads, 'o-', label='SF-VNN', color='#2E8B57', linewidth=2, markersize=8)
        axes[0,1].plot(run_ids, vanilla_fads, 's-', label='Vanilla CNN', color='#CD853F', linewidth=2, markersize=8)
        axes[0,1].set_xlabel('Run Number')
        axes[0,1].set_ylabel('Best FAD Score')
        axes[0,1].set_title('FAD Scores Across Independent Runs')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Box Plot Comparison
        data_to_plot = [sfvnn_fads, vanilla_fads]
        box_plot = axes[1,0].boxplot(data_to_plot, labels=['SF-VNN', 'Vanilla CNN'],
                                   patch_artist=True)
        box_plot['boxes'][0].set_facecolor('#2E8B57')
        box_plot['boxes'][1].set_facecolor('#CD853F')
        axes[1,0].set_ylabel('FAD Score')
        axes[1,0].set_title('Distribution of FAD Scores')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Parameter Count vs Performance
        sfvnn_params = [r['sfvnn']['num_parameters'] for r in runs]
        vanilla_params = [r['vanilla']['num_parameters'] for r in runs]
        
        axes[1,1].scatter(sfvnn_params, sfvnn_fads, label='SF-VNN', 
                        color='#2E8B57', s=100, alpha=0.7)
        axes[1,1].scatter(vanilla_params, vanilla_fads, label='Vanilla CNN',
                        color='#CD853F', s=100, alpha=0.7)
        axes[1,1].set_xlabel('Number of Parameters')
        axes[1,1].set_ylabel('Best FAD Score')
        axes[1,1].set_title('Parameters vs Performance')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / 'comparison_results.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to: {plot_file}")


def main():
    """Main function to run the empirical comparison."""
    
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="sf_vnn_vs_vanilla_empirical_study",
        description="Rigorous empirical comparison for methods paper",
        random_seed=42
    )
    
    # Update configuration for faster testing (adjust for full experiments)
    config.training_config['num_epochs'] = 20  # Reduce for testing
    config.training_config['batch_size'] = 4   # Reduce for testing
    config.statistical_config['num_runs'] = 3  # Reduce for testing
    
    print("🧪 Empirical Comparison: Structure-First vs Vanilla Discriminator")
    print("=" * 70)
    print(f"Configuration: {config.experiment_name}")
    print(f"Number of runs: {config.statistical_config['num_runs']}")
    print(f"Training epochs per run: {config.training_config['num_epochs']}")
    
    # Check if we have audio data
    test_audio_dir = Path("./test_audio_dataset")
    if not test_audio_dir.exists():
        print("\n⚠️  No audio dataset found. Creating minimal test dataset...")
        from example_real_data import create_minimal_test_dataset
        create_minimal_test_dataset()
    
    # Prepare dataset
    print("\n📁 Preparing dataset...")
    audio_files = list(test_audio_dir.glob("*.wav"))
    
    if len(audio_files) < 2:
        print("❌ Need at least 2 audio files for comparison")
        return
    
    # Split dataset
    train_files = audio_files[:max(1, len(audio_files)//2)]
    val_files = audio_files[max(1, len(audio_files)//2):]
    
    print(f"✓ Dataset: {len(train_files)} train, {len(val_files)} val files")
    
    # Create datasets
    hifi_config = HiFiGANConfig()
    # Update with relevant parameters
    for key, value in config.generator_config.items():
        if hasattr(hifi_config, key):
            setattr(hifi_config, key, value)
    for key, value in config.training_config.items():
        if hasattr(hifi_config, key):
            setattr(hifi_config, key, value)
    
    train_dataset = HiFiGANSFVNNDataset(train_files, hifi_config, training=True)
    val_dataset = HiFiGANSFVNNDataset(val_files, hifi_config, training=False)
    
    # Run empirical comparison
    comparison = EmpiricalComparison(config)
    
    print("\n🔬 Starting empirical comparison...")
    results = comparison.run_multiple_comparisons(train_dataset, val_dataset)
    
    # Generate results
    print("\n📊 Generating results...")
    paper_results = comparison.generate_paper_ready_results(results)
    comparison.create_visualizations(results)
    
    print("\n🎉 Empirical comparison complete!")
    print("📄 Check the results directory for:")
    print("  • Statistical analysis")
    print("  • Publication-ready visualizations")
    print("  • Detailed experimental logs")
    print("\n📝 Ready for methods paper!")
    
    return results


if __name__ == "__main__":
    main()