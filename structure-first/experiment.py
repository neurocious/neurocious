import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import os
from datetime import datetime
import wandb
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd
from collections import defaultdict


@dataclass
class ExperimentConfig:
    """Configuration for structure-first VNN experiments."""
    
    # Architecture parameters
    vector_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    window_size: int = 5
    sigma: float = 1.0
    classifier_hidden: int = 128
    use_spatial_pooling: bool = True
    
    # Loss weights
    alpha_classification: float = 10.0
    alpha_contrastive: float = 1.0
    alpha_triplet: float = 0.5
    alpha_infonce: float = 0.5
    
    # Contrastive learning parameters
    contrastive_type: str = 'all'  # 'contrastive', 'triplet', 'infonce', 'all', 'none'
    margin: float = 1.0
    temperature: float = 0.1
    mining_strategy: str = 'hard'  # 'hard', 'random'
    
    # Training parameters
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    
    # Experiment metadata
    dataset: str = 'cifar10'
    experiment_name: str = 'structure_first_baseline'
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'vector_channels': self.vector_channels,
            'window_size': self.window_size,
            'sigma': self.sigma,
            'classifier_hidden': self.classifier_hidden,
            'use_spatial_pooling': self.use_spatial_pooling,
            'alpha_classification': self.alpha_classification,
            'alpha_contrastive': self.alpha_contrastive,
            'alpha_triplet': self.alpha_triplet,
            'alpha_infonce': self.alpha_infonce,
            'contrastive_type': self.contrastive_type,
            'margin': self.margin,
            'temperature': self.temperature,
            'mining_strategy': self.mining_strategy,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'dataset': self.dataset,
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }


@dataclass
class ExperimentResults:
    """Results from a single experiment run."""
    
    config: ExperimentConfig
    
    # Performance metrics
    final_train_acc: float = 0.0
    final_val_acc: float = 0.0
    best_val_acc: float = 0.0
    convergence_epoch: int = -1
    
    # Structural consistency metrics
    intra_class_distance: float = 0.0
    inter_class_distance: float = 0.0
    consistency_score: float = 0.0
    separation_ratio: float = 0.0
    
    # Training dynamics
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    structural_evolution: Dict[str, List[float]] = field(default_factory=dict)
    
    # Loss component analysis
    loss_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Generalization metrics
    few_shot_accuracy: Dict[int, float] = field(default_factory=dict)  # shots -> accuracy
    transfer_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for saving/analysis."""
        return {
            'config': self.config.to_dict(),
            'final_train_acc': self.final_train_acc,
            'final_val_acc': self.final_val_acc,
            'best_val_acc': self.best_val_acc,
            'convergence_epoch': self.convergence_epoch,
            'intra_class_distance': self.intra_class_distance,
            'inter_class_distance': self.inter_class_distance,
            'consistency_score': self.consistency_score,
            'separation_ratio': self.separation_ratio,
            'training_history': self.training_history,
            'structural_evolution': self.structural_evolution,
            'loss_contributions': self.loss_contributions,
            'few_shot_accuracy': self.few_shot_accuracy,
            'transfer_accuracy': self.transfer_accuracy
        }


class StructuralAnalysisTracker:
    """Tracks structural properties throughout training."""
    
    def __init__(self, model: 'StructureFirstNetwork', device: torch.device):
        self.model = model
        self.device = device
        self.history = {
            'epoch': [],
            'avg_entropy': [],
            'avg_alignment': [],
            'avg_curvature': [],
            'structural_variance': [],
            'class_separation': []
        }
    
    def analyze_batch(self, data: torch.Tensor, targets: torch.Tensor, epoch: int):
        """Analyze structural properties for a batch."""
        self.model.eval()
        
        with torch.no_grad():
            data, targets = data.to(self.device), targets.to(self.device)
            _, signatures = self.model(data, return_signature=True)
            
            # Compute average structural metrics
            avg_entropy = signatures.entropy.mean().item()
            avg_alignment = signatures.alignment.mean().item()
            avg_curvature = signatures.curvature.mean().item()
            
            # Compute structural variance (measure of diversity)
            entropy_flat = signatures.entropy.view(signatures.entropy.size(0), -1)
            structural_variance = entropy_flat.var(dim=0).mean().item()
            
            # Compute class-based structural separation
            embeddings = self.model.compute_structural_embeddings(data)
            class_separation = self._compute_class_separation(embeddings, targets)
            
            # Record metrics
            self.history['epoch'].append(epoch)
            self.history['avg_entropy'].append(avg_entropy)
            self.history['avg_alignment'].append(avg_alignment)
            self.history['avg_curvature'].append(avg_curvature)
            self.history['structural_variance'].append(structural_variance)
            self.history['class_separation'].append(class_separation)
        
        self.model.train()
    
    def _compute_class_separation(self, embeddings: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute silhouette score as measure of class separation."""
        if len(torch.unique(targets)) < 2:
            return 0.0
        
        embeddings_np = embeddings.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        try:
            return silhouette_score(embeddings_np, targets_np)
        except:
            return 0.0


class ExperimentRunner:
    """Runs and manages structure-first VNN experiments."""
    
    def __init__(self, results_dir: str = 'structure_first_experiments'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize logging
        self.experiment_log = []
    
    def run_single_experiment(self, config: ExperimentConfig, 
                            train_loader: torch.utils.data.DataLoader,
                            val_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader,
                            num_classes: int,
                            device: torch.device) -> ExperimentResults:
        """Run a single experiment with given configuration."""
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize model
        model = StructureFirstNetwork(
            input_channels=3,  # Assuming RGB images
            num_classes=num_classes,
            vector_channels=config.vector_channels,
            window_size=config.window_size,
            sigma=config.sigma,
            classifier_hidden=config.classifier_hidden,
            use_spatial_pooling=config.use_spatial_pooling
        ).to(device)
        
        # Initialize optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=config.lr, 
                                    weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        
        loss_fn = self._create_loss_function(config)
        
        # Initialize tracking
        results = ExperimentResults(config=config)
        tracker = StructuralAnalysisTracker(model, device)
        
        # Training loop
        best_val_acc = 0.0
        convergence_epoch = -1
        
        for epoch in range(config.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(model, train_loader, optimizer, loss_fn, device)
            
            # Validation phase
            val_metrics = self._validate_epoch(model, val_loader, device)
            
            # Update learning rate
            scheduler.step()
            
            # Track structural evolution
            if epoch % 10 == 0:  # Sample every 10 epochs
                sample_batch = next(iter(val_loader))
                tracker.analyze_batch(sample_batch[0], sample_batch[1], epoch)
            
            # Record training history
            for key, value in {**train_metrics, **val_metrics}.items():
                if key not in results.training_history:
                    results.training_history[key] = []
                results.training_history[key].append(value)
            
            # Check for convergence
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                convergence_epoch = epoch
                
                # Save best model
                torch.save(model.state_dict(), 
                          os.path.join(self.results_dir, f'{config.experiment_name}_best.pth'))
        
        # Final evaluation
        results.final_train_acc = results.training_history['train_accuracy'][-1]
        results.final_val_acc = results.training_history['val_accuracy'][-1]
        results.best_val_acc = best_val_acc
        results.convergence_epoch = convergence_epoch
        results.structural_evolution = tracker.history
        
        # Analyze structural consistency
        model.load_state_dict(torch.load(os.path.join(self.results_dir, f'{config.experiment_name}_best.pth')))
        consistency_metrics = self._analyze_structural_consistency(model, test_loader, num_classes, device)
        results.intra_class_distance = consistency_metrics['intra_class_distance']
        results.inter_class_distance = consistency_metrics['inter_class_distance']
        results.consistency_score = consistency_metrics['consistency_score']
        results.separation_ratio = consistency_metrics['separation_ratio']
        
        # Few-shot evaluation
        results.few_shot_accuracy = self._evaluate_few_shot(model, test_loader, device, shots=[1, 5, 10])
        
        # Analyze loss contributions
        results.loss_contributions = self._analyze_loss_contributions(
            results.training_history, config)
        
        return results
    
    def _create_loss_function(self, config: ExperimentConfig):
        """Create loss function based on configuration."""
        from structure_first_network import create_structure_first_loss  # Import from previous artifact
        
        return create_structure_first_loss(
            alpha_classification=config.alpha_classification,
            alpha_contrastive=config.alpha_contrastive,
            alpha_triplet=config.alpha_triplet,
            alpha_infonce=config.alpha_infonce,
            contrastive_type=config.contrastive_type
        )
    
    def _train_epoch(self, model, train_loader, optimizer, loss_fn, device):
        """Train for one epoch."""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        loss_components = defaultdict(float)
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            logits, signatures = model(data, return_signature=True)
            losses = loss_fn(logits, targets, signatures)
            
            losses['total'].backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += losses['total'].item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Track loss components
            for key, value in losses.items():
                if key != 'total':
                    loss_components[key] += value.item()
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': 100. * correct / total,
            **{f'train_{k}': v / len(train_loader) for k, v in loss_components.items()}
        }
    
    def _validate_epoch(self, model, val_loader, device):
        """Validate for one epoch."""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                logits = model(data)
                
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': 100. * correct / total,
            'accuracy': 100. * correct / total  # For tracking best model
        }
    
    def _analyze_structural_consistency(self, model, test_loader, num_classes, device):
        """Analyze structural consistency on test set."""
        from structure_first_network import analyze_structural_consistency  # Import from previous artifact
        return analyze_structural_consistency(model, test_loader, num_classes, device)
    
    def _evaluate_few_shot(self, model, test_loader, device, shots=[1, 5, 10]):
        """Evaluate few-shot learning performance."""
        model.eval()
        
        # Collect embeddings and labels
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                embeddings = model.compute_structural_embeddings(data)
                all_embeddings.append(embeddings)
                all_labels.append(targets)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Few-shot evaluation using nearest neighbor
        few_shot_results = {}
        
        for n_shots in shots:
            accuracies = []
            
            for trial in range(5):  # Average over 5 trials
                # Sample support set
                support_embeddings = []
                support_labels = []
                query_embeddings = []
                query_labels = []
                
                for class_id in torch.unique(all_labels):
                    class_mask = all_labels == class_id
                    class_embeddings = all_embeddings[class_mask]
                    
                    if len(class_embeddings) < n_shots + 1:
                        continue
                    
                    # Random permutation
                    perm = torch.randperm(len(class_embeddings))
                    
                    # Support set
                    support_embeddings.append(class_embeddings[perm[:n_shots]])
                    support_labels.extend([class_id] * n_shots)
                    
                    # Query set
                    query_embeddings.append(class_embeddings[perm[n_shots:n_shots+10]])  # 10 queries per class
                    query_labels.extend([class_id] * 10)
                
                if not support_embeddings:
                    continue
                
                support_embeddings = torch.cat(support_embeddings, dim=0)
                support_labels = torch.tensor(support_labels, device=device)
                query_embeddings = torch.cat(query_embeddings, dim=0)
                query_labels = torch.tensor(query_labels, device=device)
                
                # Nearest neighbor classification
                correct = 0
                for i, query_emb in enumerate(query_embeddings):
                    distances = F.mse_loss(query_emb.unsqueeze(0), support_embeddings, reduction='none').mean(dim=1)
                    predicted_label = support_labels[distances.argmin()]
                    if predicted_label == query_labels[i]:
                        correct += 1
                
                accuracies.append(100.0 * correct / len(query_embeddings))
            
            few_shot_results[n_shots] = np.mean(accuracies) if accuracies else 0.0
        
        return few_shot_results
    
    def _analyze_loss_contributions(self, training_history, config):
        """Analyze contribution of different loss components."""
        contributions = {}
        
        # Average loss values over last 10 epochs
        last_epochs = 10
        
        for loss_type in ['classification', 'contrastive', 'triplet', 'infonce']:
            key = f'train_{loss_type}'
            if key in training_history and len(training_history[key]) >= last_epochs:
                contributions[loss_type] = np.mean(training_history[key][-last_epochs:])
        
        return contributions


class ExperimentSuite:
    """Manages and runs comprehensive experimental studies."""
    
    def __init__(self, results_dir: str = 'structure_first_experiments'):
        self.runner = ExperimentRunner(results_dir)
        self.results_dir = results_dir
        self.all_results = []
    
    def run_contrastive_loss_study(self, base_config: ExperimentConfig,
                                 train_loader, val_loader, test_loader, 
                                 num_classes: int, device: torch.device) -> List[ExperimentResults]:
        """Study: Which contrastive loss works best?"""
        
        print("🧪 Running Contrastive Loss Comparison Study...")
        
        contrastive_configs = [
            ('none', 'No contrastive loss'),
            ('contrastive', 'Pairwise contrastive'),
            ('triplet', 'Triplet loss only'),
            ('infonce', 'InfoNCE only'),
            ('all', 'All contrastive losses')
        ]
        
        results = []
        
        for contrastive_type, description in contrastive_configs:
            config = base_config
            config.contrastive_type = contrastive_type
            config.experiment_name = f'contrastive_{contrastive_type}'
            
            print(f"  Running: {description}")
            result = self.runner.run_single_experiment(
                config, train_loader, val_loader, test_loader, num_classes, device)
            results.append(result)
            
            print(f"    Best Val Acc: {result.best_val_acc:.2f}%, "
                  f"Consistency Score: {result.consistency_score:.3f}")
        
        self.all_results.extend(results)
        return results
    
    def run_weight_ablation_study(self, base_config: ExperimentConfig,
                                train_loader, val_loader, test_loader,
                                num_classes: int, device: torch.device) -> List[ExperimentResults]:
        """Study: How do loss weights affect performance?"""
        
        print("🧪 Running Loss Weight Ablation Study...")
        
        weight_configs = [
            # (α_cls, α_cont, α_trip, α_info, description)
            (10.0, 0.0, 0.0, 0.0, 'Classification only'),
            (10.0, 1.0, 0.0, 0.0, 'Cls + Contrastive'),
            (10.0, 0.0, 1.0, 0.0, 'Cls + Triplet'),
            (10.0, 0.0, 0.0, 1.0, 'Cls + InfoNCE'),
            (10.0, 1.0, 0.5, 0.5, 'Balanced (baseline)'),
            (5.0, 2.0, 1.0, 1.0, 'Structure-heavy'),
            (20.0, 0.5, 0.25, 0.25, 'Classification-heavy'),
        ]
        
        results = []
        
        for α_cls, α_cont, α_trip, α_info, description in weight_configs:
            config = base_config
            config.alpha_classification = α_cls
            config.alpha_contrastive = α_cont
            config.alpha_triplet = α_trip
            config.alpha_infonce = α_info
            config.experiment_name = f'weights_{α_cls}_{α_cont}_{α_trip}_{α_info}'
            
            print(f"  Running: {description}")
            result = self.runner.run_single_experiment(
                config, train_loader, val_loader, test_loader, num_classes, device)
            results.append(result)
            
            print(f"    Best Val Acc: {result.best_val_acc:.2f}%, "
                  f"Consistency Score: {result.consistency_score:.3f}")
        
        self.all_results.extend(results)
        return results
    
    def run_architecture_study(self, base_config: ExperimentConfig,
                             train_loader, val_loader, test_loader,
                             num_classes: int, device: torch.device) -> List[ExperimentResults]:
        """Study: How does architecture affect structure learning?"""
        
        print("🧪 Running Architecture Study...")
        
        arch_configs = [
            # (vector_channels, window_size, sigma, description)
            ([16, 32, 64], 3, 0.5, 'Small network, small window'),
            ([32, 64, 128], 5, 1.0, 'Medium network (baseline)'),
            ([64, 128, 256], 7, 1.5, 'Large network, large window'),
            ([32, 64], 5, 1.0, 'Shallow network'),
            ([32, 64, 128, 256], 5, 1.0, 'Deep network'),
        ]
        
        results = []
        
        for channels, window, sigma, description in arch_configs:
            config = base_config
            config.vector_channels = channels
            config.window_size = window
            config.sigma = sigma
            config.experiment_name = f'arch_{len(channels)}layers_w{window}_s{sigma}'
            
            print(f"  Running: {description}")
            result = self.runner.run_single_experiment(
                config, train_loader, val_loader, test_loader, num_classes, device)
            results.append(result)
            
            print(f"    Best Val Acc: {result.best_val_acc:.2f}%, "
                  f"Consistency Score: {result.consistency_score:.3f}")
        
        self.all_results.extend(results)
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        
        print("📊 Generating Comprehensive Report...")
        
        if not self.all_results:
            print("No results to analyze!")
            return {}
        
        report = {
            'summary': self._generate_summary(),
            'best_configurations': self._find_best_configurations(),
            'correlation_analysis': self._analyze_correlations(),
            'training_dynamics': self._analyze_training_dynamics(),
            'structural_insights': self._analyze_structural_patterns(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f'comprehensive_report_{timestamp}.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {report_path}")
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary statistics."""
        accuracies = [r.best_val_acc for r in self.all_results]
        consistency_scores = [r.consistency_score for r in self.all_results]
        
        return {
            'total_experiments': len(self.all_results),
            'accuracy_stats': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'consistency_stats': {
                'mean': np.mean(consistency_scores),
                'std': np.std(consistency_scores),
                'min': np.min(consistency_scores),
                'max': np.max(consistency_scores)
            }
        }
    
    def _find_best_configurations(self) -> Dict[str, Any]:
        """Find best configurations for different metrics."""
        best_accuracy = max(self.all_results, key=lambda r: r.best_val_acc)
        best_consistency = max(self.all_results, key=lambda r: r.consistency_score)
        best_few_shot = max(self.all_results, key=lambda r: r.few_shot_accuracy.get(5, 0))
        
        return {
            'best_accuracy': {
                'experiment': best_accuracy.config.experiment_name,
                'accuracy': best_accuracy.best_val_acc,
                'config': best_accuracy.config.to_dict()
            },
            'best_consistency': {
                'experiment': best_consistency.config.experiment_name,
                'consistency_score': best_consistency.consistency_score,
                'config': best_consistency.config.to_dict()
            },
            'best_few_shot': {
                'experiment': best_few_shot.config.experiment_name,
                'few_shot_5': best_few_shot.few_shot_accuracy.get(5, 0),
                'config': best_few_shot.config.to_dict()
            }
        }
    
    def _analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between hyperparameters and performance."""
        # Extract features and targets
        features = []
        accuracies = []
        consistency_scores = []
        
        for result in self.all_results:
            config = result.config
            feature_vector = [
                config.alpha_classification,
                config.alpha_contrastive,
                config.alpha_triplet,
                config.alpha_infonce,
                len(config.vector_channels),
                config.window_size,
                config.sigma,
                config.lr,
                1 if config.contrastive_type == 'all' else 0
            ]
            features.append(feature_vector)
            accuracies.append(result.best_val_acc)
            consistency_scores.append(result.consistency_score)
        
        features = np.array(features)
        feature_names = [
            'alpha_classification', 'alpha_contrastive', 'alpha_triplet', 'alpha_infonce',
            'network_depth', 'window_size', 'sigma', 'learning_rate', 'uses_all_contrastive'
        ]
        
        # Compute correlations
        correlations = {}
        for i, name in enumerate(feature_names):
            correlations[f'{name}_vs_accuracy'] = np.corrcoef(features[:, i], accuracies)[0, 1]
            correlations[f'{name}_vs_consistency'] = np.corrcoef(features[:, i], consistency_scores)[0, 1]
        
        return correlations
    
    def _analyze_training_dynamics(self) -> Dict[str, Any]:
        """Analyze training dynamics patterns."""
        convergence_epochs = [r.convergence_epoch for r in self.all_results if r.convergence_epoch > 0]
        
        analysis = {
            'convergence_statistics': {
                'mean_convergence_epoch': np.mean(convergence_epochs) if convergence_epochs else 0,
                'std_convergence_epoch': np.std(convergence_epochs) if convergence_epochs else 0,
                'fastest_convergence': np.min(convergence_epochs) if convergence_epochs else 0
            }
        }
        
        # Analyze which configurations converge fastest
        if convergence_epochs:
            fastest_idx = np.argmin([r.convergence_epoch for r in self.all_results if r.convergence_epoch > 0])
            fastest_config = [r for r in self.all_results if r.convergence_epoch > 0][fastest_idx]
            analysis['fastest_converging_config'] = fastest_config.config.to_dict()
        
        return analysis
    
    def _analyze_structural_patterns(self) -> Dict[str, Any]:
        """Analyze structural learning patterns."""
        # Find experiments with best structural consistency
        top_consistency = sorted(self.all_results, key=lambda r: r.consistency_score, reverse=True)[:3]
        
        patterns = {
            'top_consistent_configs': [r.config.to_dict() for r in top_consistency],
            'structure_performance_correlation': np.corrcoef(
                [r.consistency_score for r in self.all_results],
                [r.best_val_acc for r in self.all_results]
            )[0, 1] if len(self.all_results) > 1 else 0.0
        }
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """Generate practical recommendations based on results."""
        recommendations = []
        
        # Find best overall configuration
        best_result = max(self.all_results, key=lambda r: r.best_val_acc)
        recommendations.append(
            f"Best overall configuration: {best_result.config.experiment_name} "
            f"(Val Acc: {best_result.best_val_acc:.2f}%, Consistency: {best_result.consistency_score:.3f})"
        )
        
        # Analyze contrastive loss effectiveness
        contrastive_results = [r for r in self.all_results if 'contrastive' in r.config.experiment_name]
        if contrastive_results:
            best_contrastive = max(contrastive_results, key=lambda r: r.best_val_acc)
            recommendations.append(
                f"Best contrastive loss: {best_contrastive.config.contrastive_type} "
                f"(improved accuracy by {best_contrastive.best_val_acc - min(r.best_val_acc for r in contrastive_results):.2f}%)"
            )
        
        # Structure-performance correlation insight
        structure_correlation = np.corrcoef(
            [r.consistency_score for r in self.all_results],
            [r.best_val_acc for r in self.all_results]
        )[0, 1] if len(self.all_results) > 1 else 0.0
        
        if structure_correlation > 0.3:
            recommendations.append(
                f"Strong positive correlation (r={structure_correlation:.3f}) between structural consistency "
                "and classification accuracy validates the structure-first hypothesis"
            )
        elif structure_correlation < -0.3:
            recommendations.append(
                f"Negative correlation (r={structure_correlation:.3f}) suggests over-regularization. "
                "Consider reducing structural loss weights"
            )
        
        # Weight recommendations
        weight_results = [r for r in self.all_results if 'weights' in r.config.experiment_name]
        if weight_results:
            best_weights = max(weight_results, key=lambda r: r.best_val_acc)
            recommendations.append(
                f"Optimal loss weights: α_cls={best_weights.config.alpha_classification}, "
                f"α_cont={best_weights.config.alpha_contrastive}, "
                f"α_trip={best_weights.config.alpha_triplet}, "
                f"α_info={best_weights.config.alpha_infonce}"
            )
        
        # Architecture recommendations
        arch_results = [r for r in self.all_results if 'arch' in r.config.experiment_name]
        if arch_results:
            best_arch = max(arch_results, key=lambda r: r.best_val_acc)
            recommendations.append(
                f"Optimal architecture: {len(best_arch.config.vector_channels)} layers, "
                f"window_size={best_arch.config.window_size}, sigma={best_arch.config.sigma}"
            )
        
        return recommendations


class ExperimentVisualizer:
    """Creates visualizations for experimental results."""
    
    def __init__(self, results: List[ExperimentResults], save_dir: str):
        self.results = results
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_performance_comparison(self):
        """Plot performance comparison across experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        experiment_names = [r.config.experiment_name for r in self.results]
        accuracies = [r.best_val_acc for r in self.results]
        consistency_scores = [r.consistency_score for r in self.results]
        convergence_epochs = [r.convergence_epoch if r.convergence_epoch > 0 else 100 for r in self.results]
        few_shot_5 = [r.few_shot_accuracy.get(5, 0) for r in self.results]
        
        # Plot 1: Accuracy comparison
        axes[0, 0].bar(range(len(accuracies)), accuracies, color='skyblue')
        axes[0, 0].set_title('Best Validation Accuracy by Experiment')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_xticks(range(len(experiment_names)))
        axes[0, 0].set_xticklabels(experiment_names, rotation=45, ha='right')
        
        # Plot 2: Consistency scores
        axes[0, 1].bar(range(len(consistency_scores)), consistency_scores, color='lightcoral')
        axes[0, 1].set_title('Structural Consistency Score by Experiment')
        axes[0, 1].set_ylabel('Consistency Score')
        axes[0, 1].set_xticks(range(len(experiment_names)))
        axes[0, 1].set_xticklabels(experiment_names, rotation=45, ha='right')
        
        # Plot 3: Convergence speed
        axes[1, 0].bar(range(len(convergence_epochs)), convergence_epochs, color='lightgreen')
        axes[1, 0].set_title('Convergence Speed (Lower is Better)')
        axes[1, 0].set_ylabel('Epochs to Best Performance')
        axes[1, 0].set_xticks(range(len(experiment_names)))
        axes[1, 0].set_xticklabels(experiment_names, rotation=45, ha='right')
        
        # Plot 4: Few-shot performance
        axes[1, 1].bar(range(len(few_shot_5)), few_shot_5, color='gold')
        axes[1, 1].set_title('Few-Shot Performance (5-shot)')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_xticks(range(len(experiment_names)))
        axes[1, 1].set_xticklabels(experiment_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_structure_accuracy_correlation(self):
        """Plot correlation between structural consistency and accuracy."""
        accuracies = [r.best_val_acc for r in self.results]
        consistency_scores = [r.consistency_score for r in self.results]
        experiment_names = [r.config.experiment_name for r in self.results]
        
        plt.figure(figsize=(10, 8))
        
        # Color points by contrastive loss type
        colors = []
        for result in self.results:
            if result.config.contrastive_type == 'none':
                colors.append('red')
            elif result.config.contrastive_type == 'all':
                colors.append('blue')
            else:
                colors.append('orange')
        
        scatter = plt.scatter(consistency_scores, accuracies, c=colors, s=100, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(consistency_scores, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(consistency_scores, p(consistency_scores), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = np.corrcoef(consistency_scores, accuracies)[0, 1]
        
        plt.xlabel('Structural Consistency Score')
        plt.ylabel('Best Validation Accuracy (%)')
        plt.title(f'Structure-Performance Correlation (r = {correlation:.3f})')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='No Contrastive'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Single Contrastive'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='All Contrastive')
        ]
        plt.legend(handles=legend_elements)
        
        # Annotate points
        for i, name in enumerate(experiment_names):
            plt.annotate(name, (consistency_scores[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'structure_accuracy_correlation.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_dynamics(self):
        """Plot training dynamics for key experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Select representative experiments
        representative_configs = ['contrastive_none', 'contrastive_all', 'weights_10.0_1.0_0.5_0.5']
        representative_results = [r for r in self.results if r.config.experiment_name in representative_configs]
        
        if not representative_results:
            representative_results = self.results[:3]  # Fallback to first 3
        
        # Plot training accuracy
        for result in representative_results:
            if 'train_accuracy' in result.training_history:
                epochs = range(len(result.training_history['train_accuracy']))
                axes[0, 0].plot(epochs, result.training_history['train_accuracy'], 
                              label=result.config.experiment_name, linewidth=2)
        
        axes[0, 0].set_title('Training Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Accuracy (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot validation accuracy
        for result in representative_results:
            if 'val_accuracy' in result.training_history:
                epochs = range(len(result.training_history['val_accuracy']))
                axes[0, 1].plot(epochs, result.training_history['val_accuracy'], 
                              label=result.config.experiment_name, linewidth=2)
        
        axes[0, 1].set_title('Validation Accuracy Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot structural evolution (if available)
        for result in representative_results:
            if 'avg_entropy' in result.structural_evolution:
                epochs = result.structural_evolution['epoch']
                axes[1, 0].plot(epochs, result.structural_evolution['avg_entropy'],
                              label=f"{result.config.experiment_name} (entropy)", linewidth=2)
                axes[1, 0].plot(epochs, result.structural_evolution['avg_alignment'],
                              label=f"{result.config.experiment_name} (alignment)", linewidth=2, linestyle='--')
        
        axes[1, 0].set_title('Structural Metrics Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Metric Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot loss components
        for result in representative_results:
            if 'train_classification' in result.training_history:
                epochs = range(len(result.training_history['train_classification']))
                axes[1, 1].plot(epochs, result.training_history['train_classification'],
                              label=f"{result.config.experiment_name} (cls)", linewidth=2)
                
                if 'train_contrastive' in result.training_history:
                    axes[1, 1].plot(epochs, result.training_history['train_contrastive'],
                                  label=f"{result.config.experiment_name} (cont)", linewidth=2, linestyle='--')
        
        axes[1, 1].set_title('Loss Components Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hyperparameter_sensitivity(self):
        """Plot sensitivity to key hyperparameters."""
        # Focus on weight ablation results
        weight_results = [r for r in self.results if 'weights' in r.config.experiment_name]
        
        if not weight_results:
            print("No weight ablation results found for sensitivity analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract weight combinations and performance
        alpha_cls_values = [r.config.alpha_classification for r in weight_results]
        alpha_cont_values = [r.config.alpha_contrastive for r in weight_results]
        accuracies = [r.best_val_acc for r in weight_results]
        consistency_scores = [r.consistency_score for r in weight_results]
        
        # Plot 1: Classification weight vs accuracy
        axes[0, 0].scatter(alpha_cls_values, accuracies, s=100, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Classification Loss Weight (α_cls)')
        axes[0, 0].set_ylabel('Best Validation Accuracy (%)')
        axes[0, 0].set_title('Classification Weight Sensitivity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Contrastive weight vs consistency
        axes[0, 1].scatter(alpha_cont_values, consistency_scores, s=100, alpha=0.7, color='red')
        axes[0, 1].set_xlabel('Contrastive Loss Weight (α_cont)')
        axes[0, 1].set_ylabel('Structural Consistency Score')
        axes[0, 1].set_title('Contrastive Weight vs Structure Quality')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Balance ratio analysis
        balance_ratios = [cont / (cls + 1e-8) for cls, cont in zip(alpha_cls_values, alpha_cont_values)]
        axes[1, 0].scatter(balance_ratios, accuracies, s=100, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Structure/Classification Ratio (α_cont/α_cls)')
        axes[1, 0].set_ylabel('Best Validation Accuracy (%)')
        axes[1, 0].set_title('Loss Balance vs Performance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Multi-dimensional weight heatmap
        # Create a simple heatmap of performance vs two key weights
        from scipy.interpolate import griddata
        
        if len(set(alpha_cls_values)) > 1 and len(set(alpha_cont_values)) > 1:
            xi = np.linspace(min(alpha_cls_values), max(alpha_cls_values), 20)
            yi = np.linspace(min(alpha_cont_values), max(alpha_cont_values), 20)
            Xi, Yi = np.meshgrid(xi, yi)
            
            Zi = griddata((alpha_cls_values, alpha_cont_values), accuracies, (Xi, Yi), method='cubic')
            
            im = axes[1, 1].contourf(Xi, Yi, Zi, levels=15, cmap='viridis', alpha=0.8)
            axes[1, 1].scatter(alpha_cls_values, alpha_cont_values, c=accuracies, 
                             cmap='viridis', s=100, edgecolors='white', linewidth=2)
            
            axes[1, 1].set_xlabel('Classification Weight (α_cls)')
            axes[1, 1].set_ylabel('Contrastive Weight (α_cont)')
            axes[1, 1].set_title('Performance Landscape')
            
            plt.colorbar(im, ax=axes[1, 1], label='Validation Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'hyperparameter_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.show()


# Main execution framework
def run_comprehensive_study(train_loader, val_loader, test_loader, num_classes: int, 
                          device: torch.device, results_dir: str = 'structure_first_study'):
    """
    Run the complete experimental study for structure-first VNNs.
    
    This is the main function that orchestrates all experiments and generates
    the comprehensive analysis for your methods paper.
    """
    
    print("🚀 Starting Comprehensive Structure-First VNN Study")
    print("=" * 60)
    
    # Initialize experiment suite
    suite = ExperimentSuite(results_dir)
    
    # Define base configuration
    base_config = ExperimentConfig(
        vector_channels=[32, 64, 128],
        window_size=5,
        sigma=1.0,
        classifier_hidden=128,
        use_spatial_pooling=True,
        alpha_classification=10.0,
        alpha_contrastive=1.0,
        alpha_triplet=0.5,
        alpha_infonce=0.5,
        contrastive_type='all',
        margin=1.0,
        temperature=0.1,
        mining_strategy='hard',
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        num_epochs=100,
        dataset='cifar10',
        experiment_name='baseline',
        seed=42
    )
    
    # Run experimental studies
    print("\n📋 Experiment Schedule:")
    print("1. Contrastive Loss Comparison")
    print("2. Loss Weight Ablation") 
    print("3. Architecture Study")
    print("4. Comprehensive Analysis & Reporting")
    print()
    
    # Study 1: Contrastive Loss Comparison
    contrastive_results = suite.run_contrastive_loss_study(
        base_config, train_loader, val_loader, test_loader, num_classes, device)
    
    # Study 2: Loss Weight Ablation
    weight_results = suite.run_weight_ablation_study(
        base_config, train_loader, val_loader, test_loader, num_classes, device)
    
    # Study 3: Architecture Study
    arch_results = suite.run_architecture_study(
        base_config, train_loader, val_loader, test_loader, num_classes, device)
    
    # Generate comprehensive report
    print("\n📊 Generating Comprehensive Analysis...")
    report = suite.generate_comprehensive_report()
    
    # Create visualizations
    print("📈 Creating Visualizations...")
    visualizer = ExperimentVisualizer(suite.all_results, os.path.join(results_dir, 'plots'))
    visualizer.plot_performance_comparison()
    visualizer.plot_structure_accuracy_correlation()
    visualizer.plot_training_dynamics()
    visualizer.plot_hyperparameter_sensitivity()
    
    # Print key findings
    print("\n🎯 KEY FINDINGS:")
    print("=" * 50)
    for recommendation in report['recommendations']:
        print(f"• {recommendation}")
    
    print(f"\n📁 All results saved to: {results_dir}")
    print("🎉 Study Complete! Ready for paper writing.")
    
    return suite.all_results, report


# Example usage with CIFAR-10
if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Split test set into validation and test
    val_size = 5000
    test_size = len(testset) - val_size
    valset, testset = torch.utils.data.random_split(testset, [val_size, test_size])
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    
    # Run comprehensive study
    results, report = run_comprehensive_study(
        train_loader, val_loader, test_loader, 
        num_classes=10, device=device, 
        results_dir='cifar10_structure_first_study'
    )
    
    print("Study completed successfully!")
