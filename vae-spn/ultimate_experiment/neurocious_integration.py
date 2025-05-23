#!/usr/bin/env python3
"""
Neurocious - Complete Neural Architecture System
================================================

This module demonstrates the complete integration of the Neurocious system,
including Enhanced VAE, Spatial Probability Networks, and Epistemic Co-Training.

Usage:
    python neurocious_main.py --mode train --config config.yaml
    python neurocious_main.py --mode inference --checkpoint model.pth
"""

import argparse
import asyncio
import json
import yaml
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from core import (
    FieldParameters, CoTrainingConfig, ExplorationState,
    FlowPattern, BeliefExplanation, BeliefReconstructionExplanation
)
from vae import EnhancedVAE, FieldAwareKLDivergence
from spn import SpatialProbabilityNetwork
from co_training import EpistemicCoTraining, InverseFlowIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeurociousSystem:
    """Complete Neurocious neural architecture system"""
    
    def __init__(
        self,
        config: CoTrainingConfig,
        device: str = 'auto',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing Neurocious system on device: {self.device}")
        
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_models()
        self._init_training()
        
        # Training history
        self.training_history = {
            'epoch_losses': [],
            'validation_metrics': [],
            'field_diagnostics': []
        }
    
    def _init_models(self):
        """Initialize neural network models"""
        logger.info("Initializing models...")
        
        # Enhanced VAE
        self.vae = EnhancedVAE(
            input_dim=784,  # MNIST-like input
            hidden_dim=256,
            latent_dim=32,
            sequence_length=16,
            num_heads=4,
            dropout_rate=0.1
        )
        
        # Spatial Probability Network
        self.spn = SpatialProbabilityNetwork(
            vae=self.vae,
            state_dim=32,
            field_shape=(16, 16),
            vector_dim=32,  # Must match latent_dim
            buffer_size=10,
            device=self.device
        )
        
        # Move to device
        self.vae.to(self.device)
        self.spn.to(self.device)
        
        logger.info(f"VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        logger.info(f"SPN parameters: {sum(p.numel() for p in self.spn.parameters()):,}")
    
    def _init_training(self):
        """Initialize training components"""
        logger.info("Initializing training system...")
        
        # Epistemic co-training
        self.co_trainer = EpistemicCoTraining(
            vae=self.vae,
            spn=self.spn,
            config=self.config,
            device=self.device
        )
        
        # Inverse flow integration
        self.inverse_flow = InverseFlowIntegration(
            field_shape=(16, 16),
            vector_dim=8,
            buffer_size=10,
            device=self.device
        )
    
    async def train(
        self,
        training_data: Dict[str, List],
        validation_data: Optional[Dict[str, List]] = None,
        epochs: int = 100,
        save_freq: int = 10,
        eval_freq: int = 5
    ):
        """Train the complete system"""
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training step
            epoch_loss = await self._train_epoch(training_data, epoch)
            self.training_history['epoch_losses'].append(epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
            # Validation step
            if validation_data and epoch % eval_freq == 0:
                val_metrics = await self._validate_epoch(validation_data, epoch)
                self.training_history['validation_metrics'].append({
                    'epoch': epoch,
                    'metrics': val_metrics
                })
                
                val_loss = val_metrics.get('total_loss', float('inf'))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
            
            # Field diagnostics
            field_diagnostics = self.spn.get_diagnostics()
            self.training_history['field_diagnostics'].append({
                'epoch': epoch,
                'diagnostics': field_diagnostics
            })
            
            # Save checkpoint
            if epoch % save_freq == 0:
                self._save_checkpoint(epoch)
            
            # Visualize progress
            if epoch % (save_freq * 2) == 0:
                self._visualize_training_progress()
                self._visualize_field_dynamics()
        
        logger.info("Training completed!")
        self._save_final_results()
    
    async def _train_epoch(self, training_data: Dict[str, List], epoch: int) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        # Extract training sequences
        sequences = training_data['sequences']
        rewards = training_data['rewards']
        actions = training_data['actions']
        reactions = training_data['reactions']
        future_states = training_data['future_states']
        
        # Process in mini-batches
        batch_size = self.config.batch_size
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_rewards = rewards[i:i + batch_size]
            batch_actions = actions[i:i + batch_size]
            batch_reactions = reactions[i:i + batch_size]
            batch_future_states = future_states[i:i + batch_size]
            
            # Training step
            batch_loss = await self.co_trainer.train_step(
                batch_sequences[0] if batch_sequences else [],
                batch_rewards[0] if batch_rewards else [],
                batch_actions[0] if batch_actions else [],
                batch_reactions[0] if batch_reactions else [],
                batch_future_states[0] if batch_future_states else []
            )
            
            total_loss += batch_loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    async def _validate_epoch(self, validation_data: Dict[str, List], epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.vae.eval()
        self.spn.eval()
        
        val_metrics = {}
        
        with torch.no_grad():
            # Compute validation metrics
            sequences = validation_data.get('sequences', [])
            if sequences:
                # Sample a few sequences for validation
                sample_sequences = sequences[:5]
                
                total_recon_loss = 0.0
                total_routing_confidence = 0.0
                
                for seq in sample_sequences:
                    if seq:
                        # Convert to tensors
                        seq_tensors = [torch.tensor(s, dtype=torch.float32, device=self.device) 
                                     for s in seq]
                        
                        # Forward pass through VAE
                        reconstruction, field_params, mean, log_var = self.vae.forward_sequence(seq_tensors)
                        
                        # Reconstruction loss
                        target = seq_tensors[-1]
                        if target.dim() != reconstruction.dim():
                            if target.dim() == 1 and reconstruction.dim() == 2:
                                target = target.unsqueeze(0)
                        recon_loss = F.binary_cross_entropy(reconstruction, torch.sigmoid(target))
                        total_recon_loss += recon_loss.item()
                        
                        # Process through SPN using latent processing
                        latent = mean  # Use mean for validation
                        (routing, confidence, policy, reflexes, predictions, 
                         field_params, explanation, inverse_explanation) = self.spn.process_latent_state(latent)
                        
                        total_routing_confidence += confidence.item()
                
                val_metrics['reconstruction_loss'] = total_recon_loss / len(sample_sequences)
                val_metrics['routing_confidence'] = total_routing_confidence / len(sample_sequences)
                val_metrics['total_loss'] = val_metrics['reconstruction_loss']
        
        self.vae.train()
        self.spn.train()
        
        logger.info(f"Validation - Recon Loss: {val_metrics.get('reconstruction_loss', 0):.4f}, "
                   f"Routing Confidence: {val_metrics.get('routing_confidence', 0):.4f}")
        
        return val_metrics
    
    def inference(
        self, 
        input_sequence: List[torch.Tensor],
        return_explanations: bool = True
    ) -> Dict[str, Any]:
        """Run inference on input sequence"""
        self.vae.eval()
        self.spn.eval()
        
        with torch.no_grad():
            # Move to device
            sequence = [x.to(self.device) for x in input_sequence]
            
            # Process through SPN (SPN handles VAE encoding internally)
            # Use the last element of the sequence as the current state
            current_state = sequence[-1]
            (routing, confidence, policy, reflexes, predictions,
             field_params, explanation, inverse_explanation) = self.spn.process_state(current_state)
            
            # Get latent representation for reconstruction
            mean, log_var = self.vae.encode_sequence(sequence)
            latent = mean  # Use mean for inference
            
            # Decode reconstruction
            reconstruction, _ = self.vae.decode_with_field(latent)
            
            results = {
                'reconstruction': reconstruction.cpu(),
                'routing': routing.cpu(),
                'confidence': confidence.cpu(),
                'policy': policy.cpu(),
                'reflexes': reflexes.cpu(),
                'predictions': predictions.cpu(),
                'field_parameters': asdict(field_params),
                'latent_representation': latent.cpu()
            }
            
            if return_explanations:
                results['explanation'] = {
                    'belief_label': explanation.belief_label,
                    'confidence': explanation.confidence,
                    'justification': explanation.justification,
                    'top_features': explanation.top_contributing_features
                }
                results['inverse_explanation'] = {
                    'temporal_smoothness': inverse_explanation.temporal_smoothness,
                    'reconstruction_confidence': inverse_explanation.reconstruction_confidence,
                    'detailed_explanation': inverse_explanation.generate_detailed_explanation()
                }
        
        return results
    
    def analyze_field_flow(self, state: torch.Tensor, steps: int = 20) -> FlowPattern:
        """Analyze field flow patterns"""
        self.spn.eval()
        with torch.no_grad():
            state = state.to(self.device)
            flow_pattern = self.spn.analyze_field_flow(state, steps)
        return flow_pattern
    
    def simulate_world_branches(
        self, 
        current_state: FieldParameters, 
        num_branches: int = 5
    ) -> List[Dict[str, Any]]:
        """Simulate world branches from current state"""
        self.spn.eval()
        with torch.no_grad():
            branches = self.spn.simulate_world_branches(current_state, num_branches)
            
            branch_results = []
            for i, branch in enumerate(branches):
                branch_info = {
                    'branch_id': i,
                    'probability': branch.probability,
                    'value': branch.value,
                    'initial_state': asdict(branch.initial_state)
                }
                branch_results.append(branch_info)
        
        return branch_results
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        suffix = '_best' if is_best else f'_epoch_{epoch}'
        checkpoint_path = self.checkpoint_dir / f'neurocious_checkpoint{suffix}.pth'
        
        additional_info = {
            'training_history': self.training_history,
            'model_config': {
                'vae_config': {
                    'input_dim': self.vae.input_dim,
                    'hidden_dim': self.vae.hidden_dim,
                    'latent_dim': self.vae.latent_dim
                },
                'spn_config': {
                    'state_dim': self.spn.state_dim,
                    'field_shape': self.spn.field_shape,
                    'vector_dim': self.spn.vector_dim
                }
            }
        }
        
        self.co_trainer.save_checkpoint(str(checkpoint_path), epoch, additional_info)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = self.co_trainer.load_checkpoint(checkpoint_path)
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        return checkpoint
    
    def _save_final_results(self):
        """Save final training results"""
        results_path = self.log_dir / 'training_results.json'
        
        results = {
            'config': asdict(self.config),
            'training_history': self.training_history,
            'final_diagnostics': self.co_trainer.get_training_diagnostics()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_path}")
    
    def _visualize_training_progress(self):
        """Visualize training progress"""
        if not self.training_history['epoch_losses']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.training_history['epoch_losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Validation metrics
        if self.training_history['validation_metrics']:
            val_epochs = [vm['epoch'] for vm in self.training_history['validation_metrics']]
            val_losses = [vm['metrics'].get('total_loss', 0) for vm in self.training_history['validation_metrics']]
            axes[0, 1].plot(val_epochs, val_losses, 'r-', marker='o')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Field entropy over time
        if self.training_history['field_diagnostics']:
            epochs = [fd['epoch'] for fd in self.training_history['field_diagnostics']]
            entropies = [fd['diagnostics'].get('global_entropy', 0) for fd in self.training_history['field_diagnostics']]
            axes[1, 0].plot(epochs, entropies, 'g-')
            axes[1, 0].set_title('Global Field Entropy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True)
        
        # Exploration metrics
        if self.training_history['field_diagnostics']:
            exploration_rates = [fd['diagnostics'].get('exploration_rate', 0) for fd in self.training_history['field_diagnostics']]
            axes[1, 1].plot(epochs, exploration_rates, 'purple')
            axes[1, 1].set_title('Exploration Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_field_dynamics(self):
        """Visualize field dynamics"""
        with torch.no_grad():
            # Get current field state
            vector_field = self.spn.vector_field.cpu().numpy()
            entropy_field = self.spn.entropy_field.cpu().numpy()
            curvature_field = self.spn.curvature_field.cpu().numpy()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Vector field magnitude
            field_magnitude = np.sqrt(np.sum(vector_field ** 2, axis=-1))
            im1 = axes[0, 0].imshow(field_magnitude, cmap='viridis')
            axes[0, 0].set_title('Vector Field Magnitude')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Entropy field
            im2 = axes[0, 1].imshow(entropy_field, cmap='plasma')
            axes[0, 1].set_title('Entropy Field')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Curvature field
            im3 = axes[1, 0].imshow(curvature_field, cmap='coolwarm')
            axes[1, 0].set_title('Curvature Field')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Vector field directions (quiver plot)
            y, x = np.mgrid[0:vector_field.shape[0]:2, 0:vector_field.shape[1]:2]
            u = vector_field[::2, ::2, 0] if vector_field.shape[-1] > 0 else np.zeros_like(x)
            v = vector_field[::2, ::2, 1] if vector_field.shape[-1] > 1 else np.zeros_like(x)
            axes[1, 1].quiver(x, y, u, v, scale=20)
            axes[1, 1].set_title('Vector Field Directions')
            axes[1, 1].set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'field_dynamics.png', dpi=150, bbox_inches='tight')
            plt.close()


def generate_sample_data(num_sequences: int = 100, sequence_length: int = 10) -> Dict[str, List]:
    """Generate sample training data"""
    logger.info(f"Generating {num_sequences} sample sequences...")
    
    sequences = []
    rewards = []
    actions = []
    reactions = []
    future_states = []
    
    for _ in range(num_sequences):
        # Generate random sequence data
        seq = [np.random.randn(784) for _ in range(sequence_length)]
        reward_seq = [np.random.random() for _ in range(sequence_length)]
        action_seq = [np.random.randn(10) for _ in range(sequence_length)]
        reaction_seq = [np.random.randint(0, 2, 5).astype(float) for _ in range(sequence_length)]
        future_seq = [np.random.randn(4) for _ in range(sequence_length)]
        
        sequences.append(seq)
        rewards.append(reward_seq)
        actions.append(action_seq)
        reactions.append(reaction_seq)
        future_states.append(future_seq)
    
    return {
        'sequences': sequences,
        'rewards': rewards,
        'actions': actions,
        'reactions': reactions,
        'future_states': future_states
    }


def load_config(config_path: Optional[str] = None) -> CoTrainingConfig:
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        return CoTrainingConfig(**config_dict)
    else:
        logger.info("Using default configuration")
        return CoTrainingConfig()


async def run_training(args):
    """Run training mode"""
    logger.info("Starting training mode...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize system
    system = NeurociousSystem(
        config=config,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Generate or load training data
    if args.data_path:
        logger.info(f"Loading training data from {args.data_path}")
        # TODO: Implement data loading
        training_data = generate_sample_data(100, 10)
        validation_data = generate_sample_data(20, 10)
    else:
        logger.info("Generating sample training data")
        training_data = generate_sample_data(100, 10)
        validation_data = generate_sample_data(20, 10)
    
    # Run training
    await system.train(
        training_data=training_data,
        validation_data=validation_data,
        epochs=args.epochs,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq
    )


def run_inference(args):
    """Run inference mode"""
    logger.info("Starting inference mode...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize system
    system = NeurociousSystem(
        config=config,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Load checkpoint
    if args.checkpoint:
        system.load_checkpoint(args.checkpoint)
    else:
        logger.error("No checkpoint specified for inference mode")
        return
    
    # Generate sample input
    input_sequence = [torch.randn(784) for _ in range(10)]
    
    # Run inference
    results = system.inference(input_sequence, return_explanations=True)
    
    # Display results
    print("\n=== Inference Results ===")
    print(f"Routing confidence: {results['confidence'].item():.3f}")
    print(f"Field parameters: {results['field_parameters']}")
    print(f"Belief explanation: {results['explanation']['justification']}")
    print(f"Inverse explanation: {results['inverse_explanation']['detailed_explanation']}")
    
    # Analyze field flow
    state = results['latent_representation']
    flow_pattern = system.analyze_field_flow(state)
    print(f"\nFlow Analysis:")
    print(f"  Local curvature: {flow_pattern.local_curvature:.3f}")
    print(f"  Local entropy: {flow_pattern.local_entropy:.3f}")
    print(f"  Stability: {flow_pattern.stability:.3f}")
    
    # Simulate world branches
    field_params = FieldParameters(**results['field_parameters'])
    branches = system.simulate_world_branches(field_params, num_branches=3)
    print(f"\nWorld Branches:")
    for branch in branches:
        print(f"  Branch {branch['branch_id']}: probability={branch['probability']:.3f}, "
              f"value={branch['value']:.3f}")


def run_analysis(args):
    """Run analysis mode"""
    logger.info("Starting analysis mode...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize system
    system = NeurociousSystem(
        config=config,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Load checkpoint
    if args.checkpoint:
        system.load_checkpoint(args.checkpoint)
    else:
        logger.error("No checkpoint specified for analysis mode")
        return
    
    # Generate comprehensive analysis
    logger.info("Generating field dynamics visualization...")
    system._visualize_field_dynamics()
    
    # Get system diagnostics
    diagnostics = system.co_trainer.get_training_diagnostics()
    print("\n=== System Diagnostics ===")
    print(json.dumps(diagnostics, indent=2, default=str))
    
    # Analysis report
    analysis_report = {
        'system_diagnostics': diagnostics,
        'field_analysis': {
            'global_entropy': diagnostics['spn']['global_entropy'],
            'global_curvature': diagnostics['spn']['global_curvature'],
            'coherence_score': diagnostics['spn']['coherence_score']
        }
    }
    
    report_path = system.log_dir / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)
    
    logger.info(f"Analysis report saved to {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Neurocious Neural Architecture System')
    parser.add_argument('--mode', choices=['train', 'inference', 'analysis'], 
                       required=True, help='Operation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('--data-path', type=str, help='Training data path')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', default='./logs', help='Log directory')
    
    # Training specific arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--save-freq', type=int, default=10, help='Checkpoint save frequency')
    parser.add_argument('--eval-freq', type=int, default=5, help='Evaluation frequency')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            asyncio.run(run_training(args))
        elif args.mode == 'inference':
            run_inference(args)
        elif args.mode == 'analysis':
            run_analysis(args)
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()