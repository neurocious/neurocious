import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import asyncio
from collections import defaultdict
import time

from core import CoTrainingConfig, reparameterize, BeliefReconstructionExplanation, TemporalRegularizer, IInverseFlowField
from vae import EnhancedVAE, FieldAwareKLDivergence
from spn import SpatialProbabilityNetwork
from collections import deque


class EpistemicCoTraining:
    """Epistemic co-training system for VAE and SPN"""
    
    def __init__(
        self,
        vae: EnhancedVAE,
        spn: SpatialProbabilityNetwork,
        config: CoTrainingConfig,
        device: str = 'cpu'
    ):
        self.vae = vae
        self.spn = spn
        self.config = config
        self.device = device
        
        # Initialize field-aware KL divergence
        self.field_kl = FieldAwareKLDivergence(spn)
        
        # Initialize optimizers
        self.optimizer_vae = optim.Adam(
            vae.parameters(), 
            lr=config.learning_rate, 
            betas=(0.9, 0.999)
        )
        self.optimizer_spn = optim.Adam(
            spn.parameters(), 
            lr=config.learning_rate, 
            betas=(0.9, 0.999)
        )
        
        # Move to device
        self.vae.to(device)
        self.spn.to(device)
    
    async def train(
        self,
        training_sequences: List[List[torch.Tensor]],
        reward_sequences: List[List[float]],
        expected_action_sequences: List[List[torch.Tensor]],
        observed_reaction_sequences: List[List[torch.Tensor]],
        future_state_sequences: List[List[torch.Tensor]],
        epochs: int,
        validation_data: Optional[Dict[str, Any]] = None
    ):
        """Main training loop"""
        print(f"Starting epistemic co-training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            start_time = time.time()
            
            # Process sequences in batches
            for i in range(0, len(training_sequences), self.config.batch_size):
                batch_size = min(self.config.batch_size, len(training_sequences) - i)
                batch_loss = 0.0
                
                # Process each sequence in the batch
                for j in range(batch_size):
                    sequence_idx = i + j
                    
                    loss = await self.train_step(
                        training_sequences[sequence_idx],
                        reward_sequences[sequence_idx],
                        expected_action_sequences[sequence_idx],
                        observed_reaction_sequences[sequence_idx],
                        future_state_sequences[sequence_idx]
                    )
                    
                    batch_loss += loss
                
                epoch_loss += batch_loss
                batch_count += 1
                
                # Progress reporting
                if batch_count % 10 == 0:
                    avg_batch_loss = batch_loss / batch_size
                    print(f"Epoch {epoch}, Batch {batch_count}, Average Loss: {avg_batch_loss:.4f}")
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s. Average Loss: {avg_epoch_loss:.4f}")
            
            # Validation step
            if epoch % 5 == 0 and validation_data is not None:
                await self.validate_model(validation_data)
    
    async def train_step(
        self,
        input_sequence: List[torch.Tensor],
        rewards: List[float],
        expected_actions: List[torch.Tensor],
        observed_reactions: List[torch.Tensor],
        future_states: List[torch.Tensor]
    ) -> float:
        """Single training step"""
        # Convert to tensors and move to device
        input_sequence = [torch.as_tensor(x, dtype=torch.float32).to(self.device) for x in input_sequence]
        expected_actions = [torch.as_tensor(x, dtype=torch.float32).to(self.device) for x in expected_actions]
        observed_reactions = [torch.as_tensor(x, dtype=torch.float32).to(self.device) for x in observed_reactions]
        future_states = [torch.as_tensor(x, dtype=torch.float32).to(self.device) for x in future_states]
        
        # Build full latent sequence
        latent_sequence = []
        means = []
        log_vars = []
        
        for i in range(len(input_sequence)):
            current_subsequence = input_sequence[:i + 1]
            mean, log_var = self.vae.encode_sequence(current_subsequence)
            latent = reparameterize(mean, log_var)
            
            latent_sequence.append(latent)
            means.append(mean)
            log_vars.append(log_var)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Process each timestep
        for t in range(len(latent_sequence)):
            timestep_loss = await self._process_timestep(
                t, latent_sequence, means, log_vars,
                input_sequence, rewards, expected_actions,
                observed_reactions, future_states
            )
            total_loss += timestep_loss
        
        # Backward pass and optimization
        self.optimizer_vae.zero_grad()
        self.optimizer_spn.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.spn.parameters(), max_norm=1.0)
        
        self.optimizer_vae.step()
        self.optimizer_spn.step()
        
        return float(total_loss.item())
    
    async def _process_timestep(
        self,
        t: int,
        latent_sequence: List[torch.Tensor],
        means: List[torch.Tensor],
        log_vars: List[torch.Tensor],
        input_sequence: List[torch.Tensor],
        rewards: List[float],
        expected_actions: List[torch.Tensor],
        observed_reactions: List[torch.Tensor],
        future_states: List[torch.Tensor]
    ) -> torch.Tensor:
        """Process a single timestep"""
        # Full epistemic processing (use latent state method)
        (routing, confidence, policy, reflexes, predictions,
         field_params, explanation, inverse_explanation) = self.spn.process_latent_state(latent_sequence[t])
        
        # === Core Losses ===
        # Routing and reward
        routing_loss = self._calculate_routing_loss(routing, rewards[t])
        
        # Policy prediction
        policy_loss = self._calculate_policy_loss(policy, expected_actions[t])
        
        # Reflex behavior
        reflex_loss = self._calculate_reflex_loss(reflexes, observed_reactions[t])
        
        # Future prediction
        prediction_loss = self._calculate_prediction_loss(predictions, future_states[t])
        
        # === VAE Losses ===
        # Reconstruction (ensure proper latent vector shape)
        latent_vec = latent_sequence[t].squeeze(0) if latent_sequence[t].dim() > 1 else latent_sequence[t]
        reconstruction, _ = self.vae.decode_with_field(latent_vec)
        recon_loss = self._calculate_reconstruction_loss(input_sequence[t], reconstruction)
        
        # Field-aware KL
        entropy_scale = 1.0 + field_params.entropy * self.config.entropy_scaling
        curvature_scale = 1.0 + field_params.curvature * self.config.curvature_scaling
        kl_loss = self.field_kl.calculate_kl(means[t], log_vars[t], latent_sequence[t])
        
        # === Sequential Losses ===
        narrative_continuity_loss = (
            self._calculate_narrative_continuity_loss(latent_sequence[:t + 1])
            if t > 0 else torch.tensor(0.0, device=self.device)
        )
        
        field_alignment_loss = (
            self._calculate_field_alignment_loss(latent_sequence[:t + 1])
            if t > 0 else torch.tensor(0.0, device=self.device)
        )
        
        # Combine losses
        step_loss = (
            routing_loss +
            self.config.policy_weight * policy_loss +
            self.config.reflex_weight * reflex_loss +
            self.config.prediction_weight * prediction_loss +
            self.config.beta * kl_loss +
            recon_loss +
            self.config.gamma * narrative_continuity_loss +
            self.config.delta * field_alignment_loss
        )
        
        # Optional: Log detailed metrics for this timestep
        if np.random.random() < 0.01:  # 1% chance
            self._log_step_metrics(t, routing, policy, reflexes, predictions,
                                 explanation, inverse_explanation, field_params)
        
        return step_loss
    
    def _calculate_routing_loss(self, routing: torch.Tensor, reward: float) -> torch.Tensor:
        """Calculate routing loss weighted by reward"""
        reward_tensor = torch.tensor([reward], device=self.device)
        # Weight routing probabilities by reward (negative for maximization)
        return -torch.mean(routing * reward_tensor)
    
    def _calculate_policy_loss(self, policy: torch.Tensor, expected_action: torch.Tensor) -> torch.Tensor:
        """Calculate policy prediction loss"""
        return F.mse_loss(policy, expected_action)
    
    def _calculate_reflex_loss(self, reflexes: torch.Tensor, observed_reaction: torch.Tensor) -> torch.Tensor:
        """Calculate reflex behavior loss"""
        return F.binary_cross_entropy(reflexes, observed_reaction)
    
    def _calculate_prediction_loss(self, predictions: torch.Tensor, future_state: torch.Tensor) -> torch.Tensor:
        """Calculate future prediction loss"""
        return F.mse_loss(predictions, future_state)
    
    def _calculate_reconstruction_loss(self, input_tensor: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss"""
        # Ensure both tensors have the same shape
        if input_tensor.dim() != output.dim():
            if input_tensor.dim() == 1 and output.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            elif input_tensor.dim() == 2 and output.dim() == 1:
                output = output.unsqueeze(0)
        
        # Normalize input to [0,1] range for BCE
        input_normalized = torch.sigmoid(input_tensor)
        
        return F.binary_cross_entropy(output, input_normalized)
    
    def _calculate_narrative_continuity_loss(self, latent_sequence: List[torch.Tensor]) -> torch.Tensor:
        """Calculate narrative continuity loss"""
        if len(latent_sequence) < 2:
            return torch.tensor(0.0, device=self.device)
        
        losses = []
        for t in range(1, len(latent_sequence)):
            current = latent_sequence[t]
            previous = latent_sequence[t - 1]
            
            # L2 distance between consecutive states
            diff = current - previous
            squared_norm = torch.sum(diff ** 2)
            losses.append(squared_norm)
        
        return torch.mean(torch.stack(losses))
    
    def _calculate_field_alignment_loss(self, latent_sequence: List[torch.Tensor]) -> torch.Tensor:
        """Calculate field alignment loss"""
        if len(latent_sequence) < 2:
            return torch.tensor(0.0, device=self.device)
        
        losses = []
        for t in range(len(latent_sequence) - 1):
            current = latent_sequence[t]
            next_state = latent_sequence[t + 1]
            
            # Get actual transition vector
            transition_vector = next_state - current
            
            # Get field-predicted direction
            field_flat = self.spn.vector_field.view(-1, self.spn.vector_dim)
            field_direction = torch.matmul(current, field_flat.T)
            field_direction = torch.mean(field_direction, dim=-1)
            
            # Calculate cosine similarity
            cosine_sim = F.cosine_similarity(
                transition_vector.unsqueeze(0), 
                field_direction.unsqueeze(0), 
                dim=-1
            )
            
            # Loss is 1 - cosine similarity
            losses.append(1.0 - cosine_sim)
        
        return torch.mean(torch.stack(losses))
    
    def _log_step_metrics(
        self,
        timestep: int,
        routing: torch.Tensor,
        policy: torch.Tensor,
        reflexes: torch.Tensor,
        predictions: torch.Tensor,
        explanation: Any,
        inverse_explanation: Any,
        field_params: Any
    ):
        """Log detailed metrics for a timestep"""
        print(f"Timestep {timestep} Metrics:")
        print(f"  Routing confidence: {torch.max(routing):.3f}")
        print(f"  Policy certainty: {torch.max(policy):.3f}")
        print(f"  Active reflexes: {torch.sum(reflexes > 0.5).item()}")
        print(f"  Prediction confidence: {torch.mean(predictions):.3f}")
        print(f"  Field params - Entropy: {field_params.entropy:.3f}, "
              f"Curvature: {field_params.curvature:.3f}")
        print(f"  Explanation quality: {explanation.confidence:.3f}")
        print(f"  Inverse reconstruction smoothness: {inverse_explanation.temporal_smoothness:.3f}")
    
    async def validate_model(self, validation_data: Dict[str, Any]):
        """Validate model performance"""
        print("Running validation...")
        
        self.vae.eval()
        self.spn.eval()
        
        with torch.no_grad():
            # TODO: Implement comprehensive validation
            # - Check belief coherence
            # - Evaluate prediction accuracy
            # - Assess policy performance
            # - Measure reflex responsiveness
            
            # Placeholder validation metrics
            validation_metrics = {
                "belief_coherence": 0.8,
                "prediction_accuracy": 0.75,
                "policy_performance": 0.82,
                "reflex_responsiveness": 0.9
            }
            
            print("Validation Results:")
            for metric, value in validation_metrics.items():
                print(f"  {metric}: {value:.3f}")
        
        self.vae.train()
        self.spn.train()
    
    def save_checkpoint(self, filepath: str, epoch: int, additional_info: Optional[Dict] = None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'spn_state_dict': self.spn.state_dict(),
            'optimizer_vae_state_dict': self.optimizer_vae.state_dict(),
            'optimizer_spn_state_dict': self.optimizer_spn.state_dict(),
            'config': self.config.__dict__,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.spn.load_state_dict(checkpoint['spn_state_dict'])
        self.optimizer_vae.load_state_dict(checkpoint['optimizer_vae_state_dict'])
        self.optimizer_spn.load_state_dict(checkpoint['optimizer_spn_state_dict'])
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive training diagnostics"""
        vae_diagnostics = {
            'kl_weight': self.vae.kl_weight,
            'latent_dim': self.vae.latent_dimension,
        }
        
        spn_diagnostics = self.spn.get_diagnostics()
        
        return {
            'vae': vae_diagnostics,
            'spn': spn_diagnostics,
            'config': self.config.__dict__
        }


# ========================= INVERSE FLOW INTEGRATION =========================

class InverseFlowIntegration:
    """Integration system for inverse flow reconstruction"""
    
    def __init__(
        self,
        field_shape: Tuple[int, int],
        vector_dim: int,
        buffer_size: int = 10,
        device: str = 'cpu'
    ):
        self.field_shape = field_shape
        self.vector_dim = vector_dim
        self.buffer_size = buffer_size
        self.device = device
        
        # Initialize components
        from spn import InverseFlowField
        self.inverse_flow_field = InverseFlowField(field_shape, vector_dim, device=device)
        self.temporal_buffer = TemporalBuffer(buffer_size)
        self.causal_strength_cache = {}
    
    def reconstruct_prior_belief(
        self,
        current_state: torch.Tensor,
        context_state: torch.Tensor,
        potential_antecedents: Optional[List[str]] = None
    ) -> BeliefReconstructionExplanation:
        """Reconstruct prior belief state"""
        from core import TemporalRegularizer
        
        current_state = current_state.to(self.device)
        context_state = context_state.to(self.device)
        
        regularizer = TemporalRegularizer()
        result = self.inverse_flow_field.generate_previous_state_with_context(
            current_state, context_state, regularizer
        )
        
        # Identify causal antecedents
        antecedents = self._identify_causal_antecedents(
            result.warped_state, potential_antecedents
        )
        
        # Calculate attribution scores
        attribution_scores = self._calculate_attribution_scores(
            result.warped_state, antecedents
        )
        
        return BeliefReconstructionExplanation(
            warped_prior_state=result.warped_state,
            temporal_smoothness=result.temporal_smoothness,
            confidence_metrics=result.confidence_metrics,
            causal_justification=self._generate_inverse_justification(result, attribution_scores),
            causal_antecedents=antecedents,
            attribution_scores=attribution_scores,
            reconstruction_confidence=self._calculate_reconstruction_confidence(result)
        )
    
    def update_from_forward_dynamics(
        self, 
        forward_field: torch.Tensor, 
        forward_routing: torch.Tensor
    ):
        """Update from forward dynamics"""
        self.inverse_flow_field.update_from_forward_field(forward_field, forward_routing)
    
    def add_to_temporal_buffer(self, state: torch.Tensor):
        """Add state to temporal buffer"""
        self.temporal_buffer.add(state)
    
    def _identify_causal_antecedents(
        self,
        warped_state: torch.Tensor,
        potential_antecedents: Optional[List[str]]
    ) -> List[str]:
        """Identify causal antecedents"""
        if not potential_antecedents:
            return []
        
        antecedents = []
        context_window = self.temporal_buffer.get_recent_states(5)
        
        for antecedent in potential_antecedents:
            # Check temporal alignment
            temporally_valid = self._validate_temporal_alignment(antecedent, context_window)
            
            # Check causal strength
            causal_strength = self._calculate_causal_strength(warped_state, antecedent)
            
            if temporally_valid and causal_strength > 0.3:
                antecedents.append(antecedent)
                self.causal_strength_cache[antecedent] = causal_strength
        
        return antecedents
    
    def _calculate_attribution_scores(
        self,
        warped_state: torch.Tensor,
        antecedents: List[str]
    ) -> Dict[str, float]:
        """Calculate attribution scores"""
        scores = {}
        
        for antecedent in antecedents:
            if antecedent in self.causal_strength_cache:
                scores[antecedent] = self.causal_strength_cache[antecedent]
            else:
                strength = self._calculate_causal_strength(warped_state, antecedent)
                scores[antecedent] = strength
                self.causal_strength_cache[antecedent] = strength
        
        # Normalize scores
        if scores:
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _validate_temporal_alignment(
        self, 
        antecedent: str, 
        context_window: List[torch.Tensor]
    ) -> bool:
        """Validate temporal alignment of antecedent"""
        # Simple validation - could be extended with sophisticated temporal logic
        return len(context_window) > 0
    
    def _calculate_causal_strength(self, warped_state: torch.Tensor, antecedent: str) -> float:
        """Calculate causal strength between warped state and antecedent"""
        # Try to get antecedent state from temporal buffer
        antecedent_state = self.temporal_buffer.try_get_state(antecedent)
        if antecedent_state is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            warped_state.unsqueeze(0), 
            antecedent_state.unsqueeze(0), 
            dim=-1
        )
        return max(0.0, float(similarity.item()))
    
    def _generate_inverse_justification(
        self,
        result: Any,
        attribution_scores: Dict[str, float]
    ) -> str:
        """Generate inverse justification text"""
        lines = [
            f"Temporal coherence: {result.temporal_smoothness:.3f}",
        ]
        
        # Add routing confidence if available
        routing_conf = result.confidence_metrics.get("routing_confidence", 0.0)
        lines.append(f"Routing confidence: {routing_conf:.3f}")
        
        # Add causal attribution
        if attribution_scores:
            lines.append("\nCausal influences:")
            for cause, strength in sorted(attribution_scores.items(), 
                                        key=lambda x: x[1], reverse=True):
                lines.append(f"- {cause}: {strength:.3f}")
        
        return "\n".join(lines)
    
    def _calculate_reconstruction_confidence(self, result: Any) -> float:
        """Calculate reconstruction confidence"""
        temporal_weight = 0.4
        routing_weight = 0.4
        warping_weight = 0.2
        
        confidence = (
            temporal_weight * result.temporal_smoothness +
            routing_weight * result.confidence_metrics.get("routing_confidence", 0.0) +
            warping_weight * result.confidence_metrics.get("warping_confidence", 0.0)
        )
        
        return confidence


class TemporalBuffer:
    """Buffer for temporal state management"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_map = {}
    
    def add(self, state: torch.Tensor):
        """Add state to buffer"""
        state_id = f"state_{len(self.buffer)}"
        
        if len(self.buffer) >= self.capacity:
            # Remove oldest state from map
            old_id = f"state_{len(self.buffer) - self.capacity}"
            self.state_map.pop(old_id, None)
        
        self.buffer.append(state)
        self.state_map[state_id] = state
    
    def get_recent_states(self, count: int) -> List[torch.Tensor]:
        """Get recent states"""
        return list(self.buffer)[-count:] if len(self.buffer) >= count else list(self.buffer)
    
    def try_get_state(self, state_id: str) -> Optional[torch.Tensor]:
        """Try to get state by ID"""
        return self.state_map.get(state_id)


# ========================= EXAMPLE USAGE =========================

def create_training_example():
    """Create example training setup"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    config = CoTrainingConfig(
        batch_size=16,
        learning_rate=0.001,
        beta=0.7,
        gamma=0.5,
        delta=0.3
    )
    
    # Initialize models
    vae = EnhancedVAE(
        input_dim=784,
        hidden_dim=256,
        latent_dim=32,
        sequence_length=16
    )
    
    spn = SpatialProbabilityNetwork(
        vae=vae,
        state_dim=32,
        field_shape=(16, 16),
        vector_dim=8,
        buffer_size=10,
        device=device
    )
    
    # Initialize co-training
    co_trainer = EpistemicCoTraining(vae, spn, config, device)
    
    return co_trainer, vae, spn


async def run_training_example():
    """Run example training"""
    co_trainer, vae, spn = create_training_example()
    
    # Generate dummy data
    sequence_length = 10
    batch_size = 32
    input_dim = 784
    
    training_sequences = []
    reward_sequences = []
    expected_action_sequences = []
    observed_reaction_sequences = []
    future_state_sequences = []
    
    for _ in range(batch_size):
        # Random sequences
        training_seq = [torch.randn(input_dim) for _ in range(sequence_length)]
        reward_seq = [np.random.random() for _ in range(sequence_length)]
        action_seq = [torch.randn(10) for _ in range(sequence_length)]
        reaction_seq = [torch.randint(0, 2, (5,)).float() for _ in range(sequence_length)]
        future_seq = [torch.randn(4) for _ in range(sequence_length)]
        
        training_sequences.append(training_seq)
        reward_sequences.append(reward_seq)
        expected_action_sequences.append(action_seq)
        observed_reaction_sequences.append(reaction_seq)
        future_state_sequences.append(future_seq)
    
    # Run training
    await co_trainer.train(
        training_sequences=training_sequences,
        reward_sequences=reward_sequences,
        expected_action_sequences=expected_action_sequences,
        observed_reaction_sequences=observed_reaction_sequences,
        future_state_sequences=future_state_sequences,
        epochs=5
    )
    
    # Get diagnostics
    diagnostics = co_trainer.get_training_diagnostics()
    print("\nTraining Diagnostics:")
    print(diagnostics)


if __name__ == "__main__":
    # Run example
    asyncio.run(run_training_example())