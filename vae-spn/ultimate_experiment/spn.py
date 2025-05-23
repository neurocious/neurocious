import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import math
import random

from core import (
    ISpatialProbabilityNetwork, IEnhancedVAE, FieldParameters, ExplorationState,
    FlowPattern, GeometricField, BeliefExplanation, BeliefReconstructionExplanation,
    WorldBranch, normalize_vector_field, reparameterize, IInverseFlowField,
    InverseTransformationState, FieldMetrics, TemporalRegularizer
)
from vae import EnhancedVAE, PolicyNetwork, ReflexNetwork, PredictionNetwork


class SpatialProbabilityNetwork(nn.Module, ISpatialProbabilityNetwork):
    """Spatial Probability Network for belief routing and field dynamics"""
    
    # Constants
    LEARNING_RATE = 0.01
    FIELD_DECAY = 0.999
    MIN_FIELD_STRENGTH = 1e-6
    NOVELTY_WEIGHT = 0.1
    BRANCH_DECAY_RATE = 0.95
    
    def __init__(
        self,
        vae: Optional[IEnhancedVAE] = None,
        state_dim: int = 20,
        field_shape: Tuple[int, int] = (32, 32),
        vector_dim: int = 8,
        buffer_size: int = 10,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.field_shape = field_shape
        self.vector_dim = vector_dim
        self.buffer_size = buffer_size
        self.device = device
        self.vae_model = vae
        
        # Initialize fields as parameters
        self._init_fields()
        
        # Temporal buffer
        self.temporal_buffer = deque(maxlen=buffer_size)
        
        # Neural components
        self.policy_network = PolicyNetwork(state_dim, buffer_size=buffer_size)
        self.reflex_network = ReflexNetwork(state_dim)
        self.prediction_network = PredictionNetwork(state_dim, buffer_size=buffer_size)
        
        # Branching and exploration
        self.branches = []
        self.route_visits = defaultdict(int)
        
        # Move to device
        self.to(device)
    
    def _init_fields(self):
        """Initialize spatial probability fields"""
        # Vector field: [height, width, vector_dim]
        vector_field_data = torch.randn(*self.field_shape, self.vector_dim) * 0.01
        self.vector_field = nn.Parameter(normalize_vector_field(vector_field_data))
        
        # Field metrics: [height, width]
        field_size = self.field_shape[0] * self.field_shape[1]
        self.curvature_field = nn.Parameter(torch.zeros(*self.field_shape))
        self.entropy_field = nn.Parameter(torch.full(self.field_shape, 1.0 / field_size))
        self.alignment_field = nn.Parameter(torch.zeros(*self.field_shape))
    
    def process_state(self, state: torch.Tensor) -> Tuple[
        torch.Tensor,  # routing
        torch.Tensor,  # confidence
        torch.Tensor,  # policy
        torch.Tensor,  # reflexes
        torch.Tensor,  # predictions
        FieldParameters,  # field_params
        BeliefExplanation,  # explanation
        BeliefReconstructionExplanation  # inverse_explanation
    ]:
        """Process state through the spatial probability network (expects raw input)"""
        return self._process_state_internal(state, is_latent=False)
    
    def process_latent_state(self, latent_state: torch.Tensor) -> Tuple[
        torch.Tensor,  # routing
        torch.Tensor,  # confidence
        torch.Tensor,  # policy
        torch.Tensor,  # reflexes
        torch.Tensor,  # predictions
        FieldParameters,  # field_params
        BeliefExplanation,  # explanation
        BeliefReconstructionExplanation  # inverse_explanation
    ]:
        """Process latent state directly (bypasses VAE encoding)"""
        return self._process_state_internal(latent_state, is_latent=True)
    
    def _process_state_internal(self, state: torch.Tensor, is_latent: bool = False) -> Tuple[
        torch.Tensor,  # routing
        torch.Tensor,  # confidence
        torch.Tensor,  # policy
        torch.Tensor,  # reflexes
        torch.Tensor,  # predictions
        FieldParameters,  # field_params
        BeliefExplanation,  # explanation
        BeliefReconstructionExplanation  # inverse_explanation
    ]:
        """Process state through the spatial probability network"""
        # Ensure state is on correct device
        state = state.to(self.device)
        
        if is_latent:
            # State is already a latent vector, use directly
            processed_sequence = state.squeeze(0) if state.dim() > 1 else state
            
            # Calculate routing directly from latent state
            field_flat = self.vector_field.view(-1, self.vector_dim)
            similarity = torch.matmul(processed_sequence.unsqueeze(0), field_flat.T)
            routing = F.softmax(similarity, dim=-1)
            
            # Calculate field parameters
            field_params = self._calculate_field_parameters(processed_sequence, routing)
            
            # Update field metrics
            self._update_field_metrics(processed_sequence, routing, field_params)
            
            # Add exploration
            exploration = self._update_exploration(processed_sequence)
            routing = self._add_exploration_noise(routing, exploration.exploration_rate)
            
            # Calculate confidence
            confidence = self._calculate_routing_confidence(field_params)
        else:
            # State is raw input, use normal processing
            # Add to temporal buffer
            self.temporal_buffer.append(state.detach().clone())
            
            # Create sequence from buffer
            sequence = list(self.temporal_buffer)
            
            # Get base routing with exploration
            routing, confidence, field_params, processed_sequence = self._route_state_internal(sequence)
            
            # Squeeze the processed sequence to remove batch dimension if present
            processed_sequence = processed_sequence.squeeze(0) if processed_sequence.dim() > 1 else processed_sequence
        
        # Add processed sequence to latent buffer for history
        if not hasattr(self, 'latent_buffer'):
            self.latent_buffer = deque(maxlen=self.buffer_size)
        self.latent_buffer.append(processed_sequence.detach().clone())
        
        # Get temporal context
        history_tensor = self._get_history_tensor()
        
        # Generate policy and value (use processed sequence, which is the latent representation)
        # Ensure both tensors have the same dimensions
        latent_vec = processed_sequence.squeeze(0) if processed_sequence.dim() > 1 else processed_sequence
        policy, _ = self.policy_network(latent_vec, history_tensor)
        
        # Check reflexes (use processed sequence)
        reflexes = self.reflex_network(latent_vec)
        
        # Make predictions
        predictions = self.prediction_network(history_tensor)
        
        # Generate belief explanation
        explanation = self._generate_belief_explanation(
            processed_sequence, routing, field_params, confidence
        )
        
        # Generate inverse explanation (simplified for now)
        inverse_explanation = self._generate_inverse_explanation(state, history_tensor)
        
        return (routing, confidence, policy, reflexes, predictions,
                field_params, explanation, inverse_explanation)
    
    def _route_state_internal(
        self, 
        sequence: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, FieldParameters, torch.Tensor]:
        """Internal routing computation"""
        # Process through VAE if available
        routing_state = self._process_vae_sequence(sequence)
        
        # Calculate base routing through field similarity
        # Flatten vector field for matrix multiplication
        field_flat = self.vector_field.view(-1, self.vector_dim)  # [H*W, vector_dim]
        
        # Calculate similarity between state and each field position
        similarity = torch.matmul(routing_state, field_flat.T)  # [batch, H*W]
        base_routing = F.softmax(similarity, dim=-1)
        
        # Calculate field parameters
        field_params = self._calculate_field_parameters(routing_state, base_routing)
        
        # Update field metrics
        self._update_field_metrics(routing_state, base_routing, field_params)
        
        # Add exploration
        exploration = self._update_exploration(routing_state)
        routing = self._add_exploration_noise(base_routing, exploration.exploration_rate)
        
        # Calculate confidence
        confidence = self._calculate_routing_confidence(field_params)
        
        return routing, confidence, field_params, routing_state
    
    def _process_vae_sequence(self, sequence: List[torch.Tensor]) -> torch.Tensor:
        """Process sequence through VAE if available"""
        if self.vae_model is None:
            return sequence[-1] if sequence else torch.zeros(self.state_dim, device=self.device)
        
        # Encode through VAE
        mean, log_var = self.vae_model.encode_sequence(sequence)
        return reparameterize(mean, log_var)
    
    def _calculate_field_parameters(
        self, 
        state: torch.Tensor, 
        routing: torch.Tensor
    ) -> FieldParameters:
        """Calculate local field parameters"""
        # Calculate weighted field metrics based on routing
        curvature = torch.sum(routing * self.curvature_field.view(-1))
        entropy = torch.sum(routing * self.entropy_field.view(-1))
        alignment = torch.sum(routing * self.alignment_field.view(-1))
        
        return FieldParameters(
            curvature=float(curvature.item()),
            entropy=float(entropy.item()),
            alignment=float(alignment.item())
        )
    
    def _update_field_metrics(
        self, 
        state: torch.Tensor, 
        routing: torch.Tensor, 
        local_params: FieldParameters
    ):
        """Update field metrics based on current state"""
        with torch.no_grad():
            # Reshape routing to field shape
            routing_2d = routing.view(*self.field_shape)
            
            # Update curvature field
            curvature_update = local_params.curvature * (1 - self.FIELD_DECAY)
            self.curvature_field.data = (
                self.curvature_field.data * self.FIELD_DECAY + 
                routing_2d * curvature_update * self.LEARNING_RATE
            )
            
            # Update entropy field
            entropy_update = local_params.entropy * (1 - self.FIELD_DECAY)
            self.entropy_field.data = (
                self.entropy_field.data * self.FIELD_DECAY + 
                routing_2d * entropy_update * self.LEARNING_RATE
            )
            
            # Update alignment field
            alignment_update = local_params.alignment * (1 - self.FIELD_DECAY)
            self.alignment_field.data = (
                self.alignment_field.data * self.FIELD_DECAY + 
                routing_2d * alignment_update * self.LEARNING_RATE
            )
    
    def _update_exploration(self, state: torch.Tensor) -> ExplorationState:
        """Update exploration state"""
        route_signature = self._calculate_route_signature(state)
        self.route_visits[route_signature] += 1
        
        novelty_score = self._calculate_novelty_score(route_signature)
        uncertainty_score = float(torch.mean(self.entropy_field).item())
        exploration_rate = self._combine_exploration_factors(novelty_score, uncertainty_score)
        
        return ExplorationState(
            novelty_score=novelty_score,
            uncertainty_score=uncertainty_score,
            exploration_rate=exploration_rate
        )
    
    def _calculate_route_signature(self, state: torch.Tensor) -> str:
        """Calculate signature for route tracking"""
        rounded_state = torch.round(state * 100) / 100  # Round to 2 decimal places
        numpy_state = rounded_state.detach().cpu().numpy()  # Detach to avoid gradient issues
        # Flatten the array to handle multi-dimensional tensors
        if numpy_state.ndim > 1:
            numpy_state = numpy_state.flatten()
        return ",".join([f"{float(x):.2f}" for x in numpy_state])
    
    def _calculate_novelty_score(self, route_signature: str) -> float:
        """Calculate novelty score based on visit frequency"""
        visits = self.route_visits[route_signature]
        return float(math.exp(-visits * self.NOVELTY_WEIGHT))
    
    def _combine_exploration_factors(self, novelty: float, uncertainty: float) -> float:
        """Combine exploration factors"""
        base_rate = 0.1
        novelty_factor = self.NOVELTY_WEIGHT * novelty
        uncertainty_factor = (1 - self.NOVELTY_WEIGHT) * uncertainty
        return base_rate * (novelty_factor + uncertainty_factor)
    
    def _add_exploration_noise(
        self, 
        probs: torch.Tensor, 
        exploration_rate: float
    ) -> torch.Tensor:
        """Add exploration noise to routing probabilities"""
        noise = torch.randn_like(probs) * exploration_rate
        return F.softmax(probs + noise, dim=-1)
    
    def _calculate_routing_confidence(self, field_params: FieldParameters) -> torch.Tensor:
        """Calculate routing confidence"""
        confidence = (
            (1 - field_params.entropy) *                    # High confidence when entropy is low
            (1 / (1 + field_params.curvature)) *           # High confidence when curvature is low
            abs(field_params.alignment)                     # High confidence when alignment is strong
        )
        
        return torch.tensor([confidence], device=self.device)
    
    def _get_history_tensor(self) -> torch.Tensor:
        """Get history as tensor"""
        # Use latent buffer if available, otherwise use zeros
        if hasattr(self, 'latent_buffer') and self.latent_buffer:
            history_list = list(self.latent_buffer)
            
            # Pad with zeros to reach buffer_size if needed
            while len(history_list) < self.buffer_size:
                history_list.insert(0, torch.zeros(self.state_dim, device=self.device))
            
            # Take the last buffer_size elements and flatten
            history_tensors = history_list[-self.buffer_size:]
            history_tensor = torch.cat(history_tensors, dim=0)  # Flatten to [buffer_size * state_dim]
        else:
            history_tensor = torch.zeros(self.buffer_size * self.state_dim, device=self.device)
        
        return history_tensor
    
    def _generate_belief_explanation(
        self,
        latent: torch.Tensor,
        routing: torch.Tensor,
        field_params: FieldParameters,
        confidence: torch.Tensor
    ) -> BeliefExplanation:
        """Generate explanation for belief formation"""
        # Calculate feature contributions
        contributions = {}
        latent_data = latent.detach().cpu().numpy().flatten()  # Ensure 1D array and detach
        vector_field_flat = self.vector_field.view(-1, self.vector_dim).detach().cpu().numpy()
        
        # Calculate contributions through field alignment
        for i in range(min(len(latent_data), self.vector_dim)):
            if i < vector_field_flat.shape[1]:
                contribution = float(latent_data[i]) * float(np.mean(vector_field_flat[:, i]))
                contributions[f"feature_{i}"] = contribution
        
        # Get top contributors
        top_contributors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Generate counterfactuals (simplified)
        counterfactuals = {}
        for feature, _ in top_contributors:
            # Simulate zeroing out the feature
            counterfactuals[feature] = random.uniform(0.1, 0.5)  # Placeholder
        
        return BeliefExplanation(
            belief_label=self._get_top_attractor_region(routing),
            feature_contributions=contributions,
            confidence=float(confidence.item()),
            field_params=field_params,
            justification=self._generate_justification(top_contributors, counterfactuals, field_params),
            top_contributing_features=[tc[0] for tc in top_contributors],
            counterfactual_shifts=counterfactuals,
            trajectory_path=[t.cpu().numpy() for t in self.temporal_buffer]
        )
    
    def _generate_inverse_explanation(
        self, 
        state: torch.Tensor, 
        history: torch.Tensor
    ) -> BeliefReconstructionExplanation:
        """Generate inverse explanation (simplified implementation)"""
        return BeliefReconstructionExplanation(
            warped_prior_state=state,
            temporal_smoothness=0.8,  # Placeholder
            confidence_metrics={"overall": 0.7},
            causal_justification="Simplified inverse explanation",
            causal_antecedents=["state_history", "field_dynamics"],
            attribution_scores={"state_history": 0.6, "field_dynamics": 0.4},
            reconstruction_confidence=0.75
        )
    
    def _get_top_attractor_region(self, routing: torch.Tensor) -> str:
        """Get the region with highest routing probability"""
        top_idx = torch.argmax(routing).item()
        return f"Region_{top_idx}"
    
    def _generate_justification(
        self,
        top_contributors: List[Tuple[str, float]],
        counterfactuals: Dict[str, float],
        field_params: FieldParameters
    ) -> str:
        """Generate textual justification"""
        lines = ["Primary factors:"]
        for feature, contribution in top_contributors:
            impact = "supporting" if contribution > 0 else "opposing"
            lines.append(f"- {feature}: {abs(contribution):.3f} ({impact})")
        
        lines.append("\nCounterfactual impacts:")
        for feature, shift in counterfactuals.items():
            lines.append(f"- Without {feature}: {shift:.3f} belief shift")
        
        lines.append(f"\nField state: stability={1-field_params.curvature:.2f}, "
                    f"certainty={1-field_params.entropy:.2f}, "
                    f"coherence={field_params.alignment:.2f}")
        
        return "\n".join(lines)
    
    def update_fields(
        self, 
        route: torch.Tensor, 
        reward: torch.Tensor, 
        sequence: List[torch.Tensor]
    ):
        """Update fields based on routing and reward"""
        # Get current field parameters
        _, _, field_params, _ = self._route_state_internal(sequence)
        
        # Calculate adaptive learning rate
        adaptive_lr = (
            self.LEARNING_RATE * 
            (1 - field_params.entropy) *           # Learn more when certain
            (1 / (1 + field_params.curvature))     # Learn less in unstable regions
        )
        
        with torch.no_grad():
            # Update vector field based on reward
            route_2d = route.view(*self.field_shape, 1)
            reward_scaled = reward.item() * adaptive_lr
            
            # Apply weighted update to vector field
            field_update = route_2d * reward_scaled
            alignment_weight = abs(field_params.alignment)
            
            self.vector_field.data = (
                self.vector_field.data * (1 - alignment_weight * self.LEARNING_RATE) +
                field_update.expand(-1, -1, self.vector_dim)
            )
            
            # Normalize vector field
            self.vector_field.data = normalize_vector_field(self.vector_field.data)
    
    def simulate_world_branches(
        self, 
        current_state: FieldParameters, 
        num_branches: int = 3
    ) -> List[WorldBranch]:
        """Simulate world branches"""
        branches = []
        for _ in range(num_branches):
            # Perturb field parameters
            perturbed_state = self._perturb_field_parameters(current_state)
            
            # Clone network (simplified - in practice would need deep copy)
            branch_network = self._clone_network()
            
            # Calculate branch probability
            branch_prob = self._calculate_branch_probability(current_state, perturbed_state)
            
            branches.append(WorldBranch(branch_network, perturbed_state, branch_prob))
        
        return branches
    
    def _perturb_field_parameters(self, state: FieldParameters) -> FieldParameters:
        """Perturb field parameters for branching"""
        return FieldParameters(
            curvature=max(0.0, state.curvature + np.random.normal(0, 0.1)),
            entropy=max(0.0, min(1.0, state.entropy + np.random.normal(0, 0.1))),
            alignment=max(-1.0, min(1.0, state.alignment + np.random.normal(0, 0.1)))
        )
    
    def _clone_network(self) -> 'SpatialProbabilityNetwork':
        """Clone network for branching (simplified)"""
        # In practice, this would need a proper deep copy
        cloned = SpatialProbabilityNetwork(
            vae=self.vae_model,
            state_dim=self.state_dim,
            field_shape=self.field_shape,
            vector_dim=self.vector_dim,
            buffer_size=self.buffer_size,
            device=self.device
        )
        cloned.load_state_dict(self.state_dict())
        return cloned
    
    def _calculate_branch_probability(
        self, 
        original: FieldParameters, 
        perturbed: FieldParameters
    ) -> float:
        """Calculate probability of branch transition"""
        distance = math.sqrt(
            (original.curvature - perturbed.curvature) ** 2 +
            (original.entropy - perturbed.entropy) ** 2 +
            (original.alignment - perturbed.alignment) ** 2
        )
        return math.exp(-distance)
    
    def analyze_field_flow(self, state: torch.Tensor, steps: int = 10) -> FlowPattern:
        """Analyze field flow patterns"""
        # If state is raw input, process it through VAE first
        if state.shape[-1] != self.state_dim:
            # This is a raw input, encode it first
            with torch.no_grad():
                mean, _ = self.vae_model.encode_sequence([state])
                current_state = mean.squeeze(0) if mean.dim() > 1 else mean
        else:
            current_state = state.clone()
        
        history = []
        
        for _ in range(steps):
            # Get routing directly from latent state
            field_flat = self.vector_field.view(-1, self.vector_dim)
            similarity = torch.matmul(current_state.unsqueeze(0), field_flat.T)
            routing = F.softmax(similarity, dim=-1)
            
            geometry = self._calculate_field_geometry(current_state, routing)
            history.append(geometry)
            
            # Update state following field flow (simplified)
            current_state = routing.view(-1)[:self.state_dim]
        
        # Analyze flow stability and patterns
        return FlowPattern(
            position=state.cpu().numpy(),
            flow_direction=history[-1].direction,
            local_curvature=np.mean([g.local_curvature for g in history]),
            local_entropy=np.mean([-math.log(max(g.strength, 1e-10)) for g in history]),
            local_alignment=np.mean([1 - abs(g.local_rotation) for g in history]),
            stability=self._calculate_flow_stability(history)
        )
    
    def _calculate_field_geometry(
        self, 
        state: torch.Tensor, 
        routing: torch.Tensor
    ) -> GeometricField:
        """Calculate geometric properties of the field"""
        # Field direction and strength
        field_flat = self.vector_field.view(-1, self.vector_dim)
        field_direction = torch.matmul(routing, field_flat)
        field_strength = float(torch.max(routing).item())
        
        # Local curvature (simplified calculation)
        local_curvature = self._calculate_local_curvature(state, routing)
        
        # Local divergence
        local_divergence = self._calculate_local_divergence(state, routing)
        
        # Local rotation
        local_rotation = self._calculate_local_rotation(state, routing)
        
        return GeometricField(
            direction=field_direction.detach().cpu().numpy(),
            strength=field_strength,
            local_curvature=float(local_curvature.item()),
            local_divergence=float(local_divergence.item()),
            local_rotation=float(local_rotation.item())
        )
    
    def _calculate_local_curvature(self, state: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        """Calculate local field curvature"""
        epsilon = 1e-5
        
        # Perturb state slightly in latent space
        perturbed_state = state + torch.randn_like(state) * epsilon
        
        # Calculate routing directly for perturbed state
        field_flat = self.vector_field.view(-1, self.vector_dim)
        similarity = torch.matmul(perturbed_state.unsqueeze(0), field_flat.T)
        perturbed_routing = F.softmax(similarity, dim=-1)
        
        # Second derivative approximation
        curvature = torch.mean((routing - perturbed_routing) ** 2) / (epsilon ** 2)
        return curvature
    
    def _calculate_local_divergence(self, state: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        """Calculate local field divergence"""
        epsilon = 1e-5
        gradients = []
        
        field_flat = self.vector_field.view(-1, self.vector_dim)
        
        for i in range(state.shape[0]):
            # Forward and backward perturbations in latent space
            delta = torch.zeros_like(state)
            delta[i] = epsilon
            
            forward_state = state + delta
            backward_state = state - delta
            
            # Calculate routing directly
            forward_sim = torch.matmul(forward_state.unsqueeze(0), field_flat.T)
            backward_sim = torch.matmul(backward_state.unsqueeze(0), field_flat.T)
            forward_routing = F.softmax(forward_sim, dim=-1)
            backward_routing = F.softmax(backward_sim, dim=-1)
            
            gradient = (forward_routing - backward_routing) / (2 * epsilon)
            gradients.append(torch.mean(gradient))
        
        return torch.mean(torch.stack(gradients))
    
    def _calculate_local_rotation(self, state: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        """Calculate local field rotation (curl-like measure)"""
        epsilon = 1e-5
        rotations = []
        
        field_flat = self.vector_field.view(-1, self.vector_dim)
        
        for i in range(min(state.shape[0] - 1, 3)):  # Limit computation
            for j in range(i + 1, min(state.shape[0], i + 4)):
                delta_i = torch.zeros_like(state)
                delta_j = torch.zeros_like(state)
                delta_i[i] = epsilon
                delta_j[j] = epsilon
                
                state_i = state + delta_i
                state_j = state + delta_j
                
                # Calculate routing directly
                sim_i = torch.matmul(state_i.unsqueeze(0), field_flat.T)
                sim_j = torch.matmul(state_j.unsqueeze(0), field_flat.T)
                routing_i = F.softmax(sim_i, dim=-1)
                routing_j = F.softmax(sim_j, dim=-1)
                
                rotation = torch.mean(routing_i - routing_j)
                rotations.append(rotation)
        
        return torch.mean(torch.stack(rotations)) if rotations else torch.tensor(0.0)
    
    def _calculate_flow_stability(self, history: List[GeometricField]) -> float:
        """Calculate flow stability over time"""
        if len(history) < 2:
            return 1.0
        
        direction_stability = 0.0
        strength_stability = 0.0
        geometric_stability = 0.0
        
        for i in range(1, len(history)):
            # Direction consistency - flatten the directions first
            dir1 = history[i].direction.flatten()
            dir2 = history[i-1].direction.flatten()
            dir_similarity = np.dot(dir1, dir2) / (
                np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-10
            )
            direction_stability += max(0, dir_similarity)
            
            # Strength consistency
            strength_stability += 1 - abs(history[i].strength - history[i-1].strength)
            
            # Geometric consistency
            geometric_stability += 1 - (
                abs(history[i].local_curvature - history[i-1].local_curvature) +
                abs(history[i].local_divergence - history[i-1].local_divergence) +
                abs(history[i].local_rotation - history[i-1].local_rotation)
            ) / 3
        
        return (direction_stability + strength_stability + geometric_stability) / (3 * (len(history) - 1))
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic information"""
        metrics = {}
        
        # Field metrics
        with torch.no_grad():
            metrics["global_entropy"] = float(torch.mean(self.entropy_field).item())
            metrics["global_curvature"] = float(torch.mean(self.curvature_field).item())
            metrics["global_alignment"] = float(torch.mean(self.alignment_field).item())
            
            # Field stability
            vector_norms = torch.norm(self.vector_field, dim=-1)
            metrics["field_strength_mean"] = float(torch.mean(vector_norms).item())
            metrics["field_strength_std"] = float(torch.std(vector_norms).item())
            metrics["belief_stability"] = 1.0 - metrics["field_strength_std"] / (metrics["field_strength_mean"] + 1e-10)
            
            # Coherence score
            field_flat = self.vector_field.view(-1, self.vector_dim)
            mean_direction = torch.mean(field_flat, dim=0)
            coherence = torch.mean(F.cosine_similarity(field_flat, mean_direction.unsqueeze(0), dim=1))
            metrics["coherence_score"] = float(coherence.item())
        
        # Branch metrics
        metrics["active_branches"] = len(self.branches)
        metrics["mean_branch_value"] = np.mean([b.value for b in self.branches]) if self.branches else 0.0
        
        # Exploration metrics
        if self.temporal_buffer:
            exploration = self._update_exploration(self.temporal_buffer[-1])
            metrics["novelty_score"] = exploration.novelty_score
            metrics["uncertainty_score"] = exploration.uncertainty_score
            metrics["exploration_rate"] = exploration.exploration_rate
        
        return metrics
    
    def clear_temporal_buffer(self):
        """Clear the temporal buffer"""
        self.temporal_buffer.clear()
    
    def update_field_parameters(self, field_params: FieldParameters):
        """Update field parameters globally"""
        with torch.no_grad():
            # Scale field magnitudes by the field parameters
            curvature_scaling = 1.0 + field_params.curvature
            entropy_scaling = field_params.entropy
            alignment_scaling = abs(field_params.alignment)
            
            # Update curvature field
            self.curvature_field.data *= curvature_scaling
            
            # Update entropy field with regularization
            target_entropy = torch.full_like(self.entropy_field, entropy_scaling)
            self.entropy_field.data = 0.8 * self.entropy_field.data + 0.2 * target_entropy
            
            # Update alignment field
            direction = 1.0 if field_params.alignment > 0 else -1.0
            self.alignment_field.data *= (direction * alignment_scaling)
            
            # Update vector field to respect new parameters
            scale_factor = 1.0 / (1.0 + curvature_scaling)
            self.vector_field.data *= scale_factor
            
            # Apply alignment influence
            if abs(field_params.alignment) > 0.1:
                alignment_influence = self.alignment_field.unsqueeze(-1) * alignment_scaling * 0.3
                self.vector_field.data += alignment_influence.expand(-1, -1, self.vector_dim)
            
            # Normalize vector field
            self.vector_field.data = normalize_vector_field(self.vector_field.data)


# ========================= INVERSE FLOW COMPONENTS =========================

class InverseFlowField(IInverseFlowField):
    """Inverse flow field for backward belief reconstruction"""
    
    def __init__(
        self,
        field_shape: Tuple[int, int],
        vector_dim: int,
        context_dim: Optional[int] = None,
        learning_rate: float = 0.01,
        device: str = 'cpu'
    ):
        self.field_shape = field_shape
        self.vector_dim = vector_dim
        self.context_dim = context_dim or vector_dim
        self.learning_rate = learning_rate
        self.device = device
        
        # Initialize inverse fields
        self._init_fields()
        
        # Warping and context modules
        self.warping_network = WarpingModule(vector_dim, vector_dim * 2, device)
        self.context_encoder = ContextEncoder(self.context_dim, vector_dim, device)
    
    def _init_fields(self):
        """Initialize inverse flow fields"""
        # Inverse vector field
        vector_field_data = torch.randn(*self.field_shape, self.vector_dim) * 0.01
        self.inverse_vector_field = normalize_vector_field(vector_field_data).to(self.device)
        
        # Flow strength field
        field_size = self.field_shape[0] * self.field_shape[1]
        self.flow_strength_field = torch.full(self.field_shape, 1.0 / field_size).to(self.device)
    
    def generate_previous_state_with_context(
        self,
        current_state: torch.Tensor,
        context: torch.Tensor,
        temporal_regularizer: TemporalRegularizer
    ) -> InverseTransformationState:
        """Generate previous state using inverse transformation"""
        current_state = current_state.to(self.device)
        context = context.to(self.device)
        
        # Apply dynamic state warping
        warped_state = self.warping_network.warp_state(current_state, context)
        
        # Get context-aware routing
        contextual_routing = self._compute_contextual_routing(warped_state, context)
        
        # Apply inverse transformation
        previous_state = self._apply_inverse_transformation(warped_state, contextual_routing)
        
        # Check temporal smoothness
        smoothness, confidence = temporal_regularizer.analyze_transition(
            previous_state.detach(), current_state.detach()
        )
        
        # Calculate confidence metrics
        metrics = {
            "warping_confidence": self._calculate_warping_confidence(warped_state),
            "routing_confidence": self._calculate_routing_confidence(contextual_routing),
            "temporal_confidence": confidence,
            "overall_confidence": self._combine_confidences(warped_state, contextual_routing, confidence)
        }
        
        # Extract flow direction
        flow_direction = self._calculate_flow_direction(previous_state, current_state)
        
        return InverseTransformationState(
            warped_state=warped_state.detach(),
            contextual_routing=contextual_routing.detach(),
            temporal_smoothness=smoothness,
            confidence_metrics=metrics,
            flow_direction=flow_direction
        )
    
    def _compute_contextual_routing(self, warped_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute context-aware routing"""
        # Encode context
        encoded_context = self.context_encoder.encode(context)
        
        # Ensure proper tensor dimensions
        if warped_state.dim() == 1:
            warped_state = warped_state.unsqueeze(0)
        if encoded_context.dim() == 1:
            encoded_context = encoded_context.unsqueeze(0)
            
        # Weight inverse vectors by context-aware attention
        field_flat = self.inverse_vector_field.view(-1, self.vector_dim)  # (256, 8)
        
        # Create context-aware field scores
        context_state_combined = (warped_state + encoded_context) / 2  # Simple combination
        
        # Project combined state to field attention weights
        field_attention_scores = torch.matmul(field_flat, context_state_combined.T)  # (256, 1)
        field_attention = F.softmax(field_attention_scores.squeeze(-1), dim=0)  # (256,)
        
        # Weight field vectors by attention
        routing = torch.matmul(field_attention.unsqueeze(0), field_flat)  # (1, 8)
        
        # Return with proper dimensions
        return routing.squeeze(0) if routing.dim() > 1 else routing
    
    def _apply_inverse_transformation(
        self, 
        warped_state: torch.Tensor, 
        contextual_routing: torch.Tensor
    ) -> torch.Tensor:
        """Apply inverse transformation"""
        # Combine warped state with routing
        combined = warped_state * contextual_routing
        
        # Apply non-linear transformation to reconstruct previous state
        transformed = F.leaky_relu(combined)
        
        # The output should have the same dimensions as the input state
        # Use a linear projection to maintain dimensionality
        return transformed
    
    def update_from_forward_field(self, forward_field: torch.Tensor, forward_routing: torch.Tensor):
        """Update inverse field from forward dynamics"""
        with torch.no_grad():
            # Compute inverse vectors from forward field
            inverse_update = -forward_field  # Negate for inverse direction
            inverse_update = normalize_vector_field(inverse_update)
            
            # Update inverse vector field
            self.inverse_vector_field += inverse_update * self.learning_rate
            self.inverse_vector_field = normalize_vector_field(self.inverse_vector_field)
            
            # Update flow strength based on forward routing
            momentum = 0.9
            self.flow_strength_field = (momentum * self.flow_strength_field + 
                                      0.1 * forward_routing.view(*self.field_shape))
    
    def calculate_metrics(self) -> FieldMetrics:
        """Calculate field metrics"""
        with torch.no_grad():
            # Average flow strength
            avg_strength = float(torch.mean(self.flow_strength_field).item())
            
            # Directional coherence
            field_flat = self.inverse_vector_field.view(-1, self.vector_dim)
            mean_vector = torch.mean(field_flat, dim=0)
            coherence = float(torch.mean(F.cosine_similarity(field_flat, mean_vector.unsqueeze(0), dim=1)).item())
            
            # Field statistics
            vector_norms = torch.norm(self.inverse_vector_field, dim=-1)
            field_stats = {
                "max_magnitude": float(torch.max(vector_norms).item()),
                "min_magnitude": float(torch.min(vector_norms).item()),
                "mean_magnitude": float(torch.mean(vector_norms).item()),
                "magnitude_std": float(torch.std(vector_norms).item())
            }
            field_stats["confidence"] = 1.0 - field_stats["magnitude_std"] / (field_stats["mean_magnitude"] + 1e-10)
            
            return FieldMetrics(
                average_flow_strength=avg_strength,
                directional_coherence=coherence,
                backward_confidence=field_stats["confidence"],
                field_statistics=field_stats
            )
    
    def _calculate_warping_confidence(self, warped_state: torch.Tensor) -> float:
        """Calculate warping confidence"""
        variance = torch.var(warped_state)
        return float((1.0 - variance).item())
    
    def _calculate_routing_confidence(self, routing: torch.Tensor) -> float:
        """Calculate routing confidence"""
        entropy = -torch.sum(routing * torch.log(routing + 1e-10))
        return float((1.0 - entropy).item())
    
    def _combine_confidences(
        self, 
        warped_state: torch.Tensor, 
        routing: torch.Tensor, 
        temporal_confidence: float
    ) -> float:
        """Combine different confidence measures"""
        warping_conf = self._calculate_warping_confidence(warped_state)
        routing_conf = self._calculate_routing_confidence(routing)
        
        return 0.4 * warping_conf + 0.4 * routing_conf + 0.2 * temporal_confidence
    
    def _calculate_flow_direction(
        self, 
        previous_state: torch.Tensor, 
        current_state: torch.Tensor
    ) -> np.ndarray:
        """Calculate flow direction"""
        flow = current_state - previous_state
        norm = torch.norm(flow)
        if norm > 1e-6:
            flow = flow / norm
        return flow.detach().cpu().numpy()


class WarpingModule(nn.Module):
    """Neural module for state warping"""
    
    def __init__(self, input_dim: int, hidden_dim: int, device: str = 'cpu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.to(device)
    
    def warp_state(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Warp state using context"""
        # Non-linear transformation with attention
        hidden = F.leaky_relu(self.fc1(state))
        
        # Apply context-aware attention
        attention_features = self.attention(hidden)
        
        # Ensure proper tensor dimensions for attention
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if attention_features.dim() == 1:
            attention_features = attention_features.unsqueeze(0)
            
        # Ensure context dimensions match for attention
        if context.shape[-1] != attention_features.shape[-1]:
            context_padded = F.pad(context, (0, attention_features.shape[-1] - context.shape[-1]))
        else:
            context_padded = context
        
        # Compute attention scores (batch_size x batch_size)
        attention_scores = torch.matmul(attention_features, context_padded.T)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attention_output = torch.matmul(attention_weights, context_padded).squeeze(0)
        
        # Combine and project back
        if hidden.dim() > 1:
            hidden = hidden.squeeze(0)
        combined = hidden + attention_output
        return torch.tanh(self.fc2(combined))


class ContextEncoder(nn.Module):
    """Context encoding module"""
    
    def __init__(self, context_dim: int, output_dim: int, device: str = 'cpu'):
        super().__init__()
        self.encoder = nn.Linear(context_dim, output_dim * 2)
        self.projector = nn.Linear(output_dim * 2, output_dim)
        self.to(device)
    
    def encode(self, context: torch.Tensor) -> torch.Tensor:
        """Encode context"""
        hidden = F.leaky_relu(self.encoder(context))
        return torch.tanh(self.projector(hidden))