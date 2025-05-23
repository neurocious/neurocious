import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import math
import random
from abc import ABC, abstractmethod

# ========================= CORE DATA STRUCTURES =========================

@dataclass
class FieldParameters:
    """Field parameters for spatial probability networks"""
    curvature: float = 0.0  # Non-negative, measures regime instability
    entropy: float = 0.0    # [0,1], measures narrative uncertainty
    alignment: float = 0.0  # [-1,1], measures directional coherence
    
    def __post_init__(self):
        self.curvature = max(0.0, self.curvature)
        self.entropy = max(0.0, min(1.0, self.entropy))
        self.alignment = max(-1.0, min(1.0, self.alignment))
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.curvature, self.entropy, self.alignment])
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'FieldParameters':
        return cls(
            curvature=float(torch.relu(tensor[0])),
            entropy=float(torch.sigmoid(tensor[1])),
            alignment=float(torch.tanh(tensor[2]))
        )

@dataclass
class ExplorationState:
    """State for exploration mechanics"""
    novelty_score: float = 0.0
    uncertainty_score: float = 0.0
    exploration_rate: float = 0.0

@dataclass
class FlowPattern:
    """Flow pattern analysis results"""
    position: np.ndarray
    flow_direction: np.ndarray
    local_curvature: float
    local_entropy: float
    local_alignment: float
    stability: float

@dataclass
class GeometricField:
    """Geometric field properties"""
    direction: np.ndarray
    strength: float
    local_curvature: float
    local_divergence: float
    local_rotation: float

@dataclass
class FieldMetrics:
    """Field analysis metrics"""
    average_flow_strength: float
    directional_coherence: float
    backward_confidence: float
    field_statistics: Dict[str, float]

@dataclass
class InverseTransformationState:
    """State from inverse transformation"""
    warped_state: torch.Tensor
    contextual_routing: torch.Tensor
    temporal_smoothness: float
    confidence_metrics: Dict[str, float]
    flow_direction: np.ndarray

@dataclass
class BeliefExplanation:
    """Explanation of belief formation"""
    belief_label: str
    feature_contributions: Dict[str, float]
    confidence: float
    field_params: FieldParameters
    justification: str
    top_contributing_features: List[str]
    counterfactual_shifts: Dict[str, float]
    trajectory_path: List[np.ndarray]

@dataclass
class BeliefReconstructionExplanation:
    """Explanation of belief reconstruction"""
    warped_prior_state: torch.Tensor
    temporal_smoothness: float
    confidence_metrics: Dict[str, float]
    causal_justification: str
    causal_antecedents: List[str]
    attribution_scores: Dict[str, float]
    reconstruction_confidence: float
    
    def generate_detailed_explanation(self) -> str:
        """Generate detailed explanation text"""
        lines = [
            "Belief Reconstruction Analysis:",
            f"- Temporal Smoothness: {self.temporal_smoothness:.3f}",
            f"- Reconstruction Confidence: {self.reconstruction_confidence:.3f}"
        ]
        
        if self.causal_antecedents:
            lines.append("\nCausal Antecedents:")
            for antecedent in self.causal_antecedents:
                score = self.attribution_scores.get(antecedent, 0)
                lines.append(f"- {antecedent} (strength: {score:.3f})")
        
        if self.confidence_metrics:
            lines.append("\nConfidence Metrics:")
            for metric, value in self.confidence_metrics.items():
                lines.append(f"- {metric}: {value:.3f}")
        
        if self.causal_justification:
            lines.append(f"\nJustification:\n{self.causal_justification}")
        
        return "\n".join(lines)

@dataclass
class CoTrainingConfig:
    """Configuration for co-training"""
    # Original weights
    beta: float = 0.7          # KL weight
    gamma: float = 0.5         # Narrative continuity weight
    delta: float = 0.3         # Field alignment weight
    eta: float = 1.0           # SPN loss weight
    
    # Epistemic training weights
    policy_weight: float = 0.5     # Policy prediction weight
    reflex_weight: float = 0.3     # Reflex behavior weight
    prediction_weight: float = 0.4 # Future prediction weight
    
    # Uncertainty scaling factors
    entropy_scaling: float = 2.0   # How much entropy affects variance
    curvature_scaling: float = 1.5 # How much curvature affects variance
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001

@dataclass
class FieldKLMetrics:
    """Metrics for field-aware KL divergence"""
    base_kl: float
    field_alignment_score: float
    uncertainty_adaptation: float
    detailed_metrics: Dict[str, float]

# ========================= UTILITY FUNCTIONS =========================

def xavier_init(shape: Tuple[int, ...]) -> torch.Tensor:
    """Xavier weight initialization"""
    if len(shape) == 2:
        fan_in, fan_out = shape
        std = math.sqrt(2.0 / (fan_in + fan_out))
    else:
        std = 0.02
    return torch.randn(shape) * std

def he_init(shape: Tuple[int, ...]) -> torch.Tensor:
    """He weight initialization for ReLU networks"""
    if len(shape) == 2:
        fan_in = shape[0]
        std = math.sqrt(2.0 / fan_in)
    else:
        std = 0.02
    return torch.randn(shape) * std

def normalize_vector_field(field: torch.Tensor) -> torch.Tensor:
    """Normalize vector field to unit vectors"""
    # field shape: [height, width, vector_dim]
    norm = torch.norm(field, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-6)  # Prevent division by zero
    return field / norm

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Calculate cosine similarity between vectors"""
    return F.cosine_similarity(v1, v2, dim=-1)

def reparameterize(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick for VAE"""
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + eps * std

# ========================= ABSTRACT INTERFACES =========================

class IWorldBranch(ABC):
    """Interface for world branches"""
    
    @abstractmethod
    def update_value(self, new_value: float):
        pass

class IAttentionBlock(ABC):
    """Interface for attention blocks"""
    
    @abstractmethod
    def forward(self, input_tensor: torch.Tensor, training: bool = True) -> torch.Tensor:
        pass

class IFieldRegularizer(ABC):
    """Interface for field regularizers"""
    
    @abstractmethod
    def compute_loss(self, field_params: FieldParameters) -> torch.Tensor:
        pass

class IEnhancedVAE(ABC):
    """Interface for enhanced VAE"""
    
    @property
    @abstractmethod
    def latent_dimension(self) -> int:
        pass
    
    @abstractmethod
    def encode_sequence(self, sequence: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    @abstractmethod
    def decode_with_field(self, latent_vector: torch.Tensor) -> Tuple[torch.Tensor, FieldParameters]:
        pass
    
    @abstractmethod
    def forward_sequence(self, sequence: List[torch.Tensor]) -> Tuple[torch.Tensor, FieldParameters, torch.Tensor, torch.Tensor]:
        pass
    
    @abstractmethod
    def estimate_latent_curvature(self, z: torch.Tensor) -> float:
        pass
    
    @abstractmethod
    def estimate_latent_entropy(self, z: torch.Tensor) -> float:
        pass
    
    @abstractmethod
    def extract_field_parameters(self, state: torch.Tensor) -> FieldParameters:
        pass

class IInverseFlowField(ABC):
    """Interface for inverse flow fields"""
    
    @abstractmethod
    def generate_previous_state_with_context(
        self, 
        current_state: torch.Tensor,
        context: torch.Tensor,
        temporal_regularizer: 'TemporalRegularizer'
    ) -> InverseTransformationState:
        pass
    
    @abstractmethod
    def update_from_forward_field(self, forward_field: torch.Tensor, forward_routing: torch.Tensor):
        pass
    
    @abstractmethod
    def calculate_metrics(self) -> FieldMetrics:
        pass

class IInverseFlowIntegration(ABC):
    """Interface for inverse flow integration"""
    
    @abstractmethod
    def reconstruct_prior_belief(
        self,
        current_state: torch.Tensor,
        context_state: torch.Tensor,
        potential_antecedents: Optional[List[str]] = None
    ) -> BeliefReconstructionExplanation:
        pass
    
    @abstractmethod
    def update_from_forward_dynamics(self, forward_field: torch.Tensor, forward_routing: torch.Tensor):
        pass
    
    @abstractmethod
    def add_to_temporal_buffer(self, state: torch.Tensor):
        pass

class ISpatialProbabilityNetwork(ABC):
    """Interface for spatial probability networks"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def update_fields(self, route: torch.Tensor, reward: torch.Tensor, sequence: List[torch.Tensor]):
        pass
    
    @abstractmethod
    def simulate_world_branches(self, current_state: FieldParameters, num_branches: int = 3) -> List['WorldBranch']:
        pass
    
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def analyze_field_flow(self, state: torch.Tensor, steps: int = 10) -> FlowPattern:
        pass

class IFieldAwareKLDivergence(ABC):
    """Interface for field-aware KL divergence"""
    
    @abstractmethod
    def calculate_kl(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        latent_state: torch.Tensor
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def analyze_kl_contribution(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        latent_state: torch.Tensor
    ) -> FieldKLMetrics:
        pass

# ========================= TEMPORAL COMPONENTS =========================

class TemporalRegularizer:
    """Temporal regularization for smooth transitions"""
    
    def __init__(self, history_length: int = 10, smoothness_threshold: float = 0.5):
        self.history_length = history_length
        self.smoothness_threshold = smoothness_threshold
        self.state_history = deque(maxlen=history_length)
    
    def analyze_transition(
        self, 
        current_state: torch.Tensor, 
        previous_state: torch.Tensor
    ) -> Tuple[float, float]:
        """Analyze transition smoothness and confidence"""
        # Add to history
        self.state_history.append(current_state.detach().clone())
        
        # Calculate transition magnitude
        transition_magnitude = self._calculate_transition_magnitude(current_state, previous_state)
        historical_average = self._calculate_historical_average()
        
        smoothness = 1.0 / (1.0 + abs(transition_magnitude - historical_average))
        
        # Calculate confidence based on smoothness
        confidence = smoothness if smoothness > self.smoothness_threshold else \
                    smoothness * (smoothness / self.smoothness_threshold)
        
        return smoothness, confidence
    
    def _calculate_transition_magnitude(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        """Calculate magnitude of transition between states"""
        diff = current - previous
        return float(torch.norm(diff).item())
    
    def _calculate_historical_average(self) -> float:
        """Calculate historical average of transition magnitudes"""
        if len(self.state_history) < 2:
            return 0.0
        
        magnitudes = []
        states = list(self.state_history)
        for i in range(1, len(states)):
            magnitude = self._calculate_transition_magnitude(states[i], states[i-1])
            magnitudes.append(magnitude)
        
        return sum(magnitudes) / len(magnitudes) if magnitudes else 0.0

# ========================= BASIC NEURAL COMPONENTS =========================

class AttentionBlock(nn.Module, IAttentionBlock):
    """Multi-head attention block"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_tensor: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass through attention block"""
        batch_size, seq_len, _ = input_tensor.shape
        
        # Project to Q, K, V
        Q = self.query_proj(input_tensor)
        K = self.key_proj(input_tensor)
        V = self.value_proj(input_tensor)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if training
        if training and seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_tensor.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        if training:
            attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        output = self.output_proj(attention_output)
        
        # Add residual connection and layer norm
        return self.layer_norm(output + input_tensor)

class FieldRegularizer(nn.Module, IFieldRegularizer):
    """Field parameter regularization"""
    
    def __init__(self):
        super().__init__()
    
    def compute_loss(self, field_params: FieldParameters) -> torch.Tensor:
        """Compute regularization loss for field parameters"""
        losses = []
        
        # Curvature smoothness
        losses.append(field_params.curvature ** 2 * 0.1)
        
        # Entropy bounds
        losses.append(max(0, field_params.entropy - 1) * 10)
        losses.append(max(0, -field_params.entropy) * 10)
        
        # Alignment regularization
        losses.append(field_params.alignment ** 2 * 0.05)
        
        return torch.tensor(sum(losses) / len(losses))

# ========================= WORLD BRANCHING =========================

class WorldBranch(IWorldBranch):
    """World branch for simulation"""
    
    BRANCH_DECAY_RATE = 0.95
    
    def __init__(self, network: 'SpatialProbabilityNetwork', state: FieldParameters, probability: float):
        self.network = network
        self.initial_state = state
        self.probability = probability
        self.value = 0.0
    
    def update_value(self, new_value: float):
        """Update branch value with decay"""
        self.value = new_value
        self.probability *= self.BRANCH_DECAY_RATE