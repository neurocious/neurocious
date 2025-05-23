import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from collections import defaultdict

from neurocious_core import (
    IEnhancedVAE, IAttentionBlock, IFieldRegularizer, 
    FieldParameters, AttentionBlock, FieldRegularizer,
    xavier_init, he_init, reparameterize
)

class EnhancedVAE(nn.Module, IEnhancedVAE):
    """Enhanced Variational Autoencoder with field parameter extraction"""
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        sequence_length: int = 16,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        num_attention_blocks: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Sequential encoder components
        self.encoder_input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention blocks
        self.encoder_attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, dropout_rate)
            for _ in range(num_attention_blocks)
        ])
        
        self.encoder_layer_norm = nn.LayerNorm(hidden_dim)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder components with field outputs
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        self.decoder_field_output = nn.Linear(hidden_dim, 3)  # curvature, entropy, alignment
        
        # Field regularizer
        self.field_regularizer = FieldRegularizer()
        
        # KL annealing
        self.kl_weight = 0.0
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    @property
    def latent_dimension(self) -> int:
        return self.latent_dim
    
    def encode_sequence(self, sequence: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence to latent distribution parameters"""
        # Convert list to tensor and project to hidden dimension
        # sequence: List[Tensor] -> Tensor of shape [batch_size, seq_len, input_dim]
        sequence_tensor = torch.stack(sequence, dim=1)  # [batch, seq_len, input_dim]
        batch_size, seq_len, _ = sequence_tensor.shape
        
        # Project each input to hidden dimension
        projected = F.leaky_relu(self.encoder_input_proj(sequence_tensor))
        
        # Apply multi-head attention blocks
        processed = projected
        for attention_block in self.encoder_attention_blocks:
            # Reshape for attention: [batch * seq_len, 1, hidden_dim]
            reshaped = processed.view(-1, 1, self.hidden_dim)
            attended = attention_block(reshaped, training=self.training)
            processed = attended.view(batch_size, seq_len, self.hidden_dim)
        
        # Layer normalization and mean pooling
        normalized = self.encoder_layer_norm(processed)
        pooled = torch.mean(normalized, dim=1)  # [batch, hidden_dim]
        
        # Project to latent parameters
        mean = self.encoder_mean(pooled)
        log_var = self.encoder_log_var(pooled)
        
        return mean, log_var
    
    def decode_with_field(self, latent_vector: torch.Tensor) -> Tuple[torch.Tensor, FieldParameters]:
        """Decode latent vector to reconstruction and field parameters"""
        hidden1 = F.leaky_relu(self.decoder_fc1(latent_vector))
        hidden2 = F.leaky_relu(self.decoder_fc2(hidden1))
        
        # Regular reconstruction
        reconstruction = torch.sigmoid(self.decoder_output(hidden2))
        
        # Field parameters with specific activation functions
        raw_field_params = self.decoder_field_output(hidden2)
        
        # Apply specific activations for each field parameter
        curvature = F.relu(raw_field_params[:, 0])  # Non-negative
        entropy = torch.sigmoid(raw_field_params[:, 1])  # [0, 1]
        alignment = torch.tanh(raw_field_params[:, 2])  # [-1, 1]
        
        # Create field parameters (assuming batch_size = 1 for now)
        if latent_vector.shape[0] == 1:
            field_params = FieldParameters(
                curvature=float(curvature[0].item()),
                entropy=float(entropy[0].item()),
                alignment=float(alignment[0].item())
            )
        else:
            # For batch processing, return mean field parameters
            field_params = FieldParameters(
                curvature=float(torch.mean(curvature).item()),
                entropy=float(torch.mean(entropy).item()),
                alignment=float(torch.mean(alignment).item())
            )
        
        return reconstruction, field_params
    
    def forward_sequence(
        self, 
        sequence: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, FieldParameters, torch.Tensor, torch.Tensor]:
        """Full forward pass through VAE"""
        # Encode sequence to latent distribution
        mean, log_var = self.encode_sequence(sequence)
        
        # Sample latent vector using reparameterization trick
        latent_vector = reparameterize(mean, log_var)
        
        # Decode with field parameters
        reconstruction, field_params = self.decode_with_field(latent_vector)
        
        return reconstruction, field_params, mean, log_var
    
    def estimate_latent_curvature(self, z: torch.Tensor) -> float:
        """Estimate local curvature in latent space using second derivatives"""
        z = z.clone().detach().requires_grad_(True)
        epsilon = 1e-5
        
        curvatures = []
        for i in range(z.shape[-1]):
            # Create perturbations
            z_plus = z.clone()
            z_minus = z.clone()
            z_plus[..., i] += epsilon
            z_minus[..., i] -= epsilon
            
            # Get reconstructions
            recon_plus, _ = self.decode_with_field(z_plus)
            recon_minus, _ = self.decode_with_field(z_minus)
            recon_center, _ = self.decode_with_field(z)
            
            # Second derivative approximation
            second_deriv = (recon_plus.mean() - 2 * recon_center.mean() + recon_minus.mean()) / (epsilon ** 2)
            curvatures.append(abs(float(second_deriv.item())))
        
        return sum(curvatures) / len(curvatures)
    
    def estimate_latent_entropy(self, z: torch.Tensor) -> float:
        """Estimate entropy using local neighborhood sampling"""
        num_samples = 100
        radius = 0.1
        
        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample in neighborhood
                noise = torch.randn_like(z) * radius
                neighbor_z = z + noise
                
                recon, _ = self.decode_with_field(neighbor_z)
                samples.append(float(recon.mean().item()))
        
        # Compute entropy using histogram method
        samples_array = np.array(samples)
        hist, _ = np.histogram(samples_array, bins=20, density=True)
        
        # Calculate entropy
        entropy = 0.0
        for count in hist:
            if count > 0:
                p = count / len(hist)
                entropy -= p * np.log(p)
        
        return entropy
    
    def extract_field_parameters(self, state: torch.Tensor) -> FieldParameters:
        """Extract field parameters from state"""
        # Project state through field output layer
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Use decoder field output to extract parameters
        hidden1 = F.leaky_relu(self.decoder_fc1(state))
        hidden2 = F.leaky_relu(self.decoder_fc2(hidden1))
        raw_field_params = self.decoder_field_output(hidden2)
        
        # Apply activations
        curvature = float(F.relu(raw_field_params[0, 0]).item())
        entropy = float(torch.sigmoid(raw_field_params[0, 1]).item())
        alignment = float(torch.tanh(raw_field_params[0, 2]).item())
        
        return FieldParameters(curvature=curvature, entropy=entropy, alignment=alignment)
    
    def compute_loss(
        self,
        inputs: List[torch.Tensor],
        reconstruction: torch.Tensor,
        field_params: FieldParameters,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        epoch: int
    ) -> torch.Tensor:
        """Compute total VAE loss"""
        # Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(inputs, reconstruction)
        
        # KL divergence loss with annealing
        kl_loss = self._compute_kl_loss(mean, log_var)
        
        # Anneal KL weight from 0 to 1 over epochs
        self.kl_weight = min(1.0, epoch / 50.0)
        
        # Field regularization
        field_reg_loss = self.field_regularizer.compute_loss(field_params)
        
        # Contrastive loss for regime transitions
        contrastive_loss = self._compute_contrastive_loss(mean, inputs)
        
        # Combine losses
        total_loss = (recon_loss + 
                     self.kl_weight * kl_loss + 
                     0.1 * field_reg_loss + 
                     0.05 * contrastive_loss)
        
        return total_loss
    
    def _compute_reconstruction_loss(
        self, 
        inputs: List[torch.Tensor], 
        reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss (BCE)"""
        # Convert inputs to tensor if needed
        if isinstance(inputs, list):
            target = torch.stack(inputs, dim=0).mean(dim=0)  # Average over sequence
        else:
            target = inputs
        
        # Binary cross entropy loss
        return F.binary_cross_entropy(reconstruction, target, reduction='mean')
    
    def _compute_kl_loss(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss"""
        # KL divergence between approximate posterior and standard normal prior
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        return torch.mean(kl)
    
    def _compute_contrastive_loss(
        self, 
        latent_mean: torch.Tensor, 
        inputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute contrastive loss using cosine similarity"""
        if len(inputs) < 2:
            return torch.tensor(0.0, device=latent_mean.device)
        
        similarities = []
        input_tensor = torch.stack(inputs, dim=0)
        
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                # Calculate similarity in latent space
                sim = F.cosine_similarity(
                    latent_mean.unsqueeze(0), 
                    latent_mean.unsqueeze(0), 
                    dim=-1
                )
                similarities.append(sim)
        
        if similarities:
            return torch.mean(torch.stack(similarities))
        else:
            return torch.tensor(0.0, device=latent_mean.device)
    
    def train_step(
        self,
        sequence_data: List[List[torch.Tensor]],
        epoch: int,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Single training step"""
        total_loss = 0.0
        
        for sequence in sequence_data:
            # Convert sequence to tensors if needed
            sequence_tensors = [s if isinstance(s, torch.Tensor) else torch.tensor(s) 
                              for s in sequence]
            
            # Forward pass
            reconstruction, field_params, mean, log_var = self.forward_sequence(sequence_tensors)
            
            # Compute loss
            loss = self.compute_loss(sequence_tensors, reconstruction, field_params, mean, log_var, epoch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(sequence_data)
    
    def train_model(
        self,
        sequence_data: List[List[torch.Tensor]],
        epochs: int,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """Train the VAE model"""
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        
        for epoch in range(epochs):
            # Process in batches
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(sequence_data), batch_size):
                batch = sequence_data[i:i + batch_size]
                batch_loss = self.train_step(batch, epoch, optimizer)
                total_loss += batch_loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}, KL Weight: {self.kl_weight:.3f}")


# ========================= FIELD-AWARE KL DIVERGENCE =========================

class FieldAwareKLDivergence(IFieldAwareKLDivergence):
    """Field-aware KL divergence calculation"""
    
    def __init__(self, spn: 'SpatialProbabilityNetwork'):
        self.spn = spn
    
    def calculate_kl(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        latent_state: torch.Tensor
    ) -> torch.Tensor:
        """Calculate field-aware KL divergence"""
        # Get field-based prior parameters
        mu_field, sigma_field_squared = self._get_field_based_prior(latent_state)
        
        # Calculate field-aware KL divergence
        # KL(q(z|x) || N(mu_field, sigma_field))
        deviation = mean - mu_field
        normalized_deviation = (deviation ** 2) / sigma_field_squared
        
        variance_ratio = torch.exp(log_var) / sigma_field_squared
        log_sigma_ratio = torch.log(sigma_field_squared) - log_var
        
        # Combine terms:
        # 0.5 * Sum((exp(log_var) + (mean - mu_field)²)/sigma_field² - 1 - log_var + log(sigma_field²))
        kl = 0.5 * torch.sum(
            variance_ratio + normalized_deviation - 1 - log_var + log_sigma_ratio,
            dim=-1
        )
        
        return torch.mean(kl)
    
    def _get_field_based_prior(
        self, 
        latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get field-based prior parameters"""
        # Get expected direction from vector field
        field_direction = torch.matmul(latent_state, self.spn.vector_field.transpose(-2, -1))
        mu_field = field_direction
        
        # Calculate sigma based on field uncertainty
        routing, _, field_params = self.spn._route_state_internal([latent_state])
        
        # Base variance on field entropy and curvature
        base_variance = 1.0 + field_params.entropy
        curvature_scaling = 1.0 + field_params.curvature
        
        sigma_field_squared = torch.full_like(
            latent_state, 
            base_variance * curvature_scaling
        )
        
        return mu_field, sigma_field_squared
    
    def analyze_kl_contribution(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        latent_state: torch.Tensor
    ) -> FieldKLMetrics:
        """Analyze KL contribution components"""
        mu_field, sigma_field_squared = self._get_field_based_prior(latent_state)
        
        # Calculate standard KL (against N(0,1))
        standard_kl = self._calculate_standard_kl(mean, log_var)
        
        # Calculate field-based KL
        field_kl = self.calculate_kl(mean, log_var, latent_state)
        
        # Calculate alignment between mean and field direction
        alignment_score = self._calculate_field_alignment(mean, mu_field)
        
        # Calculate uncertainty adaptation
        uncertainty_score = self._calculate_uncertainty_adaptation(log_var, sigma_field_squared)
        
        detailed_metrics = {
            "mean_field_deviation": float(torch.mean((mean - mu_field) ** 2).item()),
            "variance_adaptation": uncertainty_score,
            "field_kl": float(field_kl.item()),
            "kl_reduction": float((standard_kl - field_kl).item())
        }
        
        return FieldKLMetrics(
            base_kl=float(standard_kl.item()),
            field_alignment_score=alignment_score,
            uncertainty_adaptation=uncertainty_score,
            detailed_metrics=detailed_metrics
        )
    
    def _calculate_standard_kl(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Calculate standard KL divergence against N(0,1)"""
        kl = -0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim=-1)
        return torch.mean(kl)
    
    def _calculate_field_alignment(self, mean: torch.Tensor, field_mean: torch.Tensor) -> float:
        """Calculate alignment between mean and field direction"""
        cosine_sim = F.cosine_similarity(mean, field_mean, dim=-1)
        return float(torch.mean(cosine_sim).item())
    
    def _calculate_uncertainty_adaptation(
        self, 
        log_var: torch.Tensor, 
        field_sigma_squared: torch.Tensor
    ) -> float:
        """Calculate uncertainty adaptation score"""
        predicted_var = torch.exp(log_var)
        ratio = predicted_var / field_sigma_squared
        adaptation = 1.0 - torch.abs(1.0 - ratio)
        return float(torch.mean(adaptation).item())


# ========================= POLICY, REFLEX, AND PREDICTION NETWORKS =========================

class PolicyNetwork(nn.Module):
    """Policy network for action prediction"""
    
    def __init__(self, state_dim: int, action_dim: int = 10):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, 64)
        self.history_encoder = nn.Linear(state_dim, 64)
        self.attention = nn.Linear(64, 64)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        current_state: torch.Tensor, 
        history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network"""
        # Encode current state and history
        state_encoded = F.leaky_relu(self.state_encoder(current_state))
        history_encoded = F.leaky_relu(self.history_encoder(history))
        
        # Attention mechanism
        attention_weights = F.softmax(
            torch.matmul(
                torch.matmul(state_encoded, self.attention.weight),
                history_encoded.transpose(-2, -1)
            ),
            dim=-1
        )
        attention_output = torch.matmul(attention_weights, history_encoded)
        
        # Concatenate features
        combined = torch.cat([state_encoded, attention_output], dim=-1)
        
        # Policy and value outputs
        policy = torch.sigmoid(self.policy_head(combined))
        value = self.value_head(combined)
        
        return policy, value


class ReflexNetwork(nn.Module):
    """Reflex network for immediate responses"""
    
    def __init__(self, state_dim: int, reflex_dim: int = 5):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, reflex_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through reflex network"""
        x = F.leaky_relu(self.layer1(state))
        x = F.leaky_relu(self.layer2(x))
        return torch.sigmoid(self.output_layer(x))


class PredictionNetwork(nn.Module):
    """Prediction network for future state estimation"""
    
    def __init__(self, state_dim: int, prediction_dim: int = 4):
        super().__init__()
        sequence_dim = state_dim * 10  # Assuming 10-step history
        self.sequence_encoder = nn.Linear(sequence_dim, 64)
        self.hidden = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, prediction_dim)  # [value, confidence, upper, lower]
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through prediction network"""
        x = F.leaky_relu(self.sequence_encoder(sequence))
        x = F.leaky_relu(self.hidden(x))
        return self.output_layer(x)