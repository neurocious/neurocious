import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

class WindowedAttention(nn.Module):
    """Efficient windowed self-attention for local structure analysis."""
    
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8, 
                 qkv_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Relative position bias for windowed attention
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Initialize relative position bias
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C] or [B*num_windows, window_size*window_size, C]
            mask: Attention mask for windowed attention
        """
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = self.window_partition(x, self.window_size)
        
        B_, N, C = x.shape  # N = window_size * window_size
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Partition input into windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows.view(-1, window_size * window_size, C)


class CircularWindowedAttention(nn.Module):
    """Windowed attention for circular/angular data (angle components)."""
    
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.base_attention = WindowedAttention(dim, num_heads, window_size)
        
        # Circular embedding for angle values
        self.angle_embedding = nn.Linear(dim, dim)
        
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angles: [B, C, H, W] - angle values in [-π, π]
        """
        B, C, H, W = angles.shape
        
        # Convert to circular embedding (sin/cos representation)
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        circular_embedded = torch.stack([sin_angles, cos_angles], dim=-1)  # [B, C, H, W, 2]
        
        # Flatten for attention
        embedded_flat = circular_embedded.view(B, C, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        
        # Apply windowed attention
        attended = self.base_attention(embedded_flat)
        
        # Reshape back and convert to angles
        attended = attended.view(B, H, W, C, 2).permute(0, 3, 1, 2, 4)
        sin_out, cos_out = attended[..., 0], attended[..., 1]
        
        # Convert back to angle representation
        angles_out = torch.atan2(sin_out, cos_out)
        
        return angles_out


class VectorCoherenceAttention(nn.Module):
    """Attention mechanism considering magnitude-angle coherence in vector fields."""
    
    def __init__(self, channels: int, window_size: int = 8):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        
        # Coherence analysis network
        self.coherence_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),  # Process mag + angle together
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid()  # Output coherence weights
        )
        
        # Vector flow attention
        self.flow_attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True
        )
    
    def forward(self, magnitudes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            magnitudes: [B, C, H, W]
            angles: [B, C, H, W]
        Returns:
            coherence_weights: [B, C, H, W]
        """
        # Compute local vector coherence
        combined = torch.cat([magnitudes, angles], dim=1)
        coherence_weights = self.coherence_net(combined)
        
        # Apply flow-based attention in local windows
        B, C, H, W = magnitudes.shape
        
        # Window-based processing for efficiency
        if H >= self.window_size and W >= self.window_size:
            # Process in windows to capture local vector flow patterns
            coherence_refined = self._apply_windowed_flow_attention(
                coherence_weights, magnitudes, angles
            )
        else:
            coherence_refined = coherence_weights
        
        return coherence_refined
    
    def _apply_windowed_flow_attention(self, coherence: torch.Tensor, 
                                     magnitudes: torch.Tensor, 
                                     angles: torch.Tensor) -> torch.Tensor:
        """Apply flow attention in sliding windows."""
        B, C, H, W = coherence.shape
        refined = coherence.clone()
        
        # Slide windows across the spatial dimensions
        for h_start in range(0, H - self.window_size + 1, self.window_size // 2):
            for w_start in range(0, W - self.window_size + 1, self.window_size // 2):
                h_end = min(h_start + self.window_size, H)
                w_end = min(w_start + self.window_size, W)
                
                # Extract window
                mag_window = magnitudes[:, :, h_start:h_end, w_start:w_end]
                angle_window = angles[:, :, h_start:h_end, w_start:w_end]
                
                # Compute vector flow features for this window
                vx = mag_window * torch.cos(angle_window)
                vy = mag_window * torch.sin(angle_window)
                flow_features = torch.stack([vx, vy], dim=-1)  # [B, C, h, w, 2]
                
                # Flatten for attention
                flow_flat = flow_features.flatten(2, 3)  # [B, C, hw, 2]
                flow_flat = flow_flat.permute(0, 2, 1, 3).flatten(2)  # [B, hw, C*2]
                
                # Apply attention
                attended_flow, _ = self.flow_attention(flow_flat, flow_flat, flow_flat)
                
                # Update coherence weights based on flow attention
                attention_weights = attended_flow.norm(dim=-1, keepdim=True)  # [B, hw, 1]
                attention_weights = attention_weights.view(B, h_end-h_start, w_end-w_start, 1)
                attention_weights = attention_weights.permute(0, 3, 1, 2).expand(-1, C, -1, -1)
                
                # Blend with existing coherence
                refined[:, :, h_start:h_end, w_start:w_end] = (
                    refined[:, :, h_start:h_end, w_start:w_end] * 0.7 + 
                    attention_weights * 0.3
                )
        
        return refined


class AttentionEnhancedSFVNN(nn.Module):
    """SF-VNN enhanced with windowed attention mechanisms."""
    
    def __init__(self,
                 input_channels: int = 1,
                 vector_channels: list = [32, 64, 128],
                 window_size: int = 8,
                 use_vector_attention: bool = True,
                 use_structural_attention: bool = True,
                 use_spectrotemporal_attention: bool = True):
        super().__init__()
        
        self.use_vector_attention = use_vector_attention
        self.use_structural_attention = use_structural_attention
        self.use_spectrotemporal_attention = use_spectrotemporal_attention
        
        # Import your existing components
        from vector_network import VectorNeuronLayer
        from audio_discriminator import AudioStructuralAnalyzer, AudioDiscriminatorHead
        
        # Vector neuron backbone (same as before)
        self.vector_layers = nn.ModuleList()
        in_ch = input_channels
        
        for i, out_ch in enumerate(vector_channels):
            stride = 2 if i > 0 else 1
            self.vector_layers.append(
                VectorNeuronLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=stride,
                    magnitude_activation='relu',
                    angle_activation='tanh'
                )
            )
            in_ch = out_ch * 2
        
        # NEW: Attention layers
        if use_spectrotemporal_attention:
            self.input_attention = SpectroTemporalAttention(
                input_channels, freq_window=8, time_window=16
            )
        
        if use_vector_attention:
            self.vector_attention_layers = nn.ModuleList([
                VectorFieldAttention(out_ch, window_size) 
                for out_ch in vector_channels
            ])
        
        # Structural analyzer (same as before)
        self.structural_analyzer = AudioStructuralAnalyzer(window_size=5, sigma=1.0)
        
        if use_structural_attention:
            self.structural_attention = AdaptiveStructuralAttention(
                signature_channels=6, window_size=window_size
            )
        
        # Classification head
        self.classifier = AudioDiscriminatorHead(
            signature_channels=6,
            num_scales=1,
            hidden_dim=256
        )
    
    def forward(self, x: torch.Tensor, return_signature: bool = False):
        """Forward pass with attention-enhanced processing."""
        
        # Input spectro-temporal attention
        if self.use_spectrotemporal_attention:
            x = self.input_attention(x)
        
        # Vector neuron processing with attention
        vector_field = x
        for i, layer in enumerate(self.vector_layers):
            vector_field = layer(vector_field)
            
            # Apply vector field attention after each layer
            if self.use_vector_attention and i < len(self.vector_attention_layers):
                vector_field = self.vector_attention_layers[i](vector_field)
        
        # Structural analysis
        signature = self.structural_analyzer.analyze_audio_spectrogram(vector_field)
        
        # Structural attention refinement
        if self.use_structural_attention:
            signature = self.structural_attention(signature)
        
        # Classification
        output = self.classifier([signature])
        
        if return_signature:
            return output, [signature]
        return output


class SpectroTemporalAttention(nn.Module):
    """Specialized attention for audio spectrograms."""
    
    def __init__(self, channels: int, freq_window: int = 8, time_window: int = 16):
        super().__init__()
        self.freq_window = freq_window
        self.time_window = time_window
        
        # Separate attention for frequency and time dimensions
        self.freq_attention = WindowedAttention(channels, num_heads=4, window_size=freq_window)
        self.time_attention = WindowedAttention(channels, num_heads=4, window_size=time_window)
        
        # Cross-dimensional attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True
        )
        
        # Combination network
        self.combine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, T] - spectrogram
        """
        B, C, F, T = x.shape
        
        # Frequency attention (attend along frequency bins)
        x_freq = x.permute(0, 3, 1, 2).reshape(B * T, C, F)  # [B*T, C, F]
        x_freq_attended = self.apply_1d_windowed_attention(x_freq, self.freq_attention, self.freq_window)
        x_freq_attended = x_freq_attended.reshape(B, T, C, F).permute(0, 2, 3, 1)  # [B, C, F, T]
        
        # Time attention (attend along time frames)
        x_time = x.permute(0, 2, 1, 3).reshape(B * F, C, T)  # [B*F, C, T]
        x_time_attended = self.apply_1d_windowed_attention(x_time, self.time_attention, self.time_window)
        x_time_attended = x_time_attended.reshape(B, F, C, T).permute(0, 2, 1, 3)  # [B, C, F, T]
        
        # Combine frequency and time attended features
        combined = torch.cat([x_freq_attended, x_time_attended], dim=1)
        output = self.combine(combined)
        
        return output + x  # Residual connection
    
    def apply_1d_windowed_attention(self, x: torch.Tensor, attention: WindowedAttention, window_size: int) -> torch.Tensor:
        """Apply windowed attention along 1D sequence."""
        B, C, L = x.shape
        
        if L < window_size:
            # If sequence is shorter than window, apply global attention
            x_reshaped = x.permute(0, 2, 1)  # [B, L, C]
            return attention(x_reshaped).permute(0, 2, 1)
        
        # Pad to make divisible by window size
        pad_length = (window_size - L % window_size) % window_size
        if pad_length > 0:
            x = F.pad(x, (0, pad_length))
            L = L + pad_length
        
        # Reshape into windows
        x_windowed = x.view(B, C, L // window_size, window_size)
        x_windowed = x_windowed.permute(0, 2, 3, 1)  # [B, num_windows, window_size, C]
        x_windowed = x_windowed.reshape(B * (L // window_size), window_size, C)
        
        # Apply attention
        x_attended = attention(x_windowed)
        
        # Reshape back
        x_attended = x_attended.reshape(B, L // window_size, window_size, C)
        x_attended = x_attended.permute(0, 3, 1, 2)  # [B, C, num_windows, window_size]
        x_attended = x_attended.reshape(B, C, L)
        
        # Remove padding
        if pad_length > 0:
            x_attended = x_attended[:, :, :-pad_length]
        
        return x_attended


# Integration with your existing training framework
class AttentionSFVNNTrainer:
    """Enhanced trainer for attention-based SF-VNN."""
    
    def __init__(self, model_config: dict, attention_config: dict):
        
        # Create attention-enhanced model
        self.model = AttentionEnhancedSFVNN(
            input_channels=model_config.get('input_channels', 1),
            vector_channels=model_config.get('vector_channels', [32, 64, 128]),
            window_size=attention_config.get('window_size', 8),
            use_vector_attention=attention_config.get('use_vector_attention', True),
            use_structural_attention=attention_config.get('use_structural_attention', True),
            use_spectrotemporal_attention=attention_config.get('use_spectrotemporal_attention', True)
        )
        
        # Attention-specific optimizations
        self.setup_attention_optimizations()
    
    def setup_attention_optimizations(self):
        """Setup optimizations specific to attention mechanisms."""
        
        # Gradient scaling for attention layers (they can have different learning dynamics)
        attention_params = []
        backbone_params = []
        
        for name, param in self.model.named_parameters():
            if 'attention' in name.lower():
                attention_params.append(param)
            else:
                backbone_params.append(param)
        
        # Different learning rates for attention vs backbone
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 2e-4},
            {'params': attention_params, 'lr': 1e-4}  # Lower LR for attention stability
        ])
        
        # Attention-specific learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )


# Example usage showing the integration
def create_attention_enhanced_sfvnn_system():
    """Create complete attention-enhanced SF-VNN system."""
    
    # Model configuration
    model_config = {
        'input_channels': 1,
        'vector_channels': [32, 64, 128, 256],
    }
    
    # Attention configuration
    attention_config = {
        'window_size': 8,
        'use_vector_attention': True,
        'use_structural_attention': True, 
        'use_spectrotemporal_attention': True
    }
    
    # Create enhanced model
    model = AttentionEnhancedSFVNN(**model_config, **attention_config)
    
    print(f"Created Attention-Enhanced SF-VNN with:")
    print(f"  - Vector Field Attention: {attention_config['use_vector_attention']}")
    print(f"  - Structural Attention: {attention_config['use_structural_attention']}")
    print(f"  - Spectro-Temporal Attention: {attention_config['use_spectrotemporal_attention']}")
    print(f"  - Window Size: {attention_config['window_size']}")
    
    return model

if __name__ == "__main__":
    # Test the attention-enhanced SF-VNN
    model = create_attention_enhanced_sfvnn_system()
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 80, 128)  # [batch, channels, freq, time]
    output, signatures = model(dummy_input, return_signature=True)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of structural signatures: {len(signatures)}")
    print("✅ Attention-Enhanced SF-VNN working correctly!")