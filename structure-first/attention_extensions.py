#!/usr/bin/env python3
"""
Missing Attention Components for Windowed Attention SF-VNN
Completes the windowed-attention.py implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class VectorFieldAttention(nn.Module):
    """Attention mechanism for vector fields (magnitude + angle)."""
    
    def __init__(self, channels: int, window_size: int = 8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        
        # Vector field coherence analysis
        self.magnitude_processor = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 3, padding=1)
        )
        
        self.angle_processor = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(channels // 2, channels, 3, padding=1)
        )
        
        # Cross-vector attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=channels * 2, num_heads=4, batch_first=True
        )
        
        # Vector flow coherence
        self.flow_coherence = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, vector_field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vector_field: [B, C*2, H, W] - interleaved magnitude and angle channels
        Returns:
            enhanced_field: [B, C*2, H, W] - attention-enhanced vector field
        """
        B, C_total, H, W = vector_field.shape
        C = C_total // 2  # Half channels for magnitude, half for angle
        
        # Split into magnitude and angle components
        magnitudes = vector_field[:, :C, :, :]
        angles = vector_field[:, C:, :, :]
        
        # Process each component
        mag_processed = self.magnitude_processor(magnitudes)
        angle_processed = self.angle_processor(angles)
        
        # Apply windowed attention to vector field coherence
        if H >= self.window_size and W >= self.window_size:
            enhanced_mag, enhanced_angle = self._apply_windowed_vector_attention(
                mag_processed, angle_processed
            )
        else:
            enhanced_mag, enhanced_angle = mag_processed, angle_processed
        
        # Compute flow coherence
        combined = torch.cat([enhanced_mag, enhanced_angle], dim=1)
        flow_weights = self.flow_coherence(combined)
        
        # Apply coherence weighting
        enhanced_mag = enhanced_mag * flow_weights
        enhanced_angle = enhanced_angle * flow_weights
        
        # Combine and add residual
        enhanced_field = torch.cat([enhanced_mag, enhanced_angle], dim=1)
        return enhanced_field + vector_field  # Residual connection
    
    def _apply_windowed_vector_attention(self, magnitudes: torch.Tensor, 
                                       angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention in local windows."""
        B, C, H, W = magnitudes.shape
        enhanced_mag = magnitudes.clone()
        enhanced_angle = angles.clone()
        
        # Process overlapping windows
        stride = self.window_size // 2
        for h_start in range(0, H - self.window_size + 1, stride):
            for w_start in range(0, W - self.window_size + 1, stride):
                h_end = h_start + self.window_size
                w_end = w_start + self.window_size
                
                # Extract windows
                mag_window = magnitudes[:, :, h_start:h_end, w_start:w_end]
                angle_window = angles[:, :, h_start:h_end, w_start:w_end]
                
                # Flatten for attention
                mag_flat = mag_window.flatten(2).permute(0, 2, 1)  # [B, HW, C]
                angle_flat = angle_window.flatten(2).permute(0, 2, 1)  # [B, HW, C]
                
                # Combine for cross-attention
                combined_flat = torch.cat([mag_flat, angle_flat], dim=-1)  # [B, HW, 2C]
                
                # Apply attention
                attended, _ = self.cross_attention(combined_flat, combined_flat, combined_flat)
                
                # Split back
                attended_mag = attended[:, :, :C].permute(0, 2, 1).reshape(B, C, self.window_size, self.window_size)
                attended_angle = attended[:, :, C:].permute(0, 2, 1).reshape(B, C, self.window_size, self.window_size)
                
                # Blend with existing (for overlapping windows)
                alpha = 0.3
                enhanced_mag[:, :, h_start:h_end, w_start:w_end] = (
                    enhanced_mag[:, :, h_start:h_end, w_start:w_end] * (1 - alpha) + 
                    attended_mag * alpha
                )
                enhanced_angle[:, :, h_start:h_end, w_start:w_end] = (
                    enhanced_angle[:, :, h_start:h_end, w_start:w_end] * (1 - alpha) + 
                    attended_angle * alpha
                )
        
        return enhanced_mag, enhanced_angle


class AdaptiveStructuralAttention(nn.Module):
    """Adaptive attention for structural signatures."""
    
    def __init__(self, signature_channels: int = 6, window_size: int = 8):
        super().__init__()
        self.signature_channels = signature_channels
        self.window_size = window_size
        
        # Structural importance network
        self.importance_net = nn.Sequential(
            nn.Conv2d(signature_channels, signature_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(signature_channels * 2, signature_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Multi-scale structural analysis
        self.multiscale_attention = nn.ModuleList([
            self._create_scale_attention(signature_channels, scale)
            for scale in [1, 2, 4]  # Different scales for structural analysis
        ])
        
        # Adaptive fusion
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(signature_channels * 3, signature_channels, 1),
            nn.ReLU(),
            nn.Conv2d(signature_channels, signature_channels, 1)
        )
    
    def _create_scale_attention(self, channels: int, scale: int) -> nn.Module:
        """Create attention module for specific scale."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=scale, dilation=scale),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, signature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signature: [B, C, H, W] - structural signature
        Returns:
            enhanced_signature: [B, C, H, W] - attention-enhanced signature
        """
        B, C, H, W = signature.shape
        
        # Compute structural importance
        importance_weights = self.importance_net(signature)
        
        # Multi-scale attention
        scale_features = []
        for scale_attention in self.multiscale_attention:
            scale_weights = scale_attention(signature)
            scale_features.append(signature * scale_weights)
        
        # Combine scales
        multiscale_combined = torch.cat(scale_features, dim=1)
        fused_features = self.adaptive_fusion(multiscale_combined)
        
        # Apply importance weighting
        enhanced_signature = fused_features * importance_weights
        
        return enhanced_signature + signature  # Residual connection


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial attention."""
    
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        
        # Create 2D positional encodings
        pe = torch.zeros(channels, height, width)
        
        div_term = torch.exp(torch.arange(0, channels, 2).float() * 
                           (-math.log(10000.0) / channels))
        
        # Height encoding
        pos_h = torch.arange(height).unsqueeze(1)
        pe[0::2, :, :] = torch.sin(pos_h * div_term).unsqueeze(-1).expand(-1, -1, width)
        pe[1::2, :, :] = torch.cos(pos_h * div_term).unsqueeze(-1).expand(-1, -1, width)
        
        # Width encoding (add to existing)
        pos_w = torch.arange(width).unsqueeze(0)
        pe[0::2, :, :] += torch.sin(pos_w * div_term).unsqueeze(1).expand(-1, height, -1)
        pe[1::2, :, :] += torch.cos(pos_w * div_term).unsqueeze(1).expand(-1, height, -1)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]


class AttentionDiagnostics(nn.Module):
    """Diagnostic tools for attention analysis."""
    
    def __init__(self):
        super().__init__()
        self.attention_maps = {}
        self.flow_patterns = {}
    
    def capture_attention_map(self, name: str, attention_weights: torch.Tensor):
        """Capture attention map for analysis."""
        self.attention_maps[name] = attention_weights.detach().cpu()
    
    def capture_flow_pattern(self, name: str, vector_field: torch.Tensor):
        """Capture vector flow pattern."""
        # Convert vector field to flow visualization
        B, C, H, W = vector_field.shape
        if C >= 2:
            # Assume first half is magnitude, second half is angle
            C_half = C // 2
            magnitudes = vector_field[:, :C_half, :, :]
            angles = vector_field[:, C_half:, :, :]
            
            # Compute flow vectors
            flow_x = magnitudes * torch.cos(angles)
            flow_y = magnitudes * torch.sin(angles)
            
            self.flow_patterns[name] = {
                'flow_x': flow_x.detach().cpu(),
                'flow_y': flow_y.detach().cpu(),
                'magnitude': magnitudes.detach().cpu(),
                'angle': angles.detach().cpu()
            }
    
    def get_attention_statistics(self) -> dict:
        """Get statistics about captured attention patterns."""
        stats = {}
        
        for name, attention_map in self.attention_maps.items():
            stats[name] = {
                'mean_attention': attention_map.mean().item(),
                'std_attention': attention_map.std().item(),
                'max_attention': attention_map.max().item(),
                'sparsity': (attention_map < 0.1).float().mean().item()
            }
        
        return stats
    
    def get_flow_statistics(self) -> dict:
        """Get statistics about vector flow patterns."""
        stats = {}
        
        for name, flow_data in self.flow_patterns.items():
            magnitude = flow_data['magnitude']
            angle = flow_data['angle']
            
            # Compute flow coherence
            flow_magnitude = torch.sqrt(flow_data['flow_x']**2 + flow_data['flow_y']**2)
            
            stats[name] = {
                'mean_magnitude': magnitude.mean().item(),
                'std_magnitude': magnitude.std().item(),
                'mean_flow_coherence': flow_magnitude.mean().item(),
                'angle_distribution_std': angle.std().item()
            }
        
        return stats


# Integration helpers
def create_attention_enhanced_discriminator(base_config: dict, attention_config: dict):
    """Create attention-enhanced SF-VNN discriminator."""
    
    # Import the windowed attention system
    import sys
    import importlib.util
    
    # Load the windowed attention module
    spec = importlib.util.spec_from_file_location("windowed_attention", "windowed-attention.py")
    windowed_attention = importlib.util.module_from_spec(spec)
    sys.modules["windowed_attention"] = windowed_attention
    spec.loader.exec_module(windowed_attention)
    
    # Create the attention-enhanced model
    model = windowed_attention.AttentionEnhancedSFVNN(
        input_channels=base_config.get('input_channels', 1),
        vector_channels=base_config.get('vector_channels', [32, 64, 128]),
        window_size=attention_config.get('window_size', 8),
        use_vector_attention=attention_config.get('use_vector_attention', True),
        use_structural_attention=attention_config.get('use_structural_attention', True),
        use_spectrotemporal_attention=attention_config.get('use_spectrotemporal_attention', True)
    )
    
    return model


# Test the complete attention system
def test_attention_components():
    """Test all attention components."""
    
    print("🧪 Testing Attention Components")
    print("=" * 50)
    
    # Test VectorFieldAttention
    vector_attention = VectorFieldAttention(channels=64, window_size=8)
    test_vector_field = torch.randn(2, 128, 32, 32)  # [B, C*2, H, W]
    
    output = vector_attention(test_vector_field)
    print(f"✅ VectorFieldAttention: {test_vector_field.shape} -> {output.shape}")
    
    # Test AdaptiveStructuralAttention
    structural_attention = AdaptiveStructuralAttention(signature_channels=6, window_size=8)
    test_signature = torch.randn(2, 6, 16, 16)
    
    output = structural_attention(test_signature)
    print(f"✅ AdaptiveStructuralAttention: {test_signature.shape} -> {output.shape}")
    
    # Test PositionalEncoding2D
    pos_encoding = PositionalEncoding2D(channels=64, height=32, width=32)
    test_input = torch.randn(2, 64, 32, 32)
    
    output = pos_encoding(test_input)
    print(f"✅ PositionalEncoding2D: {test_input.shape} -> {output.shape}")
    
    # Test AttentionDiagnostics
    diagnostics = AttentionDiagnostics()
    diagnostics.capture_attention_map("test", torch.rand(2, 8, 16, 16))
    diagnostics.capture_flow_pattern("test", torch.randn(2, 128, 16, 16))
    
    attention_stats = diagnostics.get_attention_statistics()
    flow_stats = diagnostics.get_flow_statistics()
    
    print(f"✅ AttentionDiagnostics: Captured {len(attention_stats)} attention maps, {len(flow_stats)} flow patterns")
    
    print("\n🎉 All attention components working correctly!")


if __name__ == "__main__":
    test_attention_components()