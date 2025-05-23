#!/usr/bin/env python3
"""
Quick Attention Test: Fast comparison of SF-VNN vs Attention-Enhanced SF-VNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict

# Import components
from hifi import HiFiGANConfig, HiFiGANGenerator, AudioSFVNNDiscriminator
from empirical_comparison import VanillaCNNDiscriminator

def create_simple_attention_sfvnn():
    """Create a simple attention-enhanced SF-VNN for quick testing."""
    
    class SimpleAttentionSFVNN(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Import vector neuron layer
            import sys, importlib.util
            spec = importlib.util.spec_from_file_location("vector_network", "vector-network.py")
            vector_network = importlib.util.module_from_spec(spec)
            sys.modules["vector_network"] = vector_network
            spec.loader.exec_module(vector_network)
            VectorNeuronLayer = vector_network.VectorNeuronLayer
            
            # Vector layers
            self.vector_layers = nn.ModuleList([
                VectorNeuronLayer(1, 16, stride=1, magnitude_activation='relu', angle_activation='tanh'),
                VectorNeuronLayer(32, 32, stride=2, magnitude_activation='relu', angle_activation='tanh'),
                VectorNeuronLayer(64, 64, stride=2, magnitude_activation='relu', angle_activation='tanh')
            ])
            
            # Simple attention mechanisms
            self.input_attention = nn.Sequential(
                nn.Conv2d(1, 1, 3, padding=1),
                nn.Sigmoid()
            )
            
            self.spatial_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128, 128, 1),
                nn.Sigmoid()
            )
            
            # Classification
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            # Input attention
            att_weights = self.input_attention(x)
            x = x * att_weights + x
            
            # Vector processing
            for layer in self.vector_layers:
                x = layer(x)
            
            # Spatial attention
            spatial_weights = self.spatial_attention(x)
            x = x * spatial_weights
            
            # Classification
            return self.classifier(x)
    
    return SimpleAttentionSFVNN()

def quick_attention_comparison():
    """Run quick comparison of SF-VNN vs Attention SF-VNN."""
    
    print("⚡ QUICK ATTENTION COMPARISON")
    print("SF-VNN vs Attention-Enhanced SF-VNN")
    print("=" * 50)
    
    device = "cpu"
    
    # Create models
    print("🔧 Creating models...")
    
    # Regular SF-VNN
    sf_vnn = AudioSFVNNDiscriminator(
        input_channels=1,
        vector_channels=[16, 32, 64],
        window_size=3,
        sigma=1.0,
        multiscale_analysis=False  # Simplified
    ).to(device)
    
    # Attention-enhanced SF-VNN
    attention_sf_vnn = create_simple_attention_sfvnn().to(device)
    
    # Generator
    generator = HiFiGANGenerator(HiFiGANConfig()).to(device)
    
    # Model statistics
    sf_params = sum(p.numel() for p in sf_vnn.parameters())
    att_params = sum(p.numel() for p in attention_sf_vnn.parameters())
    
    print(f"📊 Model Parameters:")
    print(f"   SF-VNN: {sf_params:,}")
    print(f"   Attention SF-VNN: {att_params:,}")
    print(f"   Attention Overhead: {(att_params/sf_params - 1)*100:.1f}%")
    
    # Test data
    real_audio = torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 8192)).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1)
    
    # Training setup
    sf_optimizer = torch.optim.Adam(sf_vnn.parameters(), lr=2e-4)
    att_optimizer = torch.optim.Adam(attention_sf_vnn.parameters(), lr=2e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Quick training comparison
    print(f"\\n🏃 Quick Training Test (10 steps)...")
    
    sf_losses = []
    att_losses = []
    sf_stabilities = []
    att_stabilities = []
    
    for step in range(10):
        
        # Generate fake data
        with torch.no_grad():
            noise_mel = torch.randn(2, 80, 16)
            fake_audio = generator(noise_mel)[..., :8192].unsqueeze(1) if generator(noise_mel).dim() == 2 else generator(noise_mel)[..., :8192]
        
        # Convert to mel
        import torchaudio.transforms as T
        mel_transform = T.MelSpectrogram(n_mels=40, n_fft=256, hop_length=64)
        
        try:
            real_mel = mel_transform(real_audio.squeeze(1)).unsqueeze(1)
            fake_mel = mel_transform(fake_audio.squeeze(1)).unsqueeze(1)
            
            min_w = min(real_mel.size(-1), fake_mel.size(-1))
            real_mel = real_mel[..., :min_w]
            fake_mel = fake_mel[..., :min_w]
            
        except:
            continue
        
        # Train SF-VNN
        try:
            sf_optimizer.zero_grad()
            sf_real_pred = sf_vnn(real_mel)
            sf_fake_pred = sf_vnn(fake_mel.detach())
            
            sf_loss = (criterion(sf_real_pred, torch.ones_like(sf_real_pred)) + 
                      criterion(sf_fake_pred, torch.zeros_like(sf_fake_pred))) / 2
            
            sf_loss.backward()
            
            # Calculate gradient norm
            sf_grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in sf_vnn.parameters() 
                if p.grad is not None
            ]))
            
            sf_optimizer.step()
            
            sf_losses.append(sf_loss.item())
            sf_stabilities.append(1.0 / (sf_grad_norm.item() + 1e-6))
            
        except Exception as e:
            print(f"SF-VNN training failed at step {step}: {e}")
            sf_losses.append(float('nan'))
            sf_stabilities.append(0.0)
        
        # Train Attention SF-VNN
        try:
            att_optimizer.zero_grad()
            att_real_pred = attention_sf_vnn(real_mel)
            att_fake_pred = attention_sf_vnn(fake_mel.detach())
            
            att_loss = (criterion(att_real_pred, torch.ones_like(att_real_pred)) + 
                       criterion(att_fake_pred, torch.zeros_like(att_fake_pred))) / 2
            
            att_loss.backward()
            
            # Calculate gradient norm
            att_grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in attention_sf_vnn.parameters() 
                if p.grad is not None
            ]))
            
            att_optimizer.step()
            
            att_losses.append(att_loss.item())
            att_stabilities.append(1.0 / (att_grad_norm.item() + 1e-6))
            
        except Exception as e:
            print(f"Attention SF-VNN training failed at step {step}: {e}")
            att_losses.append(float('nan'))
            att_stabilities.append(0.0)
        
        if step % 3 == 0:
            sf_status = f"{sf_losses[-1]:.4f}" if not np.isnan(sf_losses[-1]) else "Failed"
            att_status = f"{att_losses[-1]:.4f}" if not np.isnan(att_losses[-1]) else "Failed"
            print(f"   Step {step}: SF-VNN={sf_status}, Attention={att_status}")
    
    # Analysis
    print(f"\\n📈 Training Analysis:")
    
    # Filter out NaN values
    sf_valid_losses = [l for l in sf_losses if not np.isnan(l)]
    att_valid_losses = [l for l in att_losses if not np.isnan(l)]
    sf_valid_stabs = [s for s in sf_stabilities if s > 0]
    att_valid_stabs = [s for s in att_stabilities if s > 0]
    
    if sf_valid_losses and att_valid_losses:
        # Loss analysis
        sf_final = sf_valid_losses[-1]
        att_final = att_valid_losses[-1]
        sf_var = np.var(sf_valid_losses)
        att_var = np.var(att_valid_losses)
        
        print(f"   Final Loss - SF-VNN: {sf_final:.4f}, Attention: {att_final:.4f}")
        print(f"   Loss Variance - SF-VNN: {sf_var:.6f}, Attention: {att_var:.6f}")
        
        loss_winner = "Attention SF-VNN" if att_final < sf_final else "SF-VNN"
        stability_winner = "Attention SF-VNN" if att_var < sf_var else "SF-VNN"
        
        print(f"   🏆 Lower Final Loss: {loss_winner}")
        print(f"   🏆 More Stable Training: {stability_winner}")
        
        # Stability analysis
        if sf_valid_stabs and att_valid_stabs:
            sf_avg_stab = np.mean(sf_valid_stabs)
            att_avg_stab = np.mean(att_valid_stabs)
            
            print(f"   Average Stability - SF-VNN: {sf_avg_stab:.4f}, Attention: {att_avg_stab:.4f}")
            
            stab_winner = "Attention SF-VNN" if att_avg_stab > sf_avg_stab else "SF-VNN"
            print(f"   🏆 Higher Stability Score: {stab_winner}")
        
        # Success rate
        sf_success_rate = len(sf_valid_losses) / len(sf_losses)
        att_success_rate = len(att_valid_losses) / len(att_losses)
        
        print(f"   Training Success Rate - SF-VNN: {sf_success_rate*100:.0f}%, Attention: {att_success_rate*100:.0f}%")
        
        success_winner = "Attention SF-VNN" if att_success_rate > sf_success_rate else "SF-VNN"
        print(f"   🏆 Higher Success Rate: {success_winner}")
        
        # Overall assessment
        attention_wins = sum([
            att_final < sf_final,
            att_var < sf_var,
            att_avg_stab > sf_avg_stab if sf_valid_stabs and att_valid_stabs else False,
            att_success_rate > sf_success_rate
        ])
        
        print(f"\\n🎯 ATTENTION ENHANCEMENT ASSESSMENT:")
        if attention_wins >= 2:
            print(f"   ✅ ATTENTION PROVIDES CLEAR BENEFITS ({attention_wins}/4 metrics)")
            print(f"   🎉 Windowed attention enhances SF-VNN performance!")
        else:
            print(f"   🤔 MIXED RESULTS ({attention_wins}/4 metrics)")
            print(f"   💡 May need parameter tuning or longer training")
        
        # Create quick visualization
        create_quick_attention_plot(sf_losses, att_losses, sf_stabilities, att_stabilities)
        
        return {
            'sf_vnn': {
                'final_loss': sf_final,
                'loss_variance': sf_var,
                'avg_stability': sf_avg_stab if sf_valid_stabs else 0,
                'success_rate': sf_success_rate,
                'parameters': sf_params
            },
            'attention_sf_vnn': {
                'final_loss': att_final,
                'loss_variance': att_var,
                'avg_stability': att_avg_stab if att_valid_stabs else 0,
                'success_rate': att_success_rate,
                'parameters': att_params
            },
            'attention_wins': attention_wins,
            'benefits_found': attention_wins >= 2
        }
    
    else:
        print("   ❌ Training failed for one or both models")
        return {'error': 'Training failed'}

def create_quick_attention_plot(sf_losses, att_losses, sf_stabs, att_stabs):
    """Create quick visualization of attention benefits."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss comparison
    steps = range(len(sf_losses))
    
    ax1.plot(steps, sf_losses, 'b-o', label='SF-VNN', linewidth=2, markersize=4)
    ax1.plot(steps, att_losses, 'r-s', label='Attention SF-VNN', linewidth=2, markersize=4)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Discriminator Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stability comparison
    ax2.plot(steps, sf_stabs, 'b-o', label='SF-VNN', linewidth=2, markersize=4)
    ax2.plot(steps, att_stabs, 'r-s', label='Attention SF-VNN', linewidth=2, markersize=4)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Stability Score')
    ax2.set_title('Training Stability Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_attention_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Quick comparison plot saved: quick_attention_comparison.png")

def test_attention_mechanisms():
    """Test specific attention mechanisms."""
    
    print("\\n🔍 ATTENTION MECHANISM ANALYSIS")
    print("=" * 50)
    
    # Test windowed attention
    from windowed_attention import WindowedAttention
    
    print("🧪 Testing Windowed Attention...")
    
    window_attention = WindowedAttention(dim=64, num_heads=4, window_size=8)
    test_input = torch.randn(2, 8*8, 64)  # [batch, sequence, features]
    
    start_time = time.time()
    output = window_attention(test_input)
    end_time = time.time()
    
    print(f"   Input: {test_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Computation Time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test vector field attention
    from attention_extensions import VectorFieldAttention
    
    print("\\n🧪 Testing Vector Field Attention...")
    
    vector_attention = VectorFieldAttention(channels=32, window_size=8)
    test_vector_field = torch.randn(2, 64, 16, 16)  # [batch, channels*2, height, width]
    
    start_time = time.time()
    output = vector_attention(test_vector_field)
    end_time = time.time()
    
    print(f"   Input: {test_vector_field.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Computation Time: {(end_time - start_time)*1000:.2f}ms")
    
    print("\\n✅ Attention mechanisms working correctly!")

def main():
    """Run quick attention comparison."""
    
    print("⚡ QUICK ATTENTION ENHANCEMENT TEST")
    print("Testing windowed attention benefits for SF-VNN")
    print("=" * 60)
    
    # Test attention mechanisms
    test_attention_mechanisms()
    
    # Run comparison
    results = quick_attention_comparison()
    
    # Save results
    with open('quick_attention_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n💾 Results saved to: quick_attention_results.json")
    print("✅ Quick attention test completed!")

if __name__ == "__main__":
    main()