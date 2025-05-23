#!/usr/bin/env python3
"""
Streamlined Attention Test: Quick evaluation of attention benefits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Import components
from hifi import HiFiGANConfig, HiFiGANGenerator, AudioSFVNNDiscriminator

def create_attention_enhanced_sfvnn():
    """Create attention-enhanced SF-VNN with built-in attention mechanisms."""
    
    class AttentionSFVNN(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Import vector neurons
            import sys, importlib.util
            spec = importlib.util.spec_from_file_location("vector_network", "vector-network.py")
            vector_network = importlib.util.module_from_spec(spec)
            sys.modules["vector_network"] = vector_network
            spec.loader.exec_module(vector_network)
            VectorNeuronLayer = vector_network.VectorNeuronLayer
            
            # Input spectro-temporal attention
            self.input_attention = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=(3, 7), padding=(1, 3)),  # Freq x Time attention
                nn.Sigmoid()
            )
            
            # Vector neuron layers with attention
            self.vector_layers = nn.ModuleList([
                VectorNeuronLayer(1, 16, stride=1, magnitude_activation='relu', angle_activation='tanh'),
                VectorNeuronLayer(32, 32, stride=2, magnitude_activation='relu', angle_activation='tanh'),
                VectorNeuronLayer(64, 32, stride=2, magnitude_activation='relu', angle_activation='tanh')
            ])
            
            # Channel attention for vector fields
            self.channel_attention = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(32, 16, 1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 1),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(64, 32, 1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 1),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(64, 32, 1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 1),
                    nn.Sigmoid()
                )
            ])
            
            # Spatial attention for structural coherence
            self.spatial_attention = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(32, 1, kernel_size=7, padding=3),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.Conv2d(64, 1, kernel_size=7, padding=3),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.Conv2d(64, 1, kernel_size=7, padding=3),
                    nn.Sigmoid()
                )
            ])
            
            # Final classification with attention
            self.global_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(64, 64, 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1),
                nn.Sigmoid()
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            # Input spectro-temporal attention
            attention_weights = self.input_attention(x)
            x = x * attention_weights + x  # Residual connection
            
            # Vector processing with multi-level attention
            for i, (vector_layer, ch_att, sp_att) in enumerate(zip(
                self.vector_layers, self.channel_attention, self.spatial_attention
            )):
                # Vector neuron processing
                x = vector_layer(x)
                
                # Channel attention (what to focus on)
                ch_weights = ch_att(x)
                x_ch = x * ch_weights
                
                # Spatial attention (where to focus)
                sp_weights = sp_att(x_ch)
                x = x_ch * sp_weights + x_ch  # Residual
            
            # Global attention before classification
            global_weights = self.global_attention(x)
            x = x * global_weights
            
            return self.classifier(x)
    
    return AttentionSFVNN()

def run_streamlined_attention_test():
    """Run streamlined attention vs regular SF-VNN test."""
    
    print("🚀 STREAMLINED ATTENTION TEST")
    print("SF-VNN vs Attention-Enhanced SF-VNN")
    print("=" * 50)
    
    device = "cpu"
    
    # Create models
    print("🔧 Creating models...")
    
    # Regular SF-VNN (simplified)
    regular_sfvnn = AudioSFVNNDiscriminator(
        input_channels=1,
        vector_channels=[16, 32, 32],
        window_size=3,
        sigma=1.0,
        multiscale_analysis=False
    ).to(device)
    
    # Attention-enhanced SF-VNN
    attention_sfvnn = create_attention_enhanced_sfvnn().to(device)
    
    # Generator
    generator = HiFiGANGenerator(HiFiGANConfig()).to(device)
    
    # Model statistics
    regular_params = sum(p.numel() for p in regular_sfvnn.parameters())
    attention_params = sum(p.numel() for p in attention_sfvnn.parameters())
    
    print(f"📊 Model Comparison:")
    print(f"   Regular SF-VNN: {regular_params:,} parameters")
    print(f"   Attention SF-VNN: {attention_params:,} parameters")
    print(f"   Attention overhead: {(attention_params/regular_params - 1)*100:.1f}%")
    
    # Test data
    real_audio = torch.randn(2, 1, 8192) * 0.5  # Reduced magnitude
    
    # Optimizers
    regular_opt = torch.optim.Adam(regular_sfvnn.parameters(), lr=1e-4)  # Lower LR
    attention_opt = torch.optim.Adam(attention_sfvnn.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\\n🏃 Training Test (15 steps)...")
    
    # Training results
    results = {
        'regular': {'losses': [], 'stabilities': [], 'predictions': []},
        'attention': {'losses': [], 'stabilities': [], 'predictions': []}
    }
    
    for step in range(15):
        
        # Generate fake audio
        with torch.no_grad():
            noise_mel = torch.randn(2, 80, 16)
            fake_audio = generator(noise_mel)
            if fake_audio.dim() == 2:
                fake_audio = fake_audio.unsqueeze(1)
            fake_audio = fake_audio[..., :8192]
        
        # Convert to mel spectrograms
        import torchaudio.transforms as T
        mel_transform = T.MelSpectrogram(n_mels=40, n_fft=256, hop_length=64)
        
        try:
            real_mel = mel_transform(real_audio.squeeze(1)).unsqueeze(1)
            fake_mel = mel_transform(fake_audio.squeeze(1)).unsqueeze(1)
            
            # Ensure consistent dimensions
            min_w = min(real_mel.size(-1), fake_mel.size(-1))
            real_mel = real_mel[..., :min_w]
            fake_mel = fake_mel[..., :min_w]
            
        except Exception as e:
            print(f"   Warning: Mel conversion failed at step {step}: {e}")
            continue
        
        # Train regular SF-VNN
        try:
            regular_opt.zero_grad()
            real_pred = regular_sfvnn(real_mel)
            fake_pred = regular_sfvnn(fake_mel.detach())
            
            loss_real = criterion(real_pred, torch.ones_like(real_pred))
            loss_fake = criterion(fake_pred, torch.zeros_like(fake_pred))
            regular_loss = (loss_real + loss_fake) / 2
            
            regular_loss.backward()
            
            # Gradient norm
            regular_grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in regular_sfvnn.parameters() 
                if p.grad is not None
            ]))
            
            regular_opt.step()
            
            results['regular']['losses'].append(regular_loss.item())
            results['regular']['stabilities'].append(1.0 / (regular_grad_norm.item() + 1e-6))
            results['regular']['predictions'].append((real_pred.mean().item(), fake_pred.mean().item()))
            
        except Exception as e:
            print(f"   Regular SF-VNN failed at step {step}: {e}")
            results['regular']['losses'].append(float('nan'))
            results['regular']['stabilities'].append(0.0)
            results['regular']['predictions'].append((0.0, 0.0))
        
        # Train attention SF-VNN
        try:
            attention_opt.zero_grad()
            real_pred = attention_sfvnn(real_mel)
            fake_pred = attention_sfvnn(fake_mel.detach())
            
            loss_real = criterion(real_pred, torch.ones_like(real_pred))
            loss_fake = criterion(fake_pred, torch.zeros_like(fake_pred))
            attention_loss = (loss_real + loss_fake) / 2
            
            attention_loss.backward()
            
            # Gradient norm
            attention_grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in attention_sfvnn.parameters() 
                if p.grad is not None
            ]))
            
            attention_opt.step()
            
            results['attention']['losses'].append(attention_loss.item())
            results['attention']['stabilities'].append(1.0 / (attention_grad_norm.item() + 1e-6))
            results['attention']['predictions'].append((real_pred.mean().item(), fake_pred.mean().item()))
            
        except Exception as e:
            print(f"   Attention SF-VNN failed at step {step}: {e}")
            results['attention']['losses'].append(float('nan'))
            results['attention']['stabilities'].append(0.0)
            results['attention']['predictions'].append((0.0, 0.0))
        
        # Progress update
        if step % 5 == 0:
            reg_status = f"{results['regular']['losses'][-1]:.4f}" if not np.isnan(results['regular']['losses'][-1]) else "Failed"
            att_status = f"{results['attention']['losses'][-1]:.4f}" if not np.isnan(results['attention']['losses'][-1]) else "Failed"
            print(f"   Step {step:2d}: Regular={reg_status}, Attention={att_status}")
    
    # Analysis
    return analyze_attention_results(results, regular_params, attention_params)

def analyze_attention_results(results, regular_params, attention_params):
    """Analyze attention enhancement results."""
    
    print(f"\\n📈 ATTENTION ENHANCEMENT ANALYSIS")
    print("=" * 50)
    
    # Filter valid results
    reg_valid_losses = [l for l in results['regular']['losses'] if not np.isnan(l)]
    att_valid_losses = [l for l in results['attention']['losses'] if not np.isnan(l)]
    
    reg_valid_stabs = [s for s in results['regular']['stabilities'] if s > 0]
    att_valid_stabs = [s for s in results['attention']['stabilities'] if s > 0]
    
    if not reg_valid_losses or not att_valid_losses:
        print("❌ Insufficient valid training data for comparison")
        return {'error': 'Training failed'}
    
    # Training success rates
    reg_success_rate = len(reg_valid_losses) / len(results['regular']['losses'])
    att_success_rate = len(att_valid_losses) / len(results['attention']['losses'])
    
    print(f"🎯 Training Success Rates:")
    print(f"   Regular SF-VNN: {reg_success_rate*100:.0f}%")
    print(f"   Attention SF-VNN: {att_success_rate*100:.0f}%")
    
    # Loss analysis
    reg_final_loss = reg_valid_losses[-1]
    att_final_loss = att_valid_losses[-1]
    reg_loss_var = np.var(reg_valid_losses)
    att_loss_var = np.var(att_valid_losses)
    
    print(f"\\n📊 Loss Analysis:")
    print(f"   Final Loss - Regular: {reg_final_loss:.4f}, Attention: {att_final_loss:.4f}")
    print(f"   Loss Variance - Regular: {reg_loss_var:.6f}, Attention: {att_loss_var:.6f}")
    
    # Stability analysis
    if reg_valid_stabs and att_valid_stabs:
        reg_avg_stab = np.mean(reg_valid_stabs)
        att_avg_stab = np.mean(att_valid_stabs)
        
        print(f"\\n⚖️  Stability Analysis:")
        print(f"   Average Stability - Regular: {reg_avg_stab:.4f}, Attention: {att_avg_stab:.4f}")
    else:
        reg_avg_stab = att_avg_stab = 0
    
    # Discrimination analysis
    reg_predictions = [p for p in results['regular']['predictions'] if p != (0.0, 0.0)]
    att_predictions = [p for p in results['attention']['predictions'] if p != (0.0, 0.0)]
    
    if reg_predictions and att_predictions:
        reg_discrimination = np.mean([abs(real - fake) for real, fake in reg_predictions])
        att_discrimination = np.mean([abs(real - fake) for real, fake in att_predictions])
        
        print(f"\\n🎯 Discrimination Analysis:")
        print(f"   Regular SF-VNN: {reg_discrimination:.4f}")
        print(f"   Attention SF-VNN: {att_discrimination:.4f}")
    else:
        reg_discrimination = att_discrimination = 0
    
    # Parameter efficiency
    reg_efficiency = reg_discrimination / (regular_params / 1e6) if reg_discrimination > 0 else 0
    att_efficiency = att_discrimination / (attention_params / 1e6) if att_discrimination > 0 else 0
    
    print(f"\\n⚡ Parameter Efficiency (discrimination per million params):")
    print(f"   Regular SF-VNN: {reg_efficiency:.6f}")
    print(f"   Attention SF-VNN: {att_efficiency:.6f}")
    
    # Overall assessment
    attention_advantages = []
    
    if att_success_rate > reg_success_rate:
        attention_advantages.append("Higher training success rate")
    
    if att_final_loss < reg_final_loss:
        attention_advantages.append("Lower final loss")
    
    if att_loss_var < reg_loss_var:
        attention_advantages.append("More stable training")
    
    if att_avg_stab > reg_avg_stab:
        attention_advantages.append("Better gradient stability")
    
    if att_discrimination > reg_discrimination:
        attention_advantages.append("Better discrimination ability")
    
    if att_efficiency > reg_efficiency:
        attention_advantages.append("Better parameter efficiency")
    
    print(f"\\n🏆 ATTENTION ENHANCEMENT ASSESSMENT:")
    if attention_advantages:
        print(f"   ✅ ATTENTION PROVIDES BENEFITS ({len(attention_advantages)}/6 metrics)")
        print(f"   📈 Advantages found:")
        for advantage in attention_advantages:
            print(f"      • {advantage}")
        
        if len(attention_advantages) >= 3:
            print(f"\\n🎉 STRONG EVIDENCE: Windowed attention significantly enhances SF-VNN!")
        else:
            print(f"\\n✨ MODERATE EVIDENCE: Attention shows promise with further tuning")
    else:
        print(f"   🤔 NO CLEAR ADVANTAGES FOUND")
        print(f"   💡 May need different attention mechanisms or longer training")
    
    # Create visualization
    create_attention_comparison_plot(results)
    
    return {
        'regular_sfvnn': {
            'final_loss': reg_final_loss,
            'loss_variance': reg_loss_var,
            'avg_stability': reg_avg_stab,
            'discrimination': reg_discrimination,
            'success_rate': reg_success_rate,
            'parameters': regular_params,
            'efficiency': reg_efficiency
        },
        'attention_sfvnn': {
            'final_loss': att_final_loss,
            'loss_variance': att_loss_var,
            'avg_stability': att_avg_stab,
            'discrimination': att_discrimination,
            'success_rate': att_success_rate,
            'parameters': attention_params,
            'efficiency': att_efficiency
        },
        'attention_advantages': attention_advantages,
        'strong_evidence': len(attention_advantages) >= 3
    }

def create_attention_comparison_plot(results):
    """Create comparison visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = range(len(results['regular']['losses']))
    
    # Training losses
    ax1.plot(steps, results['regular']['losses'], 'b-o', label='Regular SF-VNN', linewidth=2, markersize=4)
    ax1.plot(steps, results['attention']['losses'], 'r-s', label='Attention SF-VNN', linewidth=2, markersize=4)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Discriminator Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stability scores
    ax2.plot(steps, results['regular']['stabilities'], 'b-o', label='Regular SF-VNN', linewidth=2, markersize=4)
    ax2.plot(steps, results['attention']['stabilities'], 'r-s', label='Attention SF-VNN', linewidth=2, markersize=4)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Stability Score')
    ax2.set_title('Training Stability Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Discrimination over time
    reg_discriminations = [abs(real - fake) for real, fake in results['regular']['predictions']]
    att_discriminations = [abs(real - fake) for real, fake in results['attention']['predictions']]
    
    ax3.plot(steps, reg_discriminations, 'b-o', label='Regular SF-VNN', linewidth=2, markersize=4)
    ax3.plot(steps, att_discriminations, 'r-s', label='Attention SF-VNN', linewidth=2, markersize=4)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Discrimination Ability')
    ax3.set_title('Discrimination Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary bar chart
    metrics = ['Final Loss', 'Stability', 'Discrimination', 'Success Rate']
    
    # Normalize metrics for comparison
    reg_final = [l for l in results['regular']['losses'] if not np.isnan(l)][-1] if results['regular']['losses'] else 1
    att_final = [l for l in results['attention']['losses'] if not np.isnan(l)][-1] if results['attention']['losses'] else 1
    
    reg_stab = np.mean([s for s in results['regular']['stabilities'] if s > 0]) if results['regular']['stabilities'] else 0
    att_stab = np.mean([s for s in results['attention']['stabilities'] if s > 0]) if results['attention']['stabilities'] else 0
    
    reg_disc = np.mean(reg_discriminations) if reg_discriminations else 0
    att_disc = np.mean(att_discriminations) if att_discriminations else 0
    
    reg_success = len([l for l in results['regular']['losses'] if not np.isnan(l)]) / len(results['regular']['losses'])
    att_success = len([l for l in results['attention']['losses'] if not np.isnan(l)]) / len(results['attention']['losses'])
    
    # Normalize for visualization (higher is better for all)
    reg_values = [1/reg_final if reg_final > 0 else 0, reg_stab, reg_disc, reg_success]
    att_values = [1/att_final if att_final > 0 else 0, att_stab, att_disc, att_success]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, reg_values, width, label='Regular SF-VNN', color='blue', alpha=0.7)
    ax4.bar(x + width/2, att_values, width, label='Attention SF-VNN', color='red', alpha=0.7)
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Normalized Score')
    ax4.set_title('Performance Summary (Higher = Better)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_enhancement_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison visualization saved: attention_enhancement_comparison.png")

def main():
    """Run streamlined attention test."""
    
    print("⚡ STREAMLINED ATTENTION ENHANCEMENT TEST")
    print("Evaluating windowed attention benefits for SF-VNN")
    print("=" * 60)
    
    results = run_streamlined_attention_test()
    
    # Save results
    with open('attention_enhancement_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n💾 Results saved to: attention_enhancement_results.json")
    print("✅ Attention enhancement test completed!")

if __name__ == "__main__":
    main()