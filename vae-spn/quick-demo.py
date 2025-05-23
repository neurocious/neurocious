#!/usr/bin/env python3
"""
Quick Start Example for Neurocious System
=========================================

This demonstrates the core functionality and typical usage patterns.
"""

import torch
import numpy as np
import asyncio
from neurocious_main import NeurociousSystem
from neurocious_core import CoTrainingConfig, FieldParameters

async def quick_demo():
    """Demonstrate basic usage"""
    
    # 1. CONFIGURATION
    config = CoTrainingConfig(
        batch_size=8,           # Smaller for demo
        learning_rate=0.001,
        beta=0.7,              # KL weight
        gamma=0.5,             # Narrative continuity
        delta=0.3,             # Field alignment
        policy_weight=0.5,     # Policy prediction
        reflex_weight=0.3,     # Reflex behavior
        prediction_weight=0.4  # Future prediction
    )
    
    # 2. INITIALIZE SYSTEM
    print("Initializing Neurocious system...")
    system = NeurociousSystem(
        config=config,
        device='cpu',  # Use 'cuda' if available
        checkpoint_dir='./checkpoints',
        log_dir='./logs'
    )
    
    # 3. GENERATE SAMPLE DATA
    # In practice, this would be your real sequential data
    print("Generating sample data...")
    
    def generate_sequential_data(num_sequences=50, seq_length=8):
        """Generate sample sequential data"""
        data = {
            'sequences': [],
            'rewards': [],
            'actions': [],
            'reactions': [],
            'future_states': []
        }
        
        for _ in range(num_sequences):
            # Simulate a sequence where later states depend on earlier ones
            sequence = []
            base_state = np.random.randn(784) * 0.1
            
            for t in range(seq_length):
                # Add temporal dependency
                noise = np.random.randn(784) * 0.05
                drift = base_state * 0.1 * t  # Linear drift
                state = base_state + drift + noise
                sequence.append(state)
            
            # Rewards that increase with sequence coherence
            rewards = [0.5 + 0.1 * t + np.random.normal(0, 0.1) 
                      for t in range(seq_length)]
            
            # Actions (could be motor commands, decisions, etc.)
            actions = [np.random.randn(10) for _ in range(seq_length)]
            
            # Reactions (binary reflexes)
            reactions = [np.random.randint(0, 2, 5).astype(float) 
                        for _ in range(seq_length)]
            
            # Future states (predictions)
            future_states = [np.random.randn(4) for _ in range(seq_length)]
            
            data['sequences'].append(sequence)
            data['rewards'].append(rewards)
            data['actions'].append(actions)
            data['reactions'].append(reactions)
            data['future_states'].append(future_states)
        
        return data
    
    training_data = generate_sequential_data(50, 8)
    validation_data = generate_sequential_data(10, 8)
    
    # 4. QUICK TRAINING RUN
    print("Running quick training (5 epochs)...")
    await system.train(
        training_data=training_data,
        validation_data=validation_data,
        epochs=5,
        save_freq=2,
        eval_freq=1
    )
    
    # 5. INFERENCE EXAMPLE
    print("\nRunning inference example...")
    
    # Create a test sequence
    test_sequence = [torch.randn(784) for _ in range(5)]
    
    # Run inference
    results = system.inference(test_sequence, return_explanations=True)
    
    print(f"Routing confidence: {results['confidence'].item():.3f}")
    print(f"Top belief region: {results['explanation']['belief_label']}")
    print(f"Field parameters: {results['field_parameters']}")
    print(f"Explanation: {results['explanation']['justification']}")
    
    # 6. FIELD FLOW ANALYSIS
    print("\nAnalyzing field flow...")
    
    test_state = results['latent_representation']
    flow_pattern = system.analyze_field_flow(test_state, steps=10)
    
    print(f"Flow stability: {flow_pattern.stability:.3f}")
    print(f"Local curvature: {flow_pattern.local_curvature:.3f}")
    print(f"Local entropy: {flow_pattern.local_entropy:.3f}")
    
    # 7. WORLD BRANCHING
    print("\nSimulating world branches...")
    
    current_field_params = FieldParameters(**results['field_parameters'])
    branches = system.simulate_world_branches(current_field_params, num_branches=3)
    
    print("Possible world branches:")
    for branch in branches:
        print(f"  Branch {branch['branch_id']}: "
              f"probability={branch['probability']:.3f}")
    
    print("\nDemo complete! Check ./logs/ for visualizations")

# Real-world usage patterns
def create_from_mnist():
    """Example: Using with MNIST-like data"""
    from torchvision import datasets, transforms
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Convert to sequences (group consecutive digits)
    sequences = []
    rewards = []
    
    sequence_length = 10
    for i in range(0, len(dataset) - sequence_length, sequence_length):
        sequence = []
        sequence_rewards = []
        
        for j in range(sequence_length):
            img, label = dataset[i + j]
            sequence.append(img.numpy())
            # Reward based on digit value (example objective)
            sequence_rewards.append(label / 9.0)  # Normalize to [0,1]
        
        sequences.append(sequence)
        rewards.append(sequence_rewards)
        
        if len(sequences) >= 100:  # Limit for demo
            break
    
    return {
        'sequences': sequences,
        'rewards': rewards,
        'actions': [[np.zeros(10) for _ in range(sequence_length)] for _ in sequences],
        'reactions': [[np.zeros(5) for _ in range(sequence_length)] for _ in sequences],
        'future_states': [[np.zeros(4) for _ in range(sequence_length)] for _ in sequences]
    }

def create_from_timeseries():
    """Example: Using with time series data"""
    import pandas as pd
    
    # Generate synthetic time series
    t = np.linspace(0, 100, 1000)
    signal = np.sin(0.1 * t) + 0.5 * np.sin(0.3 * t) + np.random.normal(0, 0.1, len(t))
    
    # Convert to sequences
    sequences = []
    rewards = []
    window_size = 50
    
    for i in range(len(signal) - window_size):
        # Extract window
        window = signal[i:i + window_size]
        
        # Pad to match expected input dimension (784 for compatibility)
        padded_window = np.zeros(784)
        padded_window[:len(window)] = window
        
        # Create sequence of overlapping windows
        sequence = []
        for j in range(10):  # 10 timesteps
            offset = j * 5
            if i + offset + window_size < len(signal):
                sub_window = signal[i + offset:i + offset + window_size]
                padded_sub = np.zeros(784)
                padded_sub[:len(sub_window)] = sub_window
                sequence.append(padded_sub)
            else:
                sequence.append(padded_window)
        
        # Reward based on trend (increasing = higher reward)
        trend = np.polyfit(range(len(window)), window, 1)[0]
        reward_seq = [max(0, trend + 0.5) for _ in range(10)]
        
        sequences.append(sequence)
        rewards.append(reward_seq)
        
        if len(sequences) >= 100:
            break
    
    return {
        'sequences': sequences,
        'rewards': rewards,
        'actions': [[np.zeros(10) for _ in range(10)] for _ in sequences],
        'reactions': [[np.zeros(5) for _ in range(10)] for _ in sequences],
        'future_states': [[np.zeros(4) for _ in range(10)] for _ in sequences]
    }

if __name__ == "__main__":
    asyncio.run(quick_demo())