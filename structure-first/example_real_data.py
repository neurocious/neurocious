#!/usr/bin/env python3
"""
Example: Using HiFi-GAN + SF-VNN with Real Audio Data

This script shows how to transition from dummy data to real audio files.
"""

import os
import torch
from pathlib import Path

# Import our HiFi-GAN + SF-VNN components
from hifi import (
    HiFiGANConfig, HiFiGANSFVNNDataset, HiFiGANSFVNNTrainer,
    AudioDatasetBuilder, HiFiGANSFVNNExperiment
)


def example_with_real_data():
    """Example showing how to use real audio data."""
    
    print("🎵 HiFi-GAN + SF-VNN with Real Audio Data")
    print("=" * 50)
    
    # Step 1: Specify your audio directory
    # REPLACE THIS PATH with your actual audio dataset directory
    audio_directory = "/path/to/your/audio/dataset"
    
    # For demonstration, let's check if a common dataset path exists
    possible_paths = [
        "/home/blasome/audio_dataset",
        "/tmp/audio_dataset", 
        "./audio_dataset",
        "../audio_dataset"
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if found_path:
        audio_directory = found_path
        print(f"✓ Found audio directory: {audio_directory}")
    else:
        print("⚠️  No audio directory found. Creating example setup...")
        print()
        print("TO USE WITH YOUR DATA:")
        print("1. Create a directory with your audio files:")
        print("   mkdir /path/to/your/audio_dataset")
        print("   # Copy your .wav/.flac/.mp3 files there")
        print()
        print("2. Update the audio_directory variable in this script")
        print("3. Run this script again")
        print()
        
        # Show what the code would look like with real data
        print("HERE'S HOW THE CODE WORKS WITH REAL DATA:")
        print("-" * 45)
        
        # Mock the process for demonstration
        demo_real_data_workflow()
        return
    
    # If we found a real directory, proceed with actual training
    actual_real_data_workflow(audio_directory)


def demo_real_data_workflow():
    """Demonstrate the workflow with mock data."""
    
    print("# Step 1: Build dataset from directory")
    print("train_files, val_files, test_files = AudioDatasetBuilder.build_dataset_from_directory(")
    print("    audio_dir='/path/to/your/audio_dataset',")
    print("    train_split=0.8,")
    print("    val_split=0.1,")
    print("    test_split=0.1")
    print(")")
    print()
    
    print("# Step 2: Validate audio files")
    print("train_files = AudioDatasetBuilder.validate_audio_files(train_files)")
    print("val_files = AudioDatasetBuilder.validate_audio_files(val_files)")
    print()
    
    print("# Step 3: Create configuration")
    print("config = HiFiGANConfig(")
    print("    sample_rate=22050,")
    print("    n_mels=80,")
    print("    batch_size=16,")
    print("    num_epochs=100,")
    print("    lambda_structural=1.0,")
    print("    sfvnn_multiscale=True")
    print(")")
    print()
    
    print("# Step 4: Create datasets")
    print("train_dataset = HiFiGANSFVNNDataset(train_files, config, training=True)")
    print("val_dataset = HiFiGANSFVNNDataset(val_files, config, training=False)")
    print()
    
    print("# Step 5: Create trainer and train")
    print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    print("trainer = HiFiGANSFVNNTrainer(config, train_dataset, val_dataset, device)")
    print("history = trainer.train()")
    print()
    
    print("# Step 6: Run experiments")
    print("experiment = HiFiGANSFVNNExperiment(train_files, val_files)")
    print("results = experiment.run_comparative_study()")
    print()
    
    print("EXAMPLE OUTPUT WITH REAL DATA:")
    print("-" * 30)
    print("Found 1000 audio files in /path/to/your/audio_dataset")
    print("Split: 800 train, 100 val, 100 test")
    print("Validated 995/1000 audio files")
    print("🎵 Starting HiFi-GAN + SF-VNN Training")
    print("Generator parameters: 13,914,721")
    print("Discriminator parameters: 2,847,233")
    print("Training for 100 epochs")
    print("Epoch 0, Batch 0: G_total: 45.2341, G_mel: 42.1234, ...")
    print("...")


def actual_real_data_workflow(audio_directory: str):
    """Run the actual workflow with a real audio directory."""
    
    print(f"🎵 Processing real audio data from: {audio_directory}")
    
    # Step 1: Build dataset
    try:
        train_files, val_files, test_files = AudioDatasetBuilder.build_dataset_from_directory(
            audio_dir=audio_directory,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1
        )
        
        # Step 2: Validate files
        train_files = AudioDatasetBuilder.validate_audio_files(train_files)
        val_files = AudioDatasetBuilder.validate_audio_files(val_files)
        
        if len(train_files) == 0:
            print("❌ No valid training files found!")
            return
            
        print(f"✓ Ready to train with {len(train_files)} training files")
        
        # Step 3: Quick test with small batch
        print("\n🧪 Testing with real data (small batch)...")
        
        config = HiFiGANConfig(
            sample_rate=22050,
            batch_size=2,  # Small for testing
            num_epochs=1   # Just one epoch for demo
        )
        
        # Create datasets
        train_dataset = HiFiGANSFVNNDataset(
            train_files[:10],  # Just first 10 files for demo
            config, 
            training=True
        )
        val_dataset = HiFiGANSFVNNDataset(
            val_files[:5],   # Just first 5 files for demo
            config, 
            training=False
        )
        
        print(f"✓ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Test loading one batch
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
        real_audio, real_mel = next(iter(train_loader))
        
        print(f"✓ Successfully loaded real data:")
        print(f"  Audio shape: {real_audio.shape}")
        print(f"  Mel shape: {real_mel.shape}")
        print(f"  Audio range: [{real_audio.min():.3f}, {real_audio.max():.3f}]")
        print(f"  Mel range: [{real_mel.min():.3f}, {real_mel.max():.3f}]")
        
        print("\n🚀 Ready for full training!")
        print("To start full training, increase:")
        print("  - batch_size to 16-32")
        print("  - num_epochs to 100-200") 
        print("  - Use all files (not just first 10)")
        
    except Exception as e:
        print(f"❌ Error processing audio directory: {e}")
        print("Make sure the directory contains valid audio files (.wav, .flac, .mp3)")


def create_minimal_test_dataset():
    """Create a minimal test dataset for demonstration."""
    
    print("\n🛠️  Creating minimal test dataset...")
    
    # Create a test directory
    test_dir = Path("./test_audio_dataset")
    test_dir.mkdir(exist_ok=True)
    
    # Generate some dummy audio files (very short)
    import torchaudio
    
    sample_rate = 22050
    duration = 1.0  # 1 second
    
    for i in range(5):
        # Generate simple sine wave audio
        t = torch.linspace(0, duration, int(sample_rate * duration))
        frequency = 440 + i * 110  # A4, B4, C#5, etc.
        audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        
        # Add some noise for realism
        audio += 0.1 * torch.randn_like(audio)
        
        # Save as WAV file
        filepath = test_dir / f"test_audio_{i:03d}.wav"
        torchaudio.save(str(filepath), audio, sample_rate)
    
    print(f"✓ Created {len(list(test_dir.glob('*.wav')))} test audio files in {test_dir}")
    print("You can now run:")
    print(f"  python3 example_real_data.py")
    print("And it will use these test files!")
    
    return str(test_dir)


def main():
    """Main function to demonstrate real data usage."""
    
    # Check if we have any audio data
    if not any(os.path.exists(p) for p in ["/home/blasome/audio_dataset", "./audio_dataset", "./test_audio_dataset"]):
        print("No audio dataset found. Would you like to create a minimal test dataset? (y/n)")
        response = input().lower().strip()
        
        if response == 'y' or response == 'yes':
            test_dir = create_minimal_test_dataset()
            print(f"\n🎵 Now running with test dataset...")
            actual_real_data_workflow(test_dir)
        else:
            print("Showing mock workflow instead...")
            demo_real_data_workflow()
    else:
        example_with_real_data()


if __name__ == "__main__":
    main()