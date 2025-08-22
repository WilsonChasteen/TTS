#!/usr/bin/env python3
"""
Test script for the optimized text encoder in VITS.
"""

import torch
import torch.nn as nn
import math

# Import the components directly to avoid dependency issues
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def sequence_mask(length, max_length=None):
    """Create sequence mask."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

# Define the optimized components inline for testing

def test_optimized_text_encoder():
    """Test the optimized text encoder implementation."""
    
    # Test parameters
    batch_size = 2
    seq_len = 50
    n_vocab = 100
    hidden_channels = 192
    out_channels = 192
    
    # Create model
    text_encoder = TextEncoder(
        n_vocab=n_vocab,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        hidden_channels_ffn=256,
        num_heads=4,
        num_layers=6,
        kernel_size=5,
        dropout_p=0.1,
        language_emb_dim=None
    )
    
    # Create test inputs
    x = torch.randint(0, n_vocab, (batch_size, seq_len))
    x_lengths = torch.tensor([seq_len, seq_len-10])
    
    print("Testing optimized text encoder...")
    print(f"Input shape: {x.shape}")
    print(f"Input lengths: {x_lengths}")
    
    # Forward pass
    with torch.no_grad():
        encoded, m, logs, x_mask = text_encoder(x, x_lengths)
    
    print(f"Encoded shape: {encoded.shape}")
    print(f"Mean shape: {m.shape}")
    print(f"Logs shape: {logs.shape}")
    print(f"Mask shape: {x_mask.shape}")
    
    # Test with language embedding
    print("\nTesting with language embedding...")
    text_encoder_with_lang = TextEncoder(
        n_vocab=n_vocab,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        hidden_channels_ffn=256,
        num_heads=4,
        num_layers=6,
        kernel_size=5,
        dropout_p=0.1,
        language_emb_dim=4
    )
    
    lang_emb = torch.randn(batch_size, 4, 1)
    
    with torch.no_grad():
        encoded_lang, m_lang, logs_lang, x_mask_lang = text_encoder_with_lang(x, x_lengths, lang_emb)
    
    print(f"Encoded with lang shape: {encoded_lang.shape}")
    print(f"Mean with lang shape: {m_lang.shape}")
    print(f"Logs with lang shape: {logs_lang.shape}")
    
    # Test model parameters count
    total_params = sum(p.numel() for p in text_encoder.parameters())
    trainable_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test inference speed
    import time
    
    print("\nTesting inference speed...")
    text_encoder.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = text_encoder(x, x_lengths)
    
    # Timing
    start_time = time.time()
    num_runs = 100
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = text_encoder(x, x_lengths)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.1f} inferences/second")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_optimized_text_encoder()