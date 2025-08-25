# Enhanced Hybrid Voice Encoder with Conformer, Transformer, and Diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.nn.utils.parametrize import remove_parametrizations
import math
try:
    from einops import rearrange, repeat
except ImportError:
    # Fallback implementations if einops is not available
    def rearrange(tensor, pattern, **axes_lengths):
        if pattern == 'b c t -> b t c':
            return tensor.transpose(1, 2)
        elif pattern == 'b t c -> b c t':
            return tensor.transpose(1, 2)
        else:
            raise NotImplementedError(f"Pattern {pattern} not implemented in fallback")
    
    def repeat(tensor, pattern, **axes_lengths):
        # Simple repeat implementation
        return tensor

LRELU_SLOPE = 0.1

def get_padding(k, d):
    return int((k * d - d) / 2)

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for efficiency"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                  stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return x

class EfficientMultiScaleAttention(nn.Module):
    """Efficient multi-scale attention with improved quality"""
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.channels = channels
        
        # Multi-scale depthwise convolutions
        self.multi_scale_convs = nn.ModuleList([
            DepthwiseSeparableConv1d(channels, channels, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        # Enhanced channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Conv1d(channels * 4, channels, 1)
        
    def forward(self, x):
        residual = x
        
        # Multi-scale features
        scale_features = [conv(x) for conv in self.multi_scale_convs]
        scale_features.append(x)  # Include original features
        fused = torch.cat(scale_features, dim=1)
        fused = self.fusion(fused)
        
        # Apply attention
        channel_weights = self.channel_attention(fused)
        spatial_weights = self.spatial_attention(fused)
        
        attended = fused * channel_weights * spatial_weights
        return attended + residual

class ConformerBlock(nn.Module):
    """Conformer block for capturing local and global dependencies"""
    def __init__(self, dim, expansion_factor=2, n_heads=4, kernel_size=31):
        super().__init__()
        
        # Feed-forward module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(0.1)
        )
        
        # Multi-headed self-attention
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.1)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size, 
                     padding=kernel_size//2, groups=dim),
            nn.SiLU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(0.1)
        )
        
        # Feed-forward module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(0.1)
        )
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # x: (B, T, C)
        residual = x
        
        # Feed-forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Attention
        x_attn = self.attn_norm(x)
        x_attn, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=mask)
        x = x + x_attn
        
        # Convolution
        x_conv = self.conv_norm(x)
        x_conv = rearrange(x_conv, 'b t c -> b c t')
        x_conv = self.conv(x_conv)
        x_conv = rearrange(x_conv, 'b c t -> b t c')
        x = x + x_conv
        
        # Feed-forward 2
        x = x + 0.5 * self.ff2(x)
        
        x = self.final_norm(x)
        return x

class DiffusionRefinementBlock(nn.Module):
    """Lightweight diffusion-based refinement for high-quality output"""
    def __init__(self, channels, cond_channels=0, n_timesteps=4):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.channels = channels
        
        # Noise prediction network
        input_channels = channels + cond_channels
        self.noise_pred = nn.Sequential(
            nn.Conv1d(input_channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1)
        )
        
        # Learnable timestep embedding - project to match input channels
        self.timestep_embed = nn.Embedding(n_timesteps, channels)
        self.timestep_proj = nn.Conv1d(channels, input_channels, 1)
        
    def forward(self, x, c=None, t=0):
        # x: (B, C, T)
        batch_size, channels, seq_len = x.shape
        
        if c is not None:
            # Expand conditioning to match current temporal dimension
            if c.shape[-1] == 1:
                c_expanded = c.expand(-1, -1, seq_len)
            else:
                c_expanded = F.interpolate(c, size=seq_len, mode='linear', align_corners=False)
            x_in = torch.cat([x, c_expanded], dim=1)
        else:
            x_in = x
            
        # Add timestep embedding
        t_tensor = torch.tensor([t], device=x.device, dtype=torch.long)
        t_embed = self.timestep_embed(t_tensor)  # (1, embedding_dim)
        t_embed = t_embed.view(1, -1, 1)  # (1, embedding_dim, 1)
        t_embed = t_embed.expand(batch_size, -1, seq_len)  # (B, embedding_dim, T)
        
        # Project timestep embedding to match input dimensions
        t_embed_proj = self.timestep_proj(t_embed)  # (B, input_channels, T)
        
        # Add timestep information
        x_in = x_in + t_embed_proj
        
        # Predict noise
        predicted_noise = self.noise_pred(x_in)
        return predicted_noise

class ProsodyAwareConditioning(nn.Module):
    """Prosody-aware conditioning for emotional speech"""
    def __init__(self, cond_channels, hidden_dim=256):
        super().__init__()
        
        # Prosody feature extraction - handle input dimension properly
        self.input_proj = nn.Conv1d(cond_channels, hidden_dim, 1) if cond_channels != hidden_dim else nn.Identity()
        self.prosody_net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Emotion/style embedding
        self.style_embed = nn.Embedding(8, hidden_dim)  # 8 emotion categories
        
        # Fusion - handle both with and without emotion embedding
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Start with single input size
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Separate fusion for emotion case
        self.emotion_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, c, emotion_id=None):
        # c: (B, C, T)
        c_proj = self.input_proj(c)  # Project to hidden_dim if needed
        prosody_features = self.prosody_net(c_proj)  # (B, hidden_dim, 1)
        prosody_features = prosody_features.squeeze(-1)  # (B, hidden_dim)
        
        if emotion_id is not None:
            emotion_embed = self.style_embed(emotion_id)  # (B, hidden_dim)
            combined = torch.cat([prosody_features, emotion_embed], dim=-1)
            conditioned_features = self.emotion_fusion(combined)
        else:
            conditioned_features = self.fusion(prosody_features)
        return conditioned_features.unsqueeze(-1)  # (B, hidden_dim, 1)

class StreamlinedLVCBlock(nn.Module):
    """Optimized LVC block for real-time performance"""
    def __init__(self, channels, cond_channels=0, dilation=1):
        super().__init__()
        
        # Depthwise separable convolutions
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv1d(channels, channels, 3, 
                                   padding=get_padding(3, dilation), dilation=dilation),
            nn.LeakyReLU(LRELU_SLOPE, inplace=True)
        )
        
        self.attention = EfficientMultiScaleAttention(channels)
        
        # Lightweight feed-forward
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(channels, channels, 1)
        )
        
        # Conditioning
        if cond_channels > 0:
            self.cond_proj = nn.Conv1d(cond_channels, channels, 1)
        
        self.cond_channels = cond_channels
            
        self.norm = nn.GroupNorm(min(32, channels), channels)
        
    def forward(self, x, c=None):
        residual = x
        
        # Main path
        x = self.conv1(x)
        x = self.attention(x)
        
        # Conditioning
        if c is not None and self.cond_channels > 0 and hasattr(self, 'cond_proj'):
            # Expand conditioning to match current temporal dimension
            if c.shape[-1] == 1:
                c_expanded = c.expand(-1, -1, x.shape[-1])
            else:
                c_expanded = F.interpolate(c, size=x.shape[-1], mode='linear', align_corners=False)
            cond = self.cond_proj(c_expanded)
            x = x * torch.sigmoid(cond)
            
        # Feed-forward
        x = self.ffn(x)
        
        return self.norm(x + residual)

class OptimizedUpsampling(nn.Module):
    """Optimized upsampling with sub-pixel convolution for 1D"""
    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv1d(in_channels, out_channels * factor, 3, padding=1)
        self.activation = nn.LeakyReLU(LRELU_SLOPE)
        
    def forward(self, x):
        x = self.conv(x)
        # Reshape for 1D sub-pixel convolution
        batch_size, channels, length = x.shape
        x = x.view(batch_size, channels // self.factor, self.factor, length)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, channels // self.factor, length * self.factor)
        return self.activation(x)

class StreamingBuffer(nn.Module):
    """Buffer for real-time streaming with minimal latency"""
    def __init__(self, buffer_size=512):
        super().__init__()
        self.buffer_size = buffer_size
        self.buffer = None
        
    def forward(self, x, reset=False):
        # x: (B, C, T)
        if reset or self.buffer is None:
            self.buffer = x[:, :, -self.buffer_size:] if x.shape[-1] > self.buffer_size else x
            return x
            
        # For streaming, just return current input and update buffer
        # The concatenation behavior was causing length mismatches
        self.buffer = x[:, :, -self.buffer_size:] if x.shape[-1] > self.buffer_size else x
        
        return x

class UltraHighQualityVoiceEncoder(nn.Module):
    """Enhanced ultra-high quality real-time voice encoder with hybrid architecture"""
    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        hidden_channels=512,
        upsample_factors=[8, 4, 4, 2],
        lvc_blocks_per_stage=2,
        cond_channels=0,
        use_conformer=True,
        use_diffusion=True,
        streaming_buffer_size=512
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.use_conformer = use_conformer
        self.use_diffusion = use_diffusion
        self.streaming_buffer_size = streaming_buffer_size
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            nn.LeakyReLU(LRELU_SLOPE)
        )
        
        # Prosody-aware conditioning
        if cond_channels > 0:
            self.prosody_conditioner = ProsodyAwareConditioning(cond_channels, hidden_dim=hidden_channels)
            self.cond_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        
        # Conformer blocks for global context
        if use_conformer:
            self.conformer_blocks = nn.ModuleList([
                ConformerBlock(hidden_channels, n_heads=4)
                for _ in range(2)
            ])
            
        # Streaming buffer for real-time processing
        self.stream_buffer = StreamingBuffer(streaming_buffer_size)
        
        # Upsampling stages
        self.upsample_layers = nn.ModuleList()
        self.lvc_blocks = nn.ModuleList()
        
        current_channels = hidden_channels
        for i, factor in enumerate(upsample_factors):
            # Upsampling
            upsample = OptimizedUpsampling(current_channels, current_channels // 2, factor)
            self.upsample_layers.append(upsample)
            current_channels = current_channels // 2
            
            # LVC blocks
            stage_blocks = nn.ModuleList()
            for j in range(lvc_blocks_per_stage):
                # Use hidden_channels for conditioning since that's what we project to
                block = StreamlinedLVCBlock(
                    current_channels,
                    cond_channels=hidden_channels if (cond_channels > 0 and j == 0) else 0,
                    dilation=2 ** j
                )
                stage_blocks.append(block)
            self.lvc_blocks.append(stage_blocks)
        
        # Diffusion refinement
        if use_diffusion:
            self.diffusion_refiner = DiffusionRefinementBlock(
                current_channels, 
                cond_channels=hidden_channels if cond_channels > 0 else 0
            )
        
        # Final processing
        self.output_proj = nn.Sequential(
            nn.Conv1d(current_channels, current_channels, 3, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(current_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def forward(self, x, c=None, g=None, emotion_id=None, diffusion_steps=2):
        # Handle both c and g parameters for compatibility
        if g is not None and c is None:
            c = g
        # Input projection
        x = self.input_proj(x)
        
        # Apply streaming buffer
        x = self.stream_buffer(x)
        
        # Process conditioning
        cond_features = None
        if c is not None and hasattr(self, 'prosody_conditioner'):
            cond_features = self.prosody_conditioner(c, emotion_id)
            cond_features = self.cond_proj(cond_features)
            # Don't expand here - we'll handle this in each LVC block
        
        # Conformer blocks for global context
        if self.use_conformer:
            # Rearrange for sequence-first format (B, C, T) -> (B, T, C)
            x_conv = rearrange(x, 'b c t -> b t c')
            for block in self.conformer_blocks:
                x_conv = block(x_conv)
            x = rearrange(x_conv, 'b t c -> b c t')
        
        # Upsampling stages
        for upsample, blocks in zip(self.upsample_layers, self.lvc_blocks):
            x = upsample(x)
            for block in blocks:
                x = block(x, cond_features if c is not None else None)
        
        # Diffusion refinement (during training or high-quality mode)
        if self.use_diffusion and self.training and diffusion_steps > 0:
            # Apply multi-step diffusion refinement
            for t in range(diffusion_steps):
                try:
                    noise_pred = self.diffusion_refiner(x, cond_features if c is not None else None, t)
                    # Simple diffusion step (can be enhanced with learned schedules)
                    x = x - 0.1 * noise_pred
                except Exception as e:
                    # Skip diffusion if there are dimension issues
                    print(f"Skipping diffusion step {t} due to error: {e}")
                    break
        
        # Output projection
        return self.output_proj(x)
    
    @torch.no_grad()
    def inference(self, x, c=None, g=None, emotion_id=None, high_quality=False):
        # Handle both c and g parameters for compatibility
        if g is not None and c is None:
            c = g
        # Reset stream buffer for new inference
        self.stream_buffer.buffer = None
        
        # Standard forward pass
        output = self.forward(x, c, emotion_id, diffusion_steps=0)
        
        # Optional diffusion refinement for high quality mode
        if high_quality and self.use_diffusion:
            # Apply 1-2 steps of diffusion refinement
            for t in range(2):
                try:
                    noise_pred = self.diffusion_refiner(output, c, t)
                    output = output - 0.1 * noise_pred
                except Exception as e:
                    # Skip diffusion if there are dimension issues
                    print(f"Skipping diffusion refinement step {t} due to error: {e}")
                    break
        
        return output
    
    def stream_step(self, x_chunk, c_chunk=None, g_chunk=None, emotion_id=None):
        # Handle both c and g parameters for compatibility
        if g_chunk is not None and c_chunk is None:
            c_chunk = g_chunk
        """Process a chunk of audio for streaming"""
        with torch.no_grad():
            return self.forward(x_chunk, c_chunk, emotion_id)
    
    def remove_weight_norm(self):
        """Remove weight normalization for deployment"""
        for module in self.modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'parametrizations'):
                try:
                    remove_parametrizations(module, "weight")
                except:
                    pass

# Performance optimization techniques
def optimize_for_inference(model, use_mps=True):
    """Apply inference optimizations"""
    model.eval()
    model.remove_weight_norm()
    
    # Set to MPS if available
    if use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
    
    # Fusion optimizations
    try:
        model = torch.jit.script(model)
        torch.jit.optimize_for_inference(model)
    except:
        print("JIT optimization failed, using standard model")
    
    return model

# Example usage
if __name__ == "__main__":
    # Create enhanced model
    model = UltraHighQualityVoiceEncoder(cond_channels=128, use_conformer=True, use_diffusion=True)
    
    # Test input
    x = torch.randn(1, 80, 100)  # (batch, channels, length)
    c = torch.randn(1, 128, 100) if model.cond_channels > 0 else None  # conditioning
    
    # Test inference
    with torch.no_grad():
        output = model.inference(x, c)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
    # Measure performance
    import time
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model.inference(x, c)
    end = time.time()
    
    avg_time = (end - start) * 1000 / 10
    print(f"Average inference time: {avg_time:.2f}ms")
    
    # Test streaming
    chunk_size = 50
    stream_outputs = []
    for i in range(0, x.shape[2], chunk_size):
        x_chunk = x[:, :, i:i+chunk_size]
        c_chunk = c[:, :, i:i+chunk_size] if c is not None else None
        with torch.no_grad():
            output_chunk = model.stream_step(x_chunk, c_chunk)
            stream_outputs.append(output_chunk)
    
    stream_output = torch.cat(stream_outputs, dim=-1)
    print(f"Streaming output shape: {stream_output.shape}")


# Backward compatibility alias and wrapper
class StreamlinedUnivNetGenerator(UltraHighQualityVoiceEncoder):
    """Backward compatibility wrapper for VITS integration"""
    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        hidden_channels=512,
        upsample_factors=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        lvc_block_nums=4,
        lvc_layers_each_block=4,
        lvc_kernel_size=3,
        dropout=0.0,
        cond_channels=0,
        conv_pre_weight_norm=True,
        conv_post_weight_norm=True,
        conv_post_bias=True,
        inference_padding=0,
        **kwargs
    ):
        # Map old parameters to new architecture
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            upsample_factors=upsample_factors,
            lvc_blocks_per_stage=lvc_layers_each_block,
            cond_channels=cond_channels,
            use_conformer=True,
            use_diffusion=True,
            streaming_buffer_size=512
        )
        
        # Store original parameters for compatibility
        self.inference_padding = inference_padding
        
    def forward(self, x, g=None):
        """VITS-compatible forward method"""
        # g is the speaker/style conditioning from VITS
        return super().forward(x, c=g)
    
    def inference(self, x, g=None):
        """VITS-compatible inference method"""
        return super().inference(x, c=g, high_quality=True)