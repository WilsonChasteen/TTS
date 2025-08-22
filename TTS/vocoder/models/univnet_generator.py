# Ultra-High Quality Real-Time Voice Encoder
# Redesigned for 0.9+ quality at ~40ms latency
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.nn.utils.parametrize import remove_parametrizations
import math

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
            
        self.norm = nn.GroupNorm(min(32, channels), channels)
        
    def forward(self, x, c=None):
        residual = x
        
        # Main path
        x = self.conv1(x)
        x = self.attention(x)
        
        # Conditioning
        if c is not None and hasattr(self, 'cond_proj'):
            cond = self.cond_proj(c)
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

class UltraHighQualityVoiceEncoder(nn.Module):
    """Ultra-high quality real-time voice encoder"""
    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        hidden_channels=512,
        upsample_factors=[8, 4, 4, 2],  # More gradual upsampling
        lvc_blocks_per_stage=2,
        cond_channels=0
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            nn.LeakyReLU(LRELU_SLOPE)
        )
        
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
                block = StreamlinedLVCBlock(
                    current_channels,
                    cond_channels=cond_channels if j == 0 else 0,
                    dilation=2 ** j
                )
                stage_blocks.append(block)
            self.lvc_blocks.append(stage_blocks)
        
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
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x, c=None):
        # Input projection
        x = self.input_proj(x)
        
        # Upsampling stages
        for upsample, blocks in zip(self.upsample_layers, self.lvc_blocks):
            x = upsample(x)
            for block in blocks:
                x = block(x, c)
                
        # Output projection
        return self.output_proj(x)
    
    @torch.no_grad()
    def inference(self, x, c=None):
        return self.forward(x, c)
    
    def remove_weight_norm(self):
        """Remove weight normalization for deployment"""
        for module in self.modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'parametrizations'):
                try:
                    remove_parametrizations(module, "weight")
                except:
                    pass

# Performance optimization techniques
def optimize_for_inference(model):
    """Apply inference optimizations"""
    model.eval()
    model.remove_weight_norm()
    
    # Fusion optimizations
    torch.jit.optimize_for_inference(torch.jit.script(model))
    
    return model

# Example usage
if __name__ == "__main__":
    # Create model
    model = UltraHighQualityVoiceEncoder()
    
    # Test input
    x = torch.randn(1, 80, 100)  # (batch, channels, length)
    
    # Test inference
    with torch.no_grad():
        output = model.inference(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
    # Measure performance
    import time
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model.inference(x)
    end = time.time()
    
    avg_time = (end - start) * 1000 / 10
    print(f"Average inference time: {avg_time:.2f}ms")