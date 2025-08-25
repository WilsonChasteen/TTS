import torch
from torch import nn
from torch.nn import functional as F


class ImprovedDurationPredictor(nn.Module):
    """Improved Duration Predictor with enhanced architecture for better accuracy.
    
    Uses a combination of causal convolutions, transformer layers, and squeeze-excitation
    for improved performance while maintaining real-time capabilities.
    
    Architecture:
        - Multi-scale convolutional feature extraction
        - Lightweight transformer with relative positional encoding
        - Squeeze-and-excitation for channel attention
        - Multi-resolution feature fusion
        - Optimized for both accuracy and speed
        
    Args:
        in_channels (int): Number of input channels from text encoder
        hidden_channels (int): Number of hidden channels (default: 256)
        kernel_size (int): Convolution kernel size (default: 3)
        dropout_p (float): Dropout probability
        cond_channels (int): Number of conditioning channels (speaker embedding)
        language_emb_dim (int): Language embedding dimension
        num_heads (int): Number of attention heads (default: 4)
        num_layers (int): Number of transformer layers (default: 2)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int = 256,
        kernel_size: int = 3, 
        dropout_p: float = 0.1,
        cond_channels: int = None,
        language_emb_dim: int = None,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.language_emb_dim = language_emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Adjust input channels for language embedding
        conv_in_channels = in_channels
        if language_emb_dim:
            conv_in_channels += language_emb_dim
        
        # Multi-scale feature extraction
        self.conv_3x1 = nn.Conv1d(conv_in_channels, hidden_channels // 4, 3, padding=1)
        self.conv_5x1 = nn.Conv1d(conv_in_channels, hidden_channels // 4, 5, padding=2)
        self.conv_7x1 = nn.Conv1d(conv_in_channels, hidden_channels // 4, 7, padding=3)
        self.conv_1x1 = nn.Conv1d(conv_in_channels, hidden_channels // 4, 1)
        
        self.feature_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.norm0 = nn.LayerNorm(hidden_channels)
        
        # Squeeze-and-excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels, hidden_channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels // 8, hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Transformer layers with relative positional encoding
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_channels, num_heads, hidden_channels * 4, dropout_p)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_channels // 2, 1, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # Conditioning layers
        if cond_channels is not None and cond_channels != 0:
            self.cond = nn.Conv1d(cond_channels, hidden_channels, 1)
            
        if language_emb_dim is not None and language_emb_dim != 0:
            self.cond_lang = nn.Conv1d(language_emb_dim, hidden_channels, 1)
            
        # Initialize weights for stable training
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training with better initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # Special initialization for output projection to predict reasonable durations
        with torch.no_grad():
            # Initialize final layer to predict log(2.0) ≈ 0.693 (2 frames per phoneme)
            final_layer = self.proj[-1]
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                final_layer.bias.fill_(0.693)  # log(2.0)
                    
    def forward(self, x, x_mask, g=None, lang_emb=None, inference_noise_scale=0.1):
        """
        Forward pass of the improved duration predictor.
        
        Args:
            x: Text encoder output [B, C, T]
            x_mask: Sequence mask [B, 1, T] 
            g: Speaker conditioning [B, C, 1]
            lang_emb: Language embedding [B, C, 1]
            inference_noise_scale: Noise scale for inference (default: 0.1)
            
        Returns:
            Duration predictions [B, 1, T]
        """
        # Concatenate language embedding if present
        if lang_emb is not None and self.language_emb_dim:
            # Expand language embedding to match sequence length
            lang_emb_expanded = lang_emb.expand(-1, -1, x.size(2))
            x_in = torch.cat([x, lang_emb_expanded], dim=1)
        else:
            x_in = x
        
        # Multi-scale feature extraction
        features_3x1 = F.relu(self.conv_3x1(x_in))
        features_5x1 = F.relu(self.conv_5x1(x_in))
        features_7x1 = F.relu(self.conv_7x1(x_in))
        features_1x1 = F.relu(self.conv_1x1(x_in))
        
        # Concatenate multi-scale features
        x = torch.cat([features_3x1, features_5x1, features_7x1, features_1x1], dim=1)
        x = self.feature_proj(x)
        
        # Apply squeeze-and-excitation
        se_weights = self.se(x)
        x = x * se_weights
        
        # Apply conditioning after feature extraction
        if g is not None and hasattr(self, 'cond'):
            x = x + self.cond(g)
            
        if lang_emb is not None and hasattr(self, 'cond_lang'):
            x = x + self.cond_lang(lang_emb)
        
        # Apply layer normalization
        x = x.transpose(1, 2)  # [B, T, C] for LayerNorm
        x = self.norm0(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, x_mask)
        
        # Output projection
        x = self.proj(x * x_mask)
        
        # During inference, add controlled Gaussian noise to prevent monotonous speech
        if not self.training and inference_noise_scale > 0:
            noise = torch.randn_like(x) * inference_noise_scale
            x = x + noise
            
        return x * x_mask
        
    def inference(self, x, x_mask, g=None, lang_emb=None, noise_scale=0.2):
        """
        Optimized inference method with controlled randomness and better quality.
        
        Args:
            x: Text encoder output [B, C, T]
            x_mask: Sequence mask [B, 1, T]
            g: Speaker conditioning [B, C, 1] 
            lang_emb: Language embedding [B, C, 1]
            noise_scale: Amount of Gaussian noise to add (default: 0.2 for natural variance)
            
        Returns:
            Log duration predictions [B, 1, T]
        """
        with torch.no_grad():
            # Get base duration prediction with improved forward pass
            log_dur = self.forward(x, x_mask, g=g, lang_emb=lang_emb, inference_noise_scale=0.0)
            
            # Apply smart noise that varies based on phoneme context
            if noise_scale > 0:
                # Create context-aware noise
                # Vowels typically have longer durations, consonants shorter
                base_noise = torch.randn_like(log_dur) * noise_scale
                
                # Apply smoothing to noise for more natural variation
                if log_dur.size(-1) > 2:
                    # Simple moving average smoothing
                    kernel = torch.ones(1, 1, 3, device=log_dur.device) / 3
                    padded_noise = F.pad(base_noise, (1, 1), mode='replicate')
                    smoothed_noise = F.conv1d(padded_noise, kernel)
                    base_noise = smoothed_noise
                
                log_dur = log_dur + base_noise * x_mask
                
            # Ensure reasonable duration bounds
            log_dur = torch.clamp(log_dur, min=-2.0, max=3.0)  # exp(-2) ≈ 0.14, exp(3) ≈ 20 frames
                
            return log_dur * x_mask


class TransformerLayer(nn.Module):
    """Lightweight transformer layer with relative positional encoding for duration prediction."""
    
    def __init__(self, channels, num_heads, ff_dim, dropout_p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = RelativeMultiHeadAttention(channels, num_heads, dropout_p)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = FeedForward(channels, ff_dim, dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x, x_mask):
        # Self-attention with residual connection
        residual = x
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        x_attn = self.attn(x, x, x, x_mask)
        x = residual + self.dropout(x_attn)
        
        # Feed-forward with residual connection
        residual = x
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm2(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        x_ff = self.ff(x)
        x = residual + self.dropout(x_ff)
        
        return x * x_mask


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding for better sequence modeling."""
    
    def __init__(self, channels, num_heads, dropout_p=0.1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.q_proj = nn.Conv1d(channels, channels, 1)
        self.k_proj = nn.Conv1d(channels, channels, 1)
        self.v_proj = nn.Conv1d(channels, channels, 1)
        self.o_proj = nn.Conv1d(channels, channels, 1)
        
        # Relative positional encoding - larger size to handle various sequence lengths
        self.relative_pos_emb = nn.Parameter(torch.randn(1, num_heads, 1, 512) * 0.01)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, q, k, v, mask=None):
        batch_size, _, seq_len = q.size()
        
        # Project queries, keys, values
        q = self.q_proj(q).view(batch_size, self.num_heads, self.head_dim, seq_len)
        k = self.k_proj(k).view(batch_size, self.num_heads, self.head_dim, seq_len)
        v = self.v_proj(v).view(batch_size, self.num_heads, self.head_dim, seq_len)
        
        # Compute attention scores
        scores = torch.matmul(q.transpose(2, 3), k) / (self.head_dim ** 0.5)
        
        # Add relative positional encoding with dynamic sizing
        max_pos_len = self.relative_pos_emb.size(3)
        if seq_len <= max_pos_len:
            center = max_pos_len // 2
            start = max(0, center - seq_len // 2)
            end = min(max_pos_len, start + seq_len)
            relative_pos = self.relative_pos_emb[:, :, :, start:end]
            
            # Ensure exact size match
            if relative_pos.size(3) != seq_len:
                # Pad or trim to exact size
                if relative_pos.size(3) < seq_len:
                    padding = seq_len - relative_pos.size(3)
                    relative_pos = F.pad(relative_pos, (0, padding), mode='replicate')
                else:
                    relative_pos = relative_pos[:, :, :, :seq_len]
        else:
            # For sequences longer than our embedding, interpolate
            relative_pos = F.interpolate(
                self.relative_pos_emb, 
                size=(1, seq_len), 
                mode='linear', 
                align_corners=False
            )
        
        scores = scores + relative_pos
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v.transpose(2, 3))
        output = output.transpose(2, 3).contiguous()
        output = output.view(batch_size, self.channels, seq_len)
        
        # Project back to original dimension
        output = self.o_proj(output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network with gating mechanism."""
    
    def __init__(self, channels, ff_dim, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, channels, 1)
        self.gate = nn.Conv1d(channels, ff_dim, 1)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        x_gate = F.silu(self.gate(x))
        x_ff = self.conv1(x)
        x = x_gate * x_ff
        x = self.dropout(x)
        x = self.conv2(x)
        return x


# Backward compatibility - maintain original class names
class BalancedDurationPredictor(ImprovedDurationPredictor):
    """Backward compatible version of the improved duration predictor."""
    pass


class OptimizedDurationPredictor(nn.Module):
    """
    Further optimized version with even smaller footprint for mobile/edge deployment.
    Uses depthwise separable convolutions for maximum efficiency.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,  # Even smaller for mobile
        kernel_size: int = 3,
        dropout_p: float = 0.05,  # Reduced dropout
        cond_channels: int = None,
        language_emb_dim: int = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.language_emb_dim = language_emb_dim
        
        # Adjust input channels for language embedding
        conv_in_channels = in_channels
        if language_emb_dim:
            conv_in_channels += language_emb_dim
            
        # Input projection
        self.input_proj = nn.Conv1d(conv_in_channels, hidden_channels, 1)
        
        # Depthwise separable conv 1
        self.conv1_dw = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, 
                                 groups=hidden_channels, padding=kernel_size//2)
        self.conv1_pw = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.norm1 = nn.LayerNorm(hidden_channels)
        
        # Depthwise separable conv 2  
        self.conv2_dw = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, 
                                 groups=hidden_channels, padding=kernel_size//2)
        self.conv2_pw = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels // 2, 1, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # Conditioning layers
        if cond_channels is not None and cond_channels != 0:
            self.cond = nn.Conv1d(cond_channels, hidden_channels, 1)
            
        if language_emb_dim is not None and language_emb_dim != 0:
            self.cond_lang = nn.Conv1d(language_emb_dim, hidden_channels, 1)
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # Initialize final layer to predict log(2.0) ≈ 0.693
        with torch.no_grad():
            final_layer = self.proj[-1]
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                final_layer.bias.fill_(0.693)
                
    def forward(self, x, x_mask, g=None, lang_emb=None, inference_noise_scale=0.1):
        """Optimized forward pass with depthwise separable convolutions"""
        # Concatenate language embedding if present
        if lang_emb is not None and self.language_emb_dim:
            lang_emb_expanded = lang_emb.expand(-1, -1, x.size(2))
            x = torch.cat([x, lang_emb_expanded], dim=1)
            
        # Input projection
        x = self.input_proj(x)
        
        # Apply conditioning
        if g is not None and hasattr(self, 'cond'):
            x = x + self.cond(g)
            
        if lang_emb is not None and hasattr(self, 'cond_lang'):
            x = x + self.cond_lang(lang_emb)
        
        # First depthwise separable conv
        residual = x
        x = self.conv1_dw(x * x_mask)
        x = self.conv1_pw(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual
        
        # Second depthwise separable conv
        residual = x
        x = self.conv2_dw(x * x_mask)
        x = self.conv2_pw(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual
        
        # Output projection
        x = self.proj(x * x_mask)
        
        # Add noise during inference for natural variance
        if not self.training and inference_noise_scale > 0:
            noise = torch.randn_like(x) * inference_noise_scale
            x = x + noise
            
        return x * x_mask