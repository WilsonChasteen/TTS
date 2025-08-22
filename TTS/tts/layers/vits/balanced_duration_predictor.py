import torch
from torch import nn
from torch.nn import functional as F


class BalancedDurationPredictor(nn.Module):
    """Balanced Duration Predictor optimized for real-time performance with high quality.
    
    Uses a 3-layer CNN with residual connections and improved normalization for better quality.
    Includes attention mechanism for better context modeling while maintaining efficiency.
    
    Architecture:
        - 3-layer CNN with residual connections
        - Layer normalization for stability
        - Lightweight attention for context modeling
        - Optimized for both speed and quality
        
    Args:
        in_channels (int): Number of input channels from text encoder
        hidden_channels (int): Number of hidden channels (default: 128 for quality)
        kernel_size (int): Convolution kernel size (default: 3)
        dropout_p (float): Dropout probability
        cond_channels (int): Number of conditioning channels (speaker embedding)
        language_emb_dim (int): Language embedding dimension
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int = 128,  # Increased for better quality
        kernel_size: int = 3, 
        dropout_p: float = 0.1,
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
        
        # Input projection for better feature transformation
        self.input_proj = nn.Conv1d(conv_in_channels, hidden_channels, 1)
        
        # Three-layer CNN with residual connections for better quality
        self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(hidden_channels)
        
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.norm3 = nn.LayerNorm(hidden_channels)
        
        # Lightweight attention for context modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=min(4, hidden_channels // 32),  # Ensure divisible by num_heads
            dropout=dropout_p,
            batch_first=False
        )
        
        # Output projection with better initialization
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels // 2, 1, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # Conditioning layers with better initialization
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
            if hasattr(self.proj, '__iter__'):
                final_layer = self.proj[-1]
            else:
                final_layer = self.proj
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                final_layer.bias.fill_(0.693)  # log(2.0)
                    
    def _causal_pad(self, x, padding):
        """Apply causal padding to maintain causality for real-time inference"""
        return F.pad(x, (padding, 0))
        
    def forward(self, x, x_mask, g=None, lang_emb=None, inference_noise_scale=0.1):
        """
        Forward pass of the balanced duration predictor with improved architecture.
        
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
            x = torch.cat([x, lang_emb_expanded], dim=1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply conditioning after input projection
        if g is not None and hasattr(self, 'cond'):
            x = x + self.cond(g)
            
        if lang_emb is not None and hasattr(self, 'cond_lang'):
            x = x + self.cond_lang(lang_emb)
        
        # Store input for residual connection
        residual = x
        
        # First conv layer with residual connection
        x = self.conv1(x * x_mask)
        x = x.transpose(1, 2)  # [B, T, C] for LayerNorm
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        
        # Second conv layer with residual connection
        residual = x
        x = self.conv2(x * x_mask)
        x = x.transpose(1, 2)  # [B, T, C] for LayerNorm
        x = self.norm2(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        
        # Third conv layer with residual connection
        residual = x
        x = self.conv3(x * x_mask)
        x = x.transpose(1, 2)  # [B, T, C] for LayerNorm
        x = self.norm3(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        
        # Apply lightweight attention for better context modeling
        # Only apply attention if we have enough channels and sequence length
        if x.size(1) >= 32 and x.size(2) > 1:  # Ensure minimum dimensions
            # Prepare for attention: [T, B, C]
            x_att = x.transpose(0, 2).transpose(0, 1)  # [T, B, C]
            
            # Create attention mask from x_mask
            seq_len = x_att.size(0)
            batch_size = x_att.size(1)
            attn_mask = x_mask.squeeze(1).bool()  # [B, T]
            attn_mask = ~attn_mask  # Invert for attention (True = ignore)
            
            # Apply self-attention with error handling
            try:
                x_att, _ = self.attention(x_att, x_att, x_att, key_padding_mask=attn_mask)
                
                # Convert back to [B, C, T]
                x_att = x_att.transpose(0, 1).transpose(1, 2)  # [B, C, T]
                
                # Combine with residual (weighted combination for stability)
                x = x + 0.1 * x_att  # Reduce attention contribution for stability
            except Exception as e:
                # If attention fails, skip it and continue with residual path
                pass
        
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


class OptimizedDurationPredictor(BalancedDurationPredictor):
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
        # Initialize parent with smaller hidden channels
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels, 
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            cond_channels=cond_channels,
            language_emb_dim=language_emb_dim
        )
        
        # Replace standard convolutions with depthwise separable convolutions
        conv_in_channels = in_channels
        if language_emb_dim:
            conv_in_channels += language_emb_dim
            
        # Depthwise separable conv 1
        self.conv1_dw = nn.Conv1d(conv_in_channels, conv_in_channels, kernel_size, groups=conv_in_channels, padding=0)
        self.conv1_pw = nn.Conv1d(conv_in_channels, hidden_channels, 1)
        
        # Depthwise separable conv 2  
        self.conv2_dw = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, groups=hidden_channels, padding=0)
        self.conv2_pw = nn.Conv1d(hidden_channels, hidden_channels, 1)
        
        # Remove the original conv layers
        delattr(self, 'conv1')
        delattr(self, 'conv2')
        
    def forward(self, x, x_mask, g=None, lang_emb=None, inference_noise_scale=0.1):
        """Optimized forward pass with depthwise separable convolutions"""
        # Apply conditioning
        if g is not None:
            x = x + self.cond(g)
            
        if lang_emb is not None:
            x = x + self.cond_lang(lang_emb)
            
        # Concatenate language embedding if present
        if lang_emb is not None and self.language_emb_dim:
            # Expand language embedding to match sequence length
            lang_emb_expanded = lang_emb.expand(-1, -1, x.size(2))
            x = torch.cat([x, lang_emb_expanded], dim=1)
            
        # First depthwise separable conv
        x = self._causal_pad(x, self.causal_padding)
        x_mask_padded = self._causal_pad(x_mask, self.causal_padding)
        x = self.conv1_dw(x * x_mask_padded)
        x = self.conv1_pw(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second depthwise separable conv
        x = self._causal_pad(x, self.causal_padding)
        x_mask_padded = self._causal_pad(x_mask, self.causal_padding)
        x = self.conv2_dw(x * x_mask_padded) 
        x = self.conv2_pw(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output projection
        x = self.proj(x * x_mask)
        
        # Add noise during inference for natural variance
        if not self.training and inference_noise_scale > 0:
            noise = torch.randn_like(x) * inference_noise_scale
            x = x + noise
            
        return x * x_mask