import math

import torch
from torch import nn

from TTS.tts.layers.glow_tts.glow import WN
from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer
from TTS.tts.utils.helpers import sequence_mask

LRELU_SLOPE = 0.1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise Separable Convolution for efficient local feature extraction."""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            padding=padding, dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: [B, C, T]
        x = self.depthwise(x)
        x = self.pointwise(x)
        # Transpose for LayerNorm: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        # Back to [B, C, T]
        x = x.transpose(1, 2)
        return x


class CausalConvBlock(nn.Module):
    """Causal convolution block for fast local feature extraction."""
    
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Causal padding
        
        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size, padding=self.padding
        )
        self.conv2 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size, padding=self.padding
        )
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, x_mask):
        # x: [B, C, T], x_mask: [B, 1, T]
        residual = x
        
        # First conv with causal masking
        x = self.conv1(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]  # Remove future frames
        x = x * x_mask
        x = self.dropout(x)
        
        # Second conv with causal masking
        x = self.conv2(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]  # Remove future frames
        x = x * x_mask
        x = self.dropout(x)
        
        # Residual connection
        x = x + residual
        return x


class LimitedContextTransformer(nn.Module):
    """Single-layer transformer with limited context window for efficiency."""
    
    def __init__(self, hidden_channels, num_heads=4, context_window=32, dropout_p=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.context_window = context_window
        
        self.attention = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout_p, batch_first=False
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Dropout(dropout_p)
        )
        
    def create_causal_mask(self, seq_len, device):
        """Create causal attention mask with limited context window."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        for i in range(seq_len):
            # Allow attention to previous tokens within context window
            start_idx = max(0, i - self.context_window + 1)
            mask[i, start_idx:i+1] = 0.0
            
        return mask
        
    def forward(self, x, x_mask):
        # x: [B, C, T], x_mask: [B, 1, T]
        B, C, T = x.shape
        
        # Transpose to [T, B, C] for MultiheadAttention
        x = x.transpose(0, 2).transpose(1, 2)  # [T, B, C]
        
        # Create attention mask
        attn_mask = self.create_causal_mask(T, x.device)
        
        # Create key padding mask from x_mask
        key_padding_mask = (x_mask.squeeze(1) == 0)  # [B, T]
        
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(
            x, x, x, 
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = residual + attn_out
        
        # FFN with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        # Transpose back to [B, C, T]
        x = x.transpose(0, 1).transpose(1, 2)
        
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
    ):
        """Optimized Hybrid Text Encoder for VITS model.
        
        Uses causal depthwise-separable convolutions for fast local feature extraction
        followed by a single-layer transformer with limited context for semantic understanding.

        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers (unused in this implementation).
            num_heads (int): Number of attention heads for the Transformer layer.
            num_layers (int): Number of layers (unused, we use fixed architecture).
            kernel_size (int): Kernel size for the causal convolution layers.
            dropout_p (float): Dropout rate for the layers.
            language_emb_dim (int, optional): Language embedding dimension.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Character embedding
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        
        # Adjust hidden channels if language embedding is used
        encoder_channels = hidden_channels
        if language_emb_dim:
            encoder_channels += language_emb_dim
            
        # Project to consistent channel size if needed
        self.input_proj = None
        if encoder_channels != hidden_channels:
            self.input_proj = nn.Conv1d(encoder_channels, hidden_channels, 1)
        
        # Fast local feature extraction with causal convolutions
        self.conv_layers = nn.ModuleList([
            CausalConvBlock(hidden_channels, kernel_size=kernel_size),
            CausalConvBlock(hidden_channels, kernel_size=kernel_size)
        ])
        
        # Limited context transformer for semantic understanding
        self.transformer = LimitedContextTransformer(
            hidden_channels=hidden_channels,
            num_heads=min(num_heads, 4),  # Cap at 4 heads for efficiency
            context_window=32,  # Limited context window
            dropout_p=dropout_p
        )
        
        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_lengths: :math:`[B]`
            - lang_emb: :math:`[B, lang_emb_dim, 1]`
        """
        assert x.shape[0] == x_lengths.shape[0]
        
        # Character embedding
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, H]
        
        # Concatenate language embedding if provided
        if lang_emb is not None:
            lang_emb_expanded = lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)
            x = torch.cat((x, lang_emb_expanded), dim=-1)
        
        # Transpose to [B, C, T] for convolutions
        x = x.transpose(1, 2)  # [B, H, T]
        
        # Create mask
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype).to(x.device)  # [B, 1, T]
        
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Apply mask
        x = x * x_mask
        
        # Fast local feature extraction with causal convolutions
        for conv_layer in self.conv_layers:
            x = conv_layer(x, x_mask)
        
        # Limited context transformer for semantic understanding
        x = self.transformer(x, x_mask)
        
        # Apply mask again
        x = x * x_mask
        
        # Output projection
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.

        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class MiniWaveNet(nn.Module):
    """Extremely small WaveNet with only 2 layers for efficient flow transformation."""
    
    def __init__(self, hidden_channels, kernel_size=3, cond_channels=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.cond_channels = cond_channels
        
        # Only 2 layers with minimal channels (64)
        self.residual_channels = 64
        
        # Input projection
        self.start = nn.Conv1d(hidden_channels, self.residual_channels, 1)
        
        # Two dilated convolution layers
        self.conv1 = nn.Conv1d(
            self.residual_channels, 
            self.residual_channels * 2, 
            kernel_size, 
            dilation=1, 
            padding=get_padding(kernel_size, 1)
        )
        
        self.conv2 = nn.Conv1d(
            self.residual_channels, 
            self.residual_channels * 2, 
            kernel_size, 
            dilation=2, 
            padding=get_padding(kernel_size, 2)
        )
        
        # Conditioning projections if needed
        if cond_channels > 0:
            self.cond_proj1 = nn.Conv1d(cond_channels, self.residual_channels * 2, 1)
            self.cond_proj2 = nn.Conv1d(cond_channels, self.residual_channels * 2, 1)
        
        # Residual and skip connections
        self.res_proj1 = nn.Conv1d(self.residual_channels, self.residual_channels, 1)
        self.res_proj2 = nn.Conv1d(self.residual_channels, self.residual_channels, 1)
        
        # Output projection
        self.end = nn.Conv1d(self.residual_channels, hidden_channels, 1)
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, x, x_mask, g=None):
        """
        Args:
            x: [B, C, T] input tensor
            x_mask: [B, 1, T] mask tensor
            g: [B, C_cond, 1] conditioning tensor
        """
        x = self.start(x)
        residual = x
        
        # Layer 1
        h = self.conv1(x)
        if g is not None and self.cond_channels > 0:
            h = h + self.cond_proj1(g)
        
        # Gated activation
        h_tanh, h_sigmoid = torch.split(h, self.residual_channels, dim=1)
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)
        
        # Residual connection
        x = self.res_proj1(h) + residual
        x = x * x_mask
        
        # Layer 2
        residual = x
        h = self.conv2(x)
        if g is not None and self.cond_channels > 0:
            h = h + self.cond_proj2(g)
        
        # Gated activation
        h_tanh, h_sigmoid = torch.split(h, self.residual_channels, dim=1)
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)
        
        # Residual connection
        x = self.res_proj2(h) + residual
        x = x * x_mask
        
        # Output projection
        x = self.end(x)
        return x


class OptimizedAffineCouplingLayer(nn.Module):
    """Single Glow-like Affine Coupling Layer with extremely small WaveNet."""
    
    def __init__(self, channels, hidden_channels, cond_channels=0):
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"
        
        self.half_channels = channels // 2
        
        # Extremely small WaveNet: 2 layers, 64 residual channels, kernel size 3
        self.transform_net = MiniWaveNet(
            hidden_channels=hidden_channels,
            kernel_size=3,
            cond_channels=cond_channels
        )
        
        # Input projection
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        
        # Output projection for scale and translation
        self.post = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)
        
        # Initialize output layer to zero for training stability
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)
        
    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
            x: [B, C, T] input tensor
            x_mask: [B, 1, T] mask tensor  
            g: [B, C_cond, 1] conditioning tensor
            reverse: bool, set True for inference
        """
        # Split input into two halves
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        
        # Transform first half through network
        h = self.pre(x0) * x_mask
        h = self.transform_net(h, x_mask, g=g)
        
        # Get scale and translation parameters
        params = self.post(h) * x_mask
        scale, translation = torch.split(params, [self.half_channels] * 2, dim=1)
        
        # Apply affine transformation
        if not reverse:
            # Forward: x1 = x1 * exp(scale) + translation
            x1 = x1 * torch.exp(scale) + translation
            x1 = x1 * x_mask
            logdet = torch.sum(scale * x_mask, [1, 2])
        else:
            # Reverse: x1 = (x1 - translation) * exp(-scale)
            x1 = (x1 - translation) * torch.exp(-scale)
            x1 = x1 * x_mask
            logdet = -torch.sum(scale * x_mask, [1, 2])
        
        # Concatenate halves
        x = torch.cat([x0, x1], dim=1)
        
        if not reverse:
            return x, logdet
        else:
            return x


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
    ):
        """Optimized flow with single Glow-like Affine Coupling Layer.
        
        Replaces multiple ResidualCouplingBlocks with a single extremely efficient
        affine coupling layer using a 2-layer WaveNet with 64 residual channels.
        
        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels (reduced to 64 internally).
            kernel_size (int): Kernel size (fixed to 3 for efficiency).
            dilation_rate (int): Dilation rate (unused in optimized version).
            num_layers (int): Number of layers (unused, fixed to 2).
            num_flows (int, optional): Number of flows (unused, fixed to 1).
            cond_channels (int, optional): Number of conditioning channels.
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = min(hidden_channels, 64)  # Cap at 64 for efficiency
        self.cond_channels = cond_channels
        
        # Single optimized affine coupling layer
        self.flow = OptimizedAffineCouplingLayer(
            channels=channels,
            hidden_channels=self.hidden_channels,
            cond_channels=cond_channels
        )
        
    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
            x: [B, C, T] input tensor
            x_mask: [B, 1, T] mask tensor
            g: [B, C_cond, 1] conditioning tensor
            reverse: bool, set True for inference
        """
        if not reverse:
            # Forward pass
            x, logdet = self.flow(x, x_mask, g=g, reverse=False)
            # Apply channel flip for better mixing
            x = torch.flip(x, [1])
        else:
            # Reverse pass
            x = torch.flip(x, [1])
            x = self.flow(x, x_mask, g=g, reverse=True)
            
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask
