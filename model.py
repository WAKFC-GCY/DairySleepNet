"""DairySleepNet: channel-wise and fusion encoders with selective state spaces."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (B, L, D)
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Core of Mamba

    This is a pure PyTorch implementation that doesn't require CUDA kernels.
    For production use with long sequences, consider using the official mamba-ssm package.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection: project to 2*d_inner (for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM parameters projection
        # Projects input to dt, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # dt_rank=1 for simplicity

        # dt projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize dt bias to be in a reasonable range
        dt_init_std = 1.0 / math.sqrt(self.d_inner)
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)

        # A parameter (state matrix) - initialized as negative values (decay)
        # Using real-valued diagonal SSM
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            y: (B, L, D) output tensor
        """
        B, L, D = x.shape

        # Input projection and split for gating
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # Causal: remove extra padding
        x = rearrange(x, 'b d l -> b l d')

        # Activation
        x = F.silu(x)

        # SSM
        y = self.ssm(x)

        # Gating
        z = F.silu(z)
        y = y * z

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y

    def ssm(self, x):
        """
        Selective State Space Model computation

        Args:
            x: (B, L, d_inner) - after conv and activation
        Returns:
            y: (B, L, d_inner)
        """
        B, L, D = x.shape
        N = self.d_state

        # Get A (decay rates)
        A = -torch.exp(self.A_log)  # (N,)

        # Project x to get dt, B, C
        x_dbl = self.x_proj(x)  # (B, L, 2*N + 1)

        # Split into dt, B, C
        dt, B_proj, C = torch.split(x_dbl, [1, N, N], dim=-1)

        # dt projection and softplus
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)  # Ensure positive

        # Discretize A and B
        # A_bar = exp(dt * A)
        # For diagonal A: A_bar[i] = exp(dt * A[i])
        dA = torch.einsum('b l d, n -> b l d n', dt, A)  # (B, L, D, N)
        A_bar = torch.exp(dA)

        # B_bar = dt * B (simplified ZOH for diagonal case)
        dB = torch.einsum('b l d, b l n -> b l d n', dt, B_proj)  # (B, L, D, N)

        # Sequential scan (can be parallelized with associative scan for efficiency)
        y = self.selective_scan(x, A_bar, dB, C)

        # Skip connection with D
        y = y + x * self.D

        return y

    def selective_scan(self, x, A_bar, dB, C):
        """
        Sequential scan implementation

        For short sequences this is acceptable. For long sequences,
        use parallel scan or the official CUDA implementation.

        Args:
            x: (B, L, D) input
            A_bar: (B, L, D, N) discretized state matrix
            dB: (B, L, D, N) discretized input matrix * dt
            C: (B, L, N) output matrix
        Returns:
            y: (B, L, D) output
        """
        B, L, D, N = A_bar.shape

        # Initialize state
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)

        ys = []
        for i in range(L):
            # h = A_bar * h + dB * x
            h = A_bar[:, i] * h + dB[:, i] * x[:, i, :, None]

            # y = C * h
            y_i = torch.einsum('b d n, b n -> b d', h, C[:, i])
            ys.append(y_i)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        return y


class MambaBlock(nn.Module):
    """
    A single Mamba block with residual connection and normalization
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, x):
        # Pre-norm residual
        return x + self.mamba(self.norm(x))


class MambaEncoder(nn.Module):
    """
    Stack of Mamba blocks with final RMS normalization
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DairySleepNet(nn.Module):
    """
    Mamba-based Sleep Classification Model

    Architecture:
    1. Single-channel feature extraction: Independent MambaEncoder for each channel
    2. Multi-channel fusion: MambaEncoder on concatenated features with residual
    3. Classification head: MLP

    Both encoding stages use selective state-space blocks.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        C = config.num_channels
        d = config.dim_model
        d_multi = d * C

        # Hyperparameters for Mamba
        d_state = getattr(config, 'd_state', 16)
        d_conv = getattr(config, 'd_conv', 4)
        expand = getattr(config, 'expand', 2)

        # Single-channel Mamba encoders
        self.single_encoders = nn.ModuleList([
            MambaEncoder(
                d_model=d,
                num_layers=config.num_encoder,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=config.dropout,
            )
            for _ in range(C)
        ])

        # Multi-channel fusion Mamba encoder
        self.multi_encoder = MambaEncoder(
            d_model=d_multi,
            num_layers=config.num_encoder_multi,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=config.dropout,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = RMSNorm(d_multi)

        # Classification head
        self.fc1 = nn.Linear(config.pad_size * d_multi, 1024)
        self.relu = nn.GELU()  # GELU often works better with Mamba
        self.fc2 = nn.Linear(1024, config.num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, F) = (B, num_channels, time_steps, features)
               e.g., (B, 9, 29, 128) for PSG
        Returns:
            logits: (B, num_classes)
        """
        B, C, T, F = x.shape
        if C != self.config.num_channels:
            raise ValueError(f"Expected C={self.config.num_channels}, got {C}")

        # Single-channel encoding
        outs = []
        for i in range(C):
            xi = x[:, i, :, :]  # (B, T, F)
            xi = self.single_encoders[i](xi)  # (B, T, F)
            outs.append(xi)

        # Concatenate channel features
        x_cat = torch.cat(outs, dim=2)  # (B, T, F*C)
        residual = self.layer_norm(self.dropout(x_cat))

        # Multi-channel fusion
        x_multi = self.multi_encoder(residual)  # (B, T, F*C)
        x_multi = self.layer_norm(x_multi + residual)  # Residual connection

        # Flatten and classify
        x_flat = x_multi.reshape(B, -1)  # (B, T * F * C)
        x = self.fc2(self.dropout(self.relu(self.fc1(x_flat))))

        return x



# Keep the research class name available for existing integrations.
MambaSleepClassifier = DairySleepNet
