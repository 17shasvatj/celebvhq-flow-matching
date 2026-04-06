import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=1024,
                 patch_size=(2, 2, 2), num_frames=16, latent_h=32, latent_w=32):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels,
                              hidden_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.num_tokens = (num_frames // patch_size[0]) * (latent_h // patch_size[1]) * (latent_w //patch_size[2])
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))


    def forward(self, x):
        # x: (B, T, C, H, W) = (B, 16, 4, 32, 32)
        # return: (B, num_tokens, hidden_dim)
        x = self.proj(x.permute(0, 2, 1, 3, 4)) # (B, 1024, 8, 16, 16)
        x = x.flatten(2)  # (B, 1024, 2048)
        x = x.permute(0, 2, 1) # (B, 2048, 1024)
        return x + self.pos_emb



class TimestepEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim, hidden_dim=1024):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(sinusoidal_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.max_period = 10000

    def forward(self, t):
        # t: (B,) float tensor of timesteps in [0, 1]
        # return: (B, hidden_dim)
        half_dim = self.sinusoidal_dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half_dim, device=t.device, dtype=torch.float32))/half_dim
        args = torch.outer(t, freqs) # (B, half_dim)
        sincos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.linear2(self.silu(self.linear1(sincos)))



class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, mlp_ratio*hidden_dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(mlp_ratio*hidden_dim, hidden_dim)
        self.adaln_linear = nn.Linear(hidden_dim, hidden_dim*6)
        nn.init.zeros_(self.adaln_linear.weight)
        nn.init.zeros_(self.adaln_linear.bias)

    def forward(self, x, c):
        # x: (B, num_tokens, hidden_dim)
        # c: (B, hidden_dim) — timestep conditioning
        # return: (B, num_tokens, hidden_dim)
        mod_params = self.adaln_linear(c) # (B, hidden_dim*6)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod_params.unsqueeze(1).chunk(6, dim=-1)

        # Attention
        h = self.norm1(x) * (1+scale1) + shift1
        h, _ = self.attention(h, h, h)
        x = x + gate1 * h

        # MLP
        h = self.norm2(x) * (1+scale2) + shift2
        h = self.linear2(self.silu(self.linear1(h)))
        x = x + gate2*h

        return x



class FaceDiT(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=1024, num_heads=16,
                 depth=20, mlp_ratio=4, patch_size=(2,2,2),
                 num_frames=16, latent_h=32, latent_w=32):
        super().__init__()
        # TODO: PatchEmbed
        # TODO: TimestepEmbedding
        # TODO: nn.ModuleList of DiTBlocks
        # TODO: final LayerNorm
        # TODO: linear projection hidden_dim -> patch_dim (unpatchify)

    def forward(self, x, t):
        # x: (B, T, C, H, W) noisy latent video
        # t: (B,) timesteps
        # TODO: patch embed
        # TODO: timestep embed
        # TODO: run through all DiT blocks
        # TODO: final norm + unpatchify back to (B, T, C, H, W)
        # return: predicted velocity, same shape as x
        pass