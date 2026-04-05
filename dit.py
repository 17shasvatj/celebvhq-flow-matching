import torch
import torch.nn as nn

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
    def __init__(self, hidden_dim=1024):
        super().__init__()
        # TODO: MLP that maps sinusoidal embedding -> hidden_dim
        # sinusoidal dim is typically hidden_dim // 4 or 256

    def forward(self, t):
        # t: (B,) float tensor of timesteps in [0, 1]
        # TODO: sinusoidal embed -> MLP
        # return: (B, hidden_dim)
        pass


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, mlp_ratio=4):
        super().__init__()
        # TODO: two LayerNorms
        # TODO: self-attention (nn.MultiheadAttention)
        # TODO: MLP (hidden_dim -> hidden_dim*mlp_ratio -> hidden_dim)
        # TODO: adaLN modulation: linear that maps conditioning -> 6 * hidden_dim
        #       (shift1, scale1, gate1, shift2, scale2, gate2)
        # NOTE: initialize the adaLN linear's weights to zero

    def forward(self, x, c):
        # x: (B, num_tokens, hidden_dim)
        # c: (B, hidden_dim) — timestep conditioning
        # TODO: get 6 modulation params from c
        # TODO: norm -> modulate -> attention -> gate -> residual
        # TODO: norm -> modulate -> MLP -> gate -> residual
        # return: (B, num_tokens, hidden_dim)
        pass


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