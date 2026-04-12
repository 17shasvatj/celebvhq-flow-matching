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
        self.num_tokens = (num_frames // patch_size[0]) * (latent_h // patch_size[1]) * (latent_w // patch_size[2])
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))

    def forward(self, x):
        x = self.proj(x.permute(0, 2, 1, 3, 4))
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        return x + self.pos_emb


class TimestepEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim, hidden_dim=1024):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(sinusoidal_dim, hidden_dim)
        self.silu1 = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.silu2 = nn.SiLU()
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.max_period = 10000

    def forward(self, t):
        half_dim = self.sinusoidal_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = torch.outer(t, freqs)
        sincos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.linear3(self.silu2(self.linear2(self.silu1(self.linear1(sincos)))))


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Handles label dropout for classifier-free guidance.
    Exactly matches DiT's LabelEmbedder implementation.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        """
        Drops labels to enable classifier-free guidance.
        Replaces dropped labels with num_classes (the null class index).
        """
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train):
        """
        Args:
            labels: (B,) integer class labels, 0 to num_classes-1
            train: bool, whether in training mode
        Returns:
            (B, hidden_size) embedding vectors
        """
        use_dropout = self.dropout_prob > 0
        if train and use_dropout:
            labels = self.token_drop(labels)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.mlp_dropout = nn.Dropout(0.1)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, mlp_ratio * hidden_dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(mlp_ratio * hidden_dim, hidden_dim)
        self.adaln_linear = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.zeros_(self.adaln_linear.weight)
        nn.init.zeros_(self.adaln_linear.bias)

    def forward(self, x, c):
        mod_params = self.adaln_linear(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod_params.unsqueeze(1).chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale1) + shift1

        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        h = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, D)
        h = self.out_proj(h)

        x = x + gate1 * self.attn_dropout(h)

        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.linear2(self.silu(self.linear1(h)))
        x = x + gate2 * self.mlp_dropout(h)

        return x


class FaceDiT(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=1024, num_heads=16,
                 depth=20, mlp_ratio=4, patch_size=(2, 2, 2),
                 num_frames=16, latent_h=32, latent_w=32,
                 num_classes=8, cond_dropout=0.15):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(in_channels, hidden_dim, patch_size, num_frames, latent_h, latent_w)
        self.timestep_embedding = TimestepEmbedding(hidden_dim // 4, hidden_dim)

        # Label embedder: exactly like DiT
        # num_classes + 1 entries: 0-7 = emotions, 8 = null (unconditional)
        self.label_embedder = LabelEmbedder(num_classes, hidden_dim, cond_dropout)

        self.dit_blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, patch_size[0] * patch_size[1] * patch_size[2] * in_channels)

    def forward(self, x, t, emotion):
        """
        x: (B, T, C, H, W) noisy latent video
        t: (B,) timesteps
        emotion: (B,) integer emotion labels (0-7)
        """
        patch_embedded = self.patch_embed(x)
        t_emb = self.timestep_embedding(t)

        # Label embedding with CFG dropout (replaces with null class during training)
        y_emb = self.label_embedder(emotion, self.training)

        # Combine timestep + label conditioning
        c = t_emb + y_emb

        h = patch_embedded
        for block in self.dit_blocks:
            h = block(h, c)
        h = self.layer_norm(h)
        h = self.linear(h)

        h = h.reshape(h.shape[0], self.num_frames // self.patch_size[0], self.latent_h // self.patch_size[1],
                      self.latent_w // self.patch_size[2], self.patch_size[0], self.patch_size[1],
                      self.patch_size[2], self.in_channels)
        h = h.permute(0, 1, 4, 7, 2, 5, 3, 6)
        h = h.reshape(h.shape[0], self.num_frames, self.in_channels, self.latent_h, self.latent_w)
        return h