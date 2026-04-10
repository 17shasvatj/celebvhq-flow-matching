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
        self.silu1 = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.silu2 = nn.SiLU()
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.max_period = 10000

    def forward(self, t):
        # t: (B,) float tensor of timesteps in [0, 1]
        # return: (B, hidden_dim)
        half_dim = self.sinusoidal_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = torch.outer(t, freqs) # (B, half_dim)
        sincos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.linear3(self.silu2(self.linear2(self.silu1(self.linear1(sincos)))))



class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.mlp_dropout = nn.Dropout(0.1)

        # Attention
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim*3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        #self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)


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


        h = self.norm1(x) * (1+scale1) + shift1

        # Attention
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)
        h = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # uses Flash Attention
        h = h.transpose(1, 2).reshape(B, N, D)
        h = self.out_proj(h)

        x = x + gate1 * self.attn_dropout(h)

        # MLP
        h = self.norm2(x) * (1+scale2) + shift2
        h = self.linear2(self.silu(self.linear1(h)))
        x = x + gate2*self.mlp_dropout(h)

        return x



class FaceDiT(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=1024, num_heads=16,
                 depth=20, mlp_ratio=4, patch_size=(2,2,2),
                 num_frames=16, latent_h=32, latent_w=32, clip_dim = 512, num_emotions = 8, cond_dropout=0.1,
                 clip_embeddings_path=None):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.in_channels = in_channels
        self.cond_dropout = cond_dropout
        self.num_emotions = num_emotions

        self.patch_embed = PatchEmbed(in_channels, hidden_dim, patch_size, num_frames, latent_h, latent_w)
        self.timestep_embedding = TimestepEmbedding(hidden_dim // 4, hidden_dim)

        # CLIP based emotion conditioning
        clip_embs = torch.load(clip_embeddings_path, weights_only=True)
        self.register_buffer("clip_embeddings", clip_embs)
        # Learnable projection from CLIP space to hidden_dim
        self.emotion_proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dit_blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads, mlp_ratio) for i in range(depth)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, patch_size[0]*patch_size[1]*patch_size[2]*in_channels)

    def forward(self, x, t, emotion=None):
        # x: (B, T, C, H, W) noisy latent video
        # t: (B,) timesteps
        # return: predicted velocity, same shape as x
        patch_embedded = self.patch_embed(x) # (B, num_tokens, hidden_dim)
        c = self.timestep_embedding(t)

        if emotion is not None:
            clip_emb = self.clip_embeddings[emotion]
            emotion_emb = self.emotion_proj(clip_emb)

            # CFG dropout
            if self.training and self.cond_dropout > 0:
                mask = torch.rand(emotion.shape[0], 1, device=emotion.device) > self.cond_dropout
                emotion_emb = emotion_emb * mask.float()

            c = c + emotion_emb
        h = patch_embedded
        for block in self.dit_blocks:
            h = block(h, c)
        h = self.layer_norm(h)
        h = self.linear(h) #(B, num_tokens, patch_t*patch_h*patch_w*in_ch)

        h = h.reshape(h.shape[0], self.num_frames // self.patch_size[0], self.latent_h // self.patch_size[1],
                      self.latent_w // self.patch_size[2], self.patch_size[0], self.patch_size[1],
                      self.patch_size[2], self.in_channels)
        h = h.permute(0, 1, 4, 7, 2, 5, 3, 6)
        h = h.reshape(h.shape[0], self.num_frames, self.in_channels, self.latent_h, self.latent_w)
        return h

