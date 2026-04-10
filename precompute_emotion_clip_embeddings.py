"""
precompute_clip_emotions.py — Pre-compute CLIP text embeddings for emotion conditioning.

Run on Colab (free) or any machine with internet access.
Saves a tensor of shape (8, 512) to clip_emotion_embeddings.pt

Usage:
    python precompute_clip_emotions.py
"""

import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# Descriptive prompts for each emotion — richer descriptions give better embeddings
EMOTION_PROMPTS = {
    0: "a photo of a person with a calm neutral expression, relaxed face",
    1: "a photo of a happy person smiling with joy, cheerful expression",
    2: "a photo of a sad person with a sorrowful expression, tearful eyes, frowning",
    3: "a photo of a surprised person with wide open eyes and open mouth, shocked expression",
    4: "a photo of a fearful person with a scared terrified expression, eyes wide with fear",
    5: "a photo of a person with a disgusted expression, nose wrinkled in disgust",
    6: "a photo of an angry person with a furious expression, furrowed brows, clenched jaw",
    7: "a photo of a person with a contemptuous expression, smirking with disdain",
}

# Compute embeddings
embeddings = []
with torch.no_grad():
    for idx in range(8):
        text = clip.tokenize([EMOTION_PROMPTS[idx]]).to(device)
        emb = model.encode_text(text)  # (1, 512)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
        embeddings.append(emb.cpu())

embeddings = torch.cat(embeddings, dim=0)  # (8, 512)
print(f"Embeddings shape: {embeddings.shape}")

# Verify separation
print("\nCosine similarity matrix:")
sim = embeddings @ embeddings.T
for i in range(8):
    row = [f"{sim[i,j]:.3f}" for j in range(8)]
    emotions = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]
    print(f"  {emotions[i]:>10}: {' '.join(row)}")

torch.save(embeddings, "clip_emotion_embeddings.pt")
print("\nSaved clip_emotion_embeddings.pt")