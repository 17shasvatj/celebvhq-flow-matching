import torch
import torch.nn.functional as F


def train_step(model, x1, emotion, optimizer):
    """
    x1: clean latent video batch, (B, T, C, H, W)
    emotion: (B,) int emotion labels (0-7)
    CFG dropout is handled inside model.forward() via LabelEmbedder.token_drop()
    """
    x0 = torch.randn_like(x1)
    t = torch.rand((x1.shape[0],), device=x1.device)
    t_expand = t[:, None, None, None, None]
    xt = (1 - t_expand) * x0 + t_expand * x1
    v = x1 - x0

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v_pred = model(xt, t, emotion=emotion)
        loss = F.mse_loss(v_pred, v)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def sample(model, shape, emotion, num_steps=50, cfg_scale=1.0, device="cuda"):
    """
    Generate a video using single-model CFG with null token.
    shape: (B, T, C, H, W)
    emotion: (B,) int emotion labels (0-7)
    cfg_scale: guidance scale. 1.0 = no guidance.
    """
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    steps = torch.linspace(0, 1 - dt, num_steps)
    B = shape[0]

    # Null token index = num_classes (8 for our model)
    # Handle torch.compile wrapper (_orig_mod) and DataParallel (module)
    if hasattr(model, '_orig_mod'):
        nc = model._orig_mod.num_classes
    elif hasattr(model, 'module'):
        nc = model.module.num_classes
    else:
        nc = model.num_classes
    null_labels = torch.full((B,), nc, device=device, dtype=torch.long)

    use_cfg = cfg_scale > 1.0

    for step in steps:
        t = torch.full((B,), step, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if use_cfg:
                # Both go through the SAME model, SAME pathway
                v_cond = model(x, t, emotion=emotion)
                v_uncond = model(x, t, emotion=null_labels)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, emotion=emotion)
        x = x + v * dt
    return x