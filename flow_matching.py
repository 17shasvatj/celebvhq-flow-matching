import torch
import torch.nn.functional as F
def train_step(model, x1, emotion, optimizer):
    """
    x1: clean latent video batch, (B, T, C, H, W)
    """
    # return loss value
    if torch.rand(1).item() > 0.5:
        x1 = x1.flip(-1)
    x0 = torch.randn_like(x1)
    t = torch.rand((x1.shape[0], ), device=x1.device)
    t_expand = t[:, None, None, None, None]
    xt = (1-t_expand)*x0 + t_expand*x1
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
def sample(model, shape, num_steps=50, device="cuda", emotion=None, cfg_scale=1.0):
    """
    Generate a video from pure noise.
    shape: (B, T, C, H, W)
    """
    # return x
    x = torch.randn(shape, device=device)
    dt = 1.0/num_steps
    steps = torch.linspace(0, 1-dt, num_steps)
    for step in steps:
        t = torch.full((shape[0],), step, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if cfg_scale > 1.0 and emotion is not None:
                v_cond = model(x, t, emotion=emotion) # conditional
                v_uncond = model(x, t, emotion=None) # unconditional
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, emotion)
        x = x + v*dt
    return x
