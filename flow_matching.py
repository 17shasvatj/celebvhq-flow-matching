import torch
import torch.nn.functional as F
def train_step(model, x1, optimizer):
    """
    x1: clean latent video batch, (B, T, C, H, W)
    """
    # return loss value
    x0 = torch.randn_like(x1)
    t = torch.rand((x1.shape[0], ), device=x1.device)
    t_expand = t[:, None, None, None, None]
    xt = (1-t_expand)*x0 + t_expand*x1
    v = x1 - x0
    v_pred = model(xt, t)
    loss = F.mse_loss(v_pred, v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def sample(model, shape, num_steps=50, device="cuda"):
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
        v = model(x, t)
        x = x + v*dt
    return x
