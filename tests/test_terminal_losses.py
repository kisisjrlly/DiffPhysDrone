import torch

from losses import compute_physics_losses


def test_physics_losses_accept_terminal_mask_and_post_action_clearance():
    # T x B histories. The first action is alive; later samples are terminal
    # padding and must not contribute to navigation or smoothness losses.
    v = torch.zeros(3, 1, 3)
    tv = torch.zeros_like(v)
    act = torch.zeros_like(v)
    vec = torch.tensor([[[2.0, 0.0, 0.0]], [[0.0001, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    p = torch.zeros_like(vec)
    margin = torch.ones(1)
    valid = torch.tensor([[True], [False], [False]])
    losses = compute_physics_losses(
        v, tv, act, vec, p, margin, torch.zeros(1, 3), valid_mask=valid, win=1
    )
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["loss_collide"].item() > 0.0
