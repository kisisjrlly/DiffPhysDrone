import torch
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4, use_diff_cam=False,
                 use_unified_control=False, use_cam_obs=False) -> None:
        """
        Args:
            dim_obs: base observation dimension (7 or 10).
            dim_action: flight action dimension (default 6 = 3 accel + 3 vel pred).
            use_diff_cam: legacy separate camera head (sigmoid absolute params).
            use_unified_control: Paper §2.1 — camera deltas are part of action output,
                action dim becomes dim_action + 4; cam deltas via tanh (incremental).
            use_cam_obs: Paper §2.1 — include current camera state in observation
                (adds 4 dims: normalized FOV, exposure, ISO, focus).
        """
        super().__init__()
        self.use_diff_cam = use_diff_cam
        self.use_unified_control = use_unified_control
        self.use_cam_obs = use_cam_obs

        # If cam_obs is enabled, add 4 dims to observation
        actual_obs_dim = dim_obs + (4 if use_cam_obs else 0)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.v_proj = nn.Linear(actual_obs_dim, 192)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)

        if use_unified_control:
            # Unified output: dim_action (flight) + 4 (camera deltas)
            total_action_dim = dim_action + 4
            self.fc = nn.Linear(192, total_action_dim, bias=False)
            self.fc.weight.data.mul_(0.01)
            self._flight_dim = dim_action
        else:
            self.fc = nn.Linear(192, dim_action, bias=False)
            self.fc.weight.data.mul_(0.01)
            self._flight_dim = dim_action

        if use_diff_cam and not use_unified_control:
            # Legacy separate camera head: [fov_delta, exposure, iso, focus_dist]
            # All outputs pass through sigmoid to map to [0, 1]
            self.fc_cam = nn.Linear(192, 4, bias=True)
            self.fc_cam.weight.data.mul_(0.01)
            # Initialize bias so that sigmoid output starts near 0.5 (default params)
            self.fc_cam.bias.data.zero_()

        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))
        hx = self.gru(x, hx)
        raw = self.fc(self.act(hx))

        if self.use_unified_control:
            # Split into flight action + camera deltas
            flight_act = raw[:, :self._flight_dim]
            cam_deltas = torch.tanh(raw[:, self._flight_dim:])  # (B, 4) in [-1, 1]
            return flight_act, cam_deltas, hx

        # Legacy path
        act = raw
        cam_params = None
        if self.use_diff_cam:
            cam_raw = self.fc_cam(self.act(hx))
            cam_params = torch.sigmoid(cam_raw)  # (B, 4) all in [0, 1]

        return act, cam_params, hx


if __name__ == '__main__':
    Model()
