import torch
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4, use_diff_cam=False) -> None:
        super().__init__()
        self.use_diff_cam = use_diff_cam
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
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)

        if use_diff_cam:
            # Camera parameter head: [fov_delta, exposure, iso, focus_dist]
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
        act = self.fc(self.act(hx))

        cam_params = None
        if self.use_diff_cam:
            cam_raw = self.fc_cam(self.act(hx))
            cam_params = torch.sigmoid(cam_raw)  # (B, 4) all in [0, 1]

        return act, cam_params, hx


if __name__ == '__main__':
    Model()
