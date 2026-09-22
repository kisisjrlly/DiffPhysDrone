import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """Recurrent grayscale flight policy with a 2-D exposure/gain controller."""

    def __init__(
        self,
        dim_obs=9,
        dim_action=3,
        *,
        include_camera_state_in_obs=False,
        use_policy_intent=False,
        intent_dim=9,
        gray_nn_width=48,
        gray_nn_height=36,
    ):
        super().__init__()
        self.include_camera_state_in_obs = bool(include_camera_state_in_obs)
        self.use_policy_intent = bool(use_policy_intent)
        self.intent_dim = int(intent_dim)
        self.gray_nn_width = max(int(gray_nn_width), 1)
        self.gray_nn_height = max(int(gray_nn_height), 1)
        self.camera_action_dim = 2

        def make_spatial_stem(cin):
            return nn.Sequential(
                nn.Conv2d(cin, 32, 3, padding=1, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.LeakyReLU(0.05),
                nn.AdaptiveAvgPool2d((3, 6)),
            )

        def make_stem(cin, feat_dim):
            return nn.Sequential(
                make_spatial_stem(cin),
                nn.Flatten(),
                nn.Linear(128 * 3 * 6, feat_dim, bias=False),
            )

        self.feat_dim = 192
        self.cam_state_dim = 24
        self.cam_motion_dim = 24
        self.cam_hidden_dim = 96

        actual_obs_dim = dim_obs + (2 if self.include_camera_state_in_obs else 0)

        # Flight and camera branches intentionally use separate visual stems.
        # This allows frozen-flight camera training without moving flight visual
        # features, and vice versa.
        self.stem = make_stem(2, self.feat_dim)
        self.cam_stem = make_stem(2, self.feat_dim)
        self.cam_stem.load_state_dict(self.stem.state_dict())

        self.v_proj = nn.Linear(actual_obs_dim, self.feat_dim)
        self.v_proj.weight.data.mul_(0.5)
        self.img_norm = nn.LayerNorm(self.feat_dim)
        self.v_norm = nn.LayerNorm(self.feat_dim)
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.LeakyReLU(0.05),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Sigmoid(),
        )
        self.fuse_gate[-2].weight.data.mul_(0.1)
        self.fuse_gate[-2].bias.data.zero_()

        self.gru = nn.GRUCell(self.feat_dim, self.feat_dim)
        self.gru_residual = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim, bias=False),
            nn.LeakyReLU(0.05),
            nn.Linear(self.feat_dim, self.feat_dim, bias=False),
        )
        self.gru_residual[-1].weight.data.mul_(0.01)
        self.hx_norm = nn.LayerNorm(self.feat_dim)

        self.fc = nn.Linear(self.feat_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        if self.use_policy_intent:
            self.fc_intent = nn.Linear(self.feat_dim, self.intent_dim)
            self.fc_intent.weight.data.mul_(0.01)
            self.fc_intent.bias.data.zero_()

        self.cam_spatial_stem = nn.Sequential(
            nn.Conv2d(2, 4, 3, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(4, 4, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.AdaptiveAvgPool2d((2, 3)),
        )
        self.cam_spatial_proj = nn.Linear(4 * 2 * 3, self.feat_dim)
        self.cam_spatial_proj.weight.data.mul_(0.05)
        self.cam_spatial_proj.bias.data.zero_()
        self.cam_img_adapter = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.LeakyReLU(0.05),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.cam_img_adapter[-1].weight.data.mul_(0.01)
        self.cam_img_adapter[-1].bias.data.zero_()
        self.cam_img_norm = nn.LayerNorm(self.feat_dim)

        self.cam_state_proj = nn.Linear(2, self.cam_state_dim)
        self.cam_state_norm = nn.LayerNorm(self.cam_state_dim)
        self.cam_motion_proj = nn.Linear(6, self.cam_motion_dim)
        self.cam_motion_norm = nn.LayerNorm(self.cam_motion_dim)
        self.cam_pre = nn.Sequential(
            nn.Linear(
                self.feat_dim + self.cam_state_dim + self.cam_motion_dim,
                self.cam_hidden_dim,
            ),
            nn.LeakyReLU(0.05),
        )
        self.cam_gru = nn.GRUCell(self.cam_hidden_dim, self.cam_hidden_dim)
        self.cam_hx_norm = nn.LayerNorm(self.cam_hidden_dim)
        self.fc_cam = nn.Linear(self.cam_hidden_dim, 2)
        self.fc_cam.weight.data.mul_(0.01)
        self.fc_cam.bias.data.zero_()

        self.act = nn.LeakyReLU(0.05)

    @staticmethod
    def _is_camera_parameter(name):
        return name.startswith((
            "cam_stem",
            "cam_spatial_stem",
            "cam_spatial_proj",
            "cam_img_adapter",
            "cam_img_norm",
            "cam_state_proj",
            "cam_state_norm",
            "cam_motion_proj",
            "cam_motion_norm",
            "cam_pre",
            "cam_gru",
            "cam_hx_norm",
            "fc_cam",
        ))

    def freeze_camera_for_flight_only(self):
        frozen = []
        for name, param in self.named_parameters():
            if self._is_camera_parameter(name):
                param.requires_grad_(False)
                frozen.append(name)
        return frozen

    def freeze_flight_for_camera_only(self):
        frozen = []
        for name, param in self.named_parameters():
            if not self._is_camera_parameter(name):
                param.requires_grad_(False)
                frozen.append(name)
        return frozen

    def reset(self):
        pass

    def preprocess_gray_input(self, gray_obs, add_noise=False):
        if gray_obs is None:
            raise ValueError("gray_obs is required")
        x = gray_obs
        if x.dim() == 3:
            x = x[:, None]
        if x.dim() != 4:
            raise ValueError(f"gray input must be BCHW, got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = torch.cat([x, x], dim=1)
        if x.shape[1] != 2:
            raise ValueError("gray input must contain current and previous frames")
        target = (self.gray_nn_height, self.gray_nn_width)
        if tuple(x.shape[-2:]) != target:
            x = F.adaptive_avg_pool2d(x, target)
        # Camera output is hard-clipped to [0,1] in forward, so policy input
        # can safely be normalized to [-1,1].
        x = x.clamp(0.0, 1.0)
        if add_noise:
            x = (x + torch.randn_like(x) * 0.005).clamp(0.0, 1.0)
        return x * 2.0 - 1.0

    def forward(
        self,
        v,
        hx=None,
        *,
        gray_obs=None,
        add_noise=False,
        cam_hx=None,
        camera_state=None,
        camera_motion_state=None,
        return_intent=False,
    ):
        x = self.preprocess_gray_input(gray_obs, add_noise=add_noise)
        img_feat = self.img_norm(self.stem(x))
        v_feat = self.v_norm(self.v_proj(v))
        gate = self.fuse_gate(torch.cat([img_feat, v_feat], dim=1))
        fused = self.act(gate * img_feat + (1.0 - gate) * v_feat)

        hx = self.gru(fused, hx)
        hx = self.hx_norm(hx + 0.1 * self.gru_residual(hx))
        flight_act = self.fc(self.act(hx))

        if camera_state is None:
            camera_state = torch.zeros(v.shape[0], 2, device=v.device, dtype=v.dtype)
        else:
            camera_state = camera_state.to(device=v.device, dtype=v.dtype)
        if camera_motion_state is None:
            camera_motion_state = torch.zeros(v.shape[0], 6, device=v.device, dtype=v.dtype)
        else:
            camera_motion_state = camera_motion_state.to(device=v.device, dtype=v.dtype)

        cam_raw = self.cam_stem(x)
        cam_spatial = self.cam_spatial_proj(self.cam_spatial_stem(x).flatten(1))
        cam_img = self.cam_img_norm(
            cam_raw + 0.25 * self.cam_img_adapter(cam_raw) + 0.35 * cam_spatial
        )
        cam_state_feat = self.cam_state_norm(self.cam_state_proj(camera_state))
        cam_motion_feat = self.cam_motion_norm(self.cam_motion_proj(camera_motion_state))
        cam_in = self.cam_pre(torch.cat([cam_img, cam_state_feat, cam_motion_feat], dim=1))
        cam_hx = self.cam_gru(cam_in, cam_hx)
        cam_hx = self.cam_hx_norm(cam_hx)
        cam_params = torch.sigmoid(self.fc_cam(self.act(cam_hx)))

        if return_intent and self.use_policy_intent:
            return (
                flight_act,
                cam_params,
                hx,
                self.fc_intent(self.act(hx)),
                cam_hx,
            )
        return flight_act, cam_params, hx, cam_hx


if __name__ == "__main__":
    Model()
