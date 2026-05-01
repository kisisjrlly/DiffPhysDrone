import torch
from torch import nn
import torch.nn.functional as F

from utils import g_decay

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4,
                 include_camera_state_in_obs=False,
                 use_policy_intent=False, intent_dim=9,
                 depth_nn_width=16,
                 depth_nn_height=12,
                 depth_use_pipeline=True,
                 depth_min_valid=0.3,
                 depth_max_range=6.0) -> None:
        """
        初始化无人机的策略网络模型 (Policy Network)。
        Args:
            dim_obs: 基础物理观测维度（通常是7维无里程计，或10维带里程计）。
            dim_action: 飞行控制动作维度（默认6维：3维加速度 + 3维速度预测）。
            include_camera_state_in_obs: 是否允许 camera head 使用当前相机状态。
                flight head 始终不直接接收相机状态，避免 camera 参数成为动作捷径。
        """
        super().__init__()
        self.include_camera_state_in_obs = bool(include_camera_state_in_obs)
        self.use_policy_intent = use_policy_intent
        self.intent_dim = intent_dim
        self.depth_nn_width = max(int(depth_nn_width), 1)
        self.depth_nn_height = max(int(depth_nn_height), 1)
        self.depth_use_pipeline = bool(depth_use_pipeline)
        self.depth_min_valid = max(float(depth_min_valid), 1e-3)
        self.depth_max_range = max(float(depth_max_range), self.depth_min_valid + 1e-3)

        self.feat_dim = 96
        self.cam_state_dim = 32
        self.cam_motion_dim = 32
        self.cam_hidden_dim = 64
        self.state_feat_scale = 0.60
        self.state_dropout_p = 0.15

        def make_spatial_stem(cin: int, small_input_friendly: bool = False):
            _ = small_input_friendly
            return nn.Sequential(
                nn.Conv2d(cin, 16, 3, padding=1, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.LeakyReLU(0.05),
                nn.AdaptiveAvgPool2d((2, 4)),
            )

        def make_stem(cin: int, feat_dim: int, small_input_friendly: bool = False):
            return nn.Sequential(
                make_spatial_stem(cin, small_input_friendly=small_input_friendly),
                nn.Flatten(),
                nn.Linear(64 * 2 * 4, feat_dim, bias=False),
            )

        # diff_depth-only：双通道深度分支编码器。
        # channel 0: inverse depth / near obstacle cue
        # channel 1: metric range / far opening cue
        self.stem = make_stem(
            2,
            self.feat_dim,
            small_input_friendly=True,
        )

        # 状态向量投影层：容量故意小一些，避免 fixed/randfix 过度依赖状态/时间模板。
        self.v_proj = nn.Linear(dim_obs, self.feat_dim)
        self.v_proj.weight.data.mul_(0.25)

        # 多模态门控融合（替代简单相加）
        # 目标：防止训练后期某一模态长期压制另一模态
        self.img_norm = nn.LayerNorm(self.feat_dim)
        self.v_norm = nn.LayerNorm(self.feat_dim)
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim, bias=True),
            nn.LeakyReLU(0.05),
            nn.Linear(self.feat_dim, self.feat_dim, bias=True),
            nn.Sigmoid(),
        )
        self.fuse_gate[-2].weight.data.mul_(0.1)
        self.fuse_gate[-2].bias.data.fill_(0.35)

        # 门控循环单元 (GRU)：用于处理时序信息，赋予无人机记忆能力
        # 因为无人机只能看到当前的深度图（部分可观测环境 POMDP），需要记忆来推断自身速度和环境结构
        self.gru = nn.GRUCell(self.feat_dim, self.feat_dim)
        # GRU 后残差稳态头：提升记忆表达稳定性，缓解后期策略抖动
        self.gru_residual = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim, bias=False),
            nn.LeakyReLU(0.05),
            nn.Linear(self.feat_dim, self.feat_dim, bias=False),
        )
        self.gru_residual[-1].weight.data.mul_(0.01)
        self.hx_norm = nn.LayerNorm(self.feat_dim)

        # 动作输出头 (Action Head)
        self.fc = nn.Linear(self.feat_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self._flight_dim = dim_action

        # 可选的意图输出头（用于意图域训练 + dLQR）
        if self.use_policy_intent:
            self.fc_intent = nn.Linear(self.feat_dim, intent_dim, bias=True)
            self.fc_intent.weight.data.mul_(0.01)
            self.fc_intent.bias.data.zero_()

        # Camera controller: image feature + current camera state + local
        # motion/attitude.  Local motion is included because exposure/blur are
        # motion-dependent, while target direction/distance are intentionally
        # excluded to avoid a camera schedule keyed directly on task geometry.
        self.cam_state_proj = nn.Linear(3, self.cam_state_dim)
        self.cam_state_norm = nn.LayerNorm(self.cam_state_dim)
        self.cam_motion_proj = nn.Linear(6, self.cam_motion_dim)
        self.cam_motion_norm = nn.LayerNorm(self.cam_motion_dim)
        self.cam_pre = nn.Sequential(
            nn.Linear(self.feat_dim + self.cam_state_dim + self.cam_motion_dim, self.cam_hidden_dim, bias=True),
            nn.LeakyReLU(0.05),
        )
        self.cam_gru = nn.GRUCell(self.cam_hidden_dim, self.cam_hidden_dim)
        self.cam_hx_norm = nn.LayerNorm(self.cam_hidden_dim)
        self.fc_cam = nn.Linear(self.cam_hidden_dim, 3, bias=True)
        self.fc_cam.weight.data.mul_(0.01)
        self.fc_cam.bias.data.zero_()

        # 全局使用的激活函数
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        # 重置函数，当前为空。
        # 因为 GRU 的隐藏状态 hx 是在外部循环 (main_cuda.py) 中手动维护并传入的，而不是存在模型内部。
        pass

    @staticmethod
    def _as_bchw(x: torch.Tensor):
        if x is None:
            return None
        if x.dim() == 3:
            return x[:, None]
        return x

    def _depth_pipeline(self, depth_like: torch.Tensor):
        """
        diff_depth 深度输入流水线：
        1) 近障碍通道：深度反转并归一化，近处值更大。
        2) 远处/开口通道：metric range 归一化，远处值更大。
        3) 两个通道使用不同 pooling：近障碍用 max-pool，开口 cue 用 avg-pool。
        4) 保留无效深度为 0，避免被误当作近距离障碍物或开口。
        """
        # 支持 (B,H,W) 或 (B,1,H,W)
        d = depth_like
        if d.dim() == 4 and d.shape[1] == 1:
            d = d[:, 0]
        elif d.dim() != 3:
            raise ValueError(f'depth pipeline 期望输入为 (B,H,W) 或 (B,1,H,W)，实际: {tuple(d.shape)}')

        min_depth = self.depth_min_valid
        max_depth = self.depth_max_range
        valid = d >= min_depth
        d_valid = torch.where(valid, d.clamp(min_depth, max_depth), torch.full_like(d, max_depth))
        inv = 1.0 / d_valid
        inv_min = 1.0 / max_depth
        inv_max = 1.0 / min_depth
        near = ((inv - inv_min) / (inv_max - inv_min)).clamp(0.0, 1.0) * valid.float()
        far = ((d_valid - min_depth) / (max_depth - min_depth)).clamp(0.0, 1.0) * valid.float()

        target_h = max(int(self.depth_nn_height), 1)
        target_w = max(int(self.depth_nn_width), 1)
        up_h = max(int(near.shape[-2]), target_h)
        up_w = max(int(near.shape[-1]), target_w)
        if up_h != int(near.shape[-2]) or up_w != int(near.shape[-1]):
            near = F.interpolate(
                near[:, None],
                size=(up_h, up_w),
                mode='nearest',
            )[:, 0]
            far = F.interpolate(
                far[:, None],
                size=(up_h, up_w),
                mode='nearest',
            )[:, 0]

        near = F.adaptive_max_pool2d(near[:, None], (target_h, target_w))
        far = F.adaptive_avg_pool2d(far[:, None], (target_h, target_w))
        return torch.cat([near, far], dim=1)

    def preprocess_depth_input(self, depth_obs=None, add_noise=False):
        """
        将原始深度观测转换成模型可直接消费的张量。

        这一步做两类事情：
        1) 数值域映射（例如把深度映射到更适合网络学习的范围）
        2) 形状整理（确保是 B,C,H,W）
        """
        if depth_obs is None:
            raise ValueError('diff_depth-only 模型需要 depth_obs 输入')

        if self.depth_use_pipeline:
            x_depth = self._depth_pipeline(depth_obs)
            if add_noise:
                x_depth = (x_depth + torch.randn_like(x_depth) * 0.01).clamp(0.0, 1.0)
            x_depth = x_depth * 2.0 - 1.0
        else:
            min_depth = self.depth_min_valid
            max_depth = self.depth_max_range
            valid = depth_obs >= min_depth
            safe_depth = torch.where(
                valid,
                depth_obs.clamp(min_depth, max_depth),
                torch.full_like(depth_obs, max_depth),
            )
            inv = 1.0 / safe_depth
            inv_min = 1.0 / max_depth
            inv_max = 1.0 / min_depth
            near = ((inv - inv_min) / (inv_max - inv_min)).clamp(0.0, 1.0) * valid.float()
            far = ((safe_depth - min_depth) / (max_depth - min_depth)).clamp(0.0, 1.0) * valid.float()
            x_depth = torch.stack([near, far], dim=1)
            if add_noise:
                x_depth = (x_depth + torch.randn_like(x_depth) * 0.01).clamp(0.0, 1.0)
            x_depth = x_depth * 2.0 - 1.0

        return x_depth

    def forward(self, v, hx=None, return_intent=False,
                depth_obs=None, add_noise=False, cam_hx=None,
                camera_state=None, camera_motion_state=None):
        """
        前向传播函数。
        Args:
            x: 视觉观测输入（深度图 Tensor）。
            v: flight 物理状态观测输入（不包含相机状态）。
            hx: flight GRU 的隐藏状态（记忆）。
            camera_state: 当前 power/exposure/gain，归一化到 [-1, 1]。
            camera_motion_state: 局部速度 + 姿态/up，用于 motion-aware camera 控制。
            cam_hx: camera GRU 的隐藏状态（只建模成像/运动状态历史）。
        Returns:
            flight_act: 飞行控制动作。
            cam_params: 相机控制参数（增量或绝对值）。
            hx/cam_hx: 更新后的 flight/camera GRU 隐藏状态。
        """
        # ==========================
        # A. 深度输入预处理
        x = self.preprocess_depth_input(depth_obs=depth_obs, add_noise=add_noise)

        # ==========================
        # B. 视觉特征提取
        # ==========================
        img_feat = self.stem(x)
        
        # ==========================
        # C. 多模态融合 + 时序建模
        # ==========================
        # 门控融合：平衡视觉特征与状态向量，降低单模态长期压制风险
        img_feat = self.img_norm(img_feat)
        v_feat = self.v_norm(self.v_proj(v)) * self.state_feat_scale
        if self.training and self.state_dropout_p > 0.0:
            v_feat = F.dropout(v_feat, p=self.state_dropout_p, training=True)
        fuse_gate = self.fuse_gate(torch.cat([img_feat, v_feat], dim=1))
        flight_feat = self.act(fuse_gate * img_feat + (1.0 - fuse_gate) * v_feat)
        
        # GRU 累积时序上下文（POMDP 下尤为关键）
        hx = self.gru(flight_feat, hx)
        hx = self.hx_norm(hx + 0.1 * self.gru_residual(hx))
        
        # 输出动作头原始值
        raw = self.fc(self.act(hx))

        act = raw
        if camera_state is None or not self.include_camera_state_in_obs:
            cam_state = torch.zeros(v.shape[0], 3, device=v.device, dtype=v.dtype)
        else:
            cam_state = camera_state.to(device=v.device, dtype=v.dtype)
        if camera_motion_state is None:
            camera_motion_state = torch.zeros(v.shape[0], 6, device=v.device, dtype=v.dtype)
        else:
            camera_motion_state = camera_motion_state.to(device=v.device, dtype=v.dtype)
        cam_feat = self.cam_state_norm(self.cam_state_proj(cam_state))
        cam_motion_feat = self.cam_motion_norm(self.cam_motion_proj(camera_motion_state))
        cam_in = self.cam_pre(torch.cat([img_feat, cam_feat, cam_motion_feat], dim=1))
        cam_hx = self.cam_gru(cam_in, cam_hx)
        cam_hx = self.cam_hx_norm(cam_hx)
        cam_raw = self.fc_cam(self.act(cam_hx))
        cam_params = torch.tanh(cam_raw)  # normalized camera delta in [-1, 1]
        if return_intent and self.use_policy_intent:
            intent_raw = self.fc_intent(self.act(hx))
            return act, cam_params, hx, intent_raw, cam_hx
        return act, cam_params, hx, cam_hx

if __name__ == '__main__':
    # 简单的测试代码，确保模型可以被实例化
    Model()
