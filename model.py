import torch
from torch import nn

from utils import g_decay

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4,
                 include_camera_state_in_obs=False,
                 in_channels=1, use_policy_intent=False, intent_dim=9,
                 main_in_channels=1,
                 enable_camera_head=True,
                 depth_nn_width=16,
                 depth_nn_height=12,
                 depth_use_pipeline=True,
                 sensor_mode='camera_luma_plus_depth') -> None:
        """
        初始化无人机的策略网络模型 (Policy Network)。
        Args:
            dim_obs: 基础物理观测维度（通常是7维无里程计，或10维带里程计）。
            dim_action: 飞行控制动作维度（默认6维：3维加速度 + 3维速度预测）。
            include_camera_state_in_obs: 是否将当前相机状态加入到观测向量中。
        """
        super().__init__()
        self.include_camera_state_in_obs = bool(include_camera_state_in_obs)
        self.use_policy_intent = use_policy_intent
        self.intent_dim = intent_dim
        self.main_in_channels = main_in_channels
        self.enable_camera_head = bool(enable_camera_head)
        self.depth_nn_width = max(int(depth_nn_width), 1)
        self.depth_nn_height = max(int(depth_nn_height), 1)
        self.depth_use_pipeline = bool(depth_use_pipeline)
        self.sensor_mode = self._normalize_sensor_mode(sensor_mode)

        def make_spatial_stem(cin: int):
            return nn.Sequential(
                nn.Conv2d(cin, 32, 2, 2, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(32, 64, 3, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(64, 128, 3, bias=False),
                nn.LeakyReLU(0.05),
                nn.AdaptiveAvgPool2d((2, 4)),
            )

        def make_stem(cin: int, feat_dim: int):
            return nn.Sequential(
                make_spatial_stem(cin),
                nn.Flatten(),
                nn.Linear(128 * 2 * 4, feat_dim, bias=False),
            )

        # 如果启用了相机状态观测，物理观测维度需要增加 3 维（FOV, 曝光, ISO）
        actual_obs_dim = dim_obs + (3 if self.include_camera_state_in_obs else 0)

        # 视觉特征提取主干网络 (CNN Stem)
        # 按 sensor_mode 固定网络结构：
        #   depth: 单分支
        #   camera_luma: 单分支
        #   camera_luma_plus_depth: 双分支 + 特征融合
        #   diff_depth: 单分支（只用可微深度）
        if self.sensor_mode == 'camera_luma_plus_depth':
            # 池化后网格级空间引导融合 (Grid-level Spatial Guidance):
            # 将两路特征都在 2x4 (即 8 个空间块) 级别进行对齐，再用主视觉为深度生成空间门控。
            # 保留了 DEPTHOR 的空间语义对应特性，同时也完美避免高分辨率对齐带来的 OOM。
            self.stem_main_spatial = make_spatial_stem(self.main_in_channels)
            depth_in = 1
            self.stem_depth_spatial = make_spatial_stem(depth_in)
            self.depth_gate = nn.Sequential(
                nn.Conv2d(128, 32, 1, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(32, 128, 1, bias=False),
                nn.Sigmoid(),
            )
            self.depth_gate[0].weight.data.mul_(0.5)
            self.fuse_spatial = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 2 * 4, 192, bias=False)
            )
            self.fuse_spatial[-1].weight.data.mul_(0.5)
        elif self.sensor_mode in ('depth', 'camera_luma', 'diff_depth'):
            stem_in_channels = in_channels
            if self.sensor_mode == 'diff_depth':
                stem_in_channels = 1
            self.stem = make_stem(stem_in_channels, 192)
        else:
            raise ValueError(f'unsupported sensor_mode: {self.sensor_mode}')

        # 状态向量投影层：将无人机的物理状态（速度、姿态、目标距离等）映射到 192 维
        self.v_proj = nn.Linear(actual_obs_dim, 192)
        # 初始化权重，乘以 0.5 使其初始输出较小，防止物理状态特征在初期淹没视觉特征
        self.v_proj.weight.data.mul_(0.5)

        # 门控循环单元 (GRU)：用于处理时序信息，赋予无人机记忆能力
        # 因为无人机只能看到当前的深度图（部分可观测环境 POMDP），需要记忆来推断自身速度和环境结构
        self.gru = nn.GRUCell(192, 192)

        # 动作输出头 (Action Head)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self._flight_dim = dim_action

        # 可选的意图输出头（用于意图域训练 + dLQR）
        if self.use_policy_intent:
            self.fc_intent = nn.Linear(192, intent_dim, bias=True)
            self.fc_intent.weight.data.mul_(0.01)
            self.fc_intent.bias.data.zero_()

        # 可微相机头（绝对参数）
        if self.enable_camera_head:
            # 输出 3 个相机参数的绝对值: FOV, Exposure, ISO
            self.fc_cam = nn.Linear(192, 3, bias=True)
            self.fc_cam.weight.data.mul_(0.01)
            # 初始化偏置为 0，这样经过后续的 sigmoid 激活后，初始输出在 0.5 附近（即默认的居中参数）
            self.fc_cam.bias.data.zero_()

        # 全局使用的激活函数
        self.act = nn.LeakyReLU(0.05)

    @staticmethod
    def _normalize_sensor_mode(raw_mode: str) -> str:
        key = str(raw_mode).strip().lower()
        allowed = {
            'depth',
            'camera_luma',
            'camera_luma_plus_depth',
            'diff_depth',
        }
        if key not in allowed:
            raise ValueError(
                f"unsupported sensor_mode: {raw_mode}. "
                f"allowed={sorted(list(allowed))}"
            )
        return key

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
        深度输入流水线（可用于 diff_depth / camera_luma_plus_depth）：
        1) 深度反转并归一化（近处值更大，范围 [0.05~24m] -> [0,1]）
        2) 直接 2x2 最大池化从 (2*H_nn, 2*W_nn) 到 (H_nn, W_nn)
        
        假设输入原始深度分辨率为 (2*H_nn, 2*W_nn)，例如 32×24。
        最大池化保留每个 2×2 窗口内的最近物体信息，输出为 16×12。
        """
        # 支持 (B,H,W) 或 (B,1,H,W)
        d = depth_like
        if d.dim() == 4 and d.shape[1] == 1:
            d = d[:, 0]
        elif d.dim() != 3:
            raise ValueError(f'depth pipeline 期望输入为 (B,H,W) 或 (B,1,H,W)，实际: {tuple(d.shape)}')

        # 反转 + 归一化：把 0.05~24m 映射到 [0,1]，近处更亮
        d = d.clamp(0.05, 24.0)
        inv = 1.0 / d
        inv_min = 1.0 / 24.0
        inv_max = 1.0 / 0.05
        x = (inv - inv_min) / (inv_max - inv_min)
        x = x.clamp(0.0, 1.0)

        # 直接从原始深度分辨率 (2*H_nn, 2*W_nn) 通过 2x2 最大池化到目标尺寸 (H_nn, W_nn)
        # 例如：32×24 -> 16×12
        x = torch.nn.functional.max_pool2d(x[:, None], kernel_size=2, stride=2)
        return x

    def preprocess_sensor_inputs(self, main_obs=None, depth_obs=None,
                                 add_noise=False):
        """
        将“原始传感器观测”转换成模型可直接消费的张量。

        这一步做两类事情：
        1) 数值域映射（例如把深度映射到更适合网络学习的范围）
        2) 形状整理（确保是 B,C,H,W）

        返回:
            x_fused: 单张量融合输入（兼容单输入路径）
            x_main:  主传感器分支输入（给 dual-encoder 的 main stem）
            x_depth_pack: 深度分支输入
        """
        x_main = None
        x_depth = None

        if self.sensor_mode in ('depth', 'camera_luma', 'camera_luma_plus_depth'):
            if main_obs is None:
                raise ValueError(f"sensor_mode={self.sensor_mode} 需要 main_obs 输入")
            if self.sensor_mode == 'depth':
                x_main = 3 / main_obs.clamp(0.3, 24) - 0.6
                if add_noise:
                    x_main = x_main + torch.randn_like(x_main) * 0.02
            else:
                x_main = main_obs.clamp(0.0, 1.0)
                if add_noise:
                    x_main = (x_main + torch.randn_like(x_main) * 0.01).clamp(0.0, 1.0)
                x_main = x_main * 2.0 - 1.0
            x_main = self._as_bchw(x_main)
        elif self.sensor_mode == 'diff_depth':
            pass
        else:
            raise ValueError(f"unsupported sensor_mode: {self.sensor_mode}")

        if self.sensor_mode in ('camera_luma_plus_depth', 'diff_depth'):
            if depth_obs is None:
                raise ValueError(f"sensor_mode={self.sensor_mode} 需要 depth_obs 输入")
            if self.depth_use_pipeline:
                x_depth = self._depth_pipeline(depth_obs)
                if add_noise:
                    x_depth = (x_depth + torch.randn_like(x_depth) * 0.01).clamp(0.0, 1.0)
                x_depth = x_depth * 2.0 - 1.0
            else:
                if self.sensor_mode == 'diff_depth':
                    # diff_depth 直输模式：回退到原逻辑，不做分辨率流水线处理
                    x_depth = 3 / depth_obs.clamp(0.05, 24) - 0.6
                    if add_noise:
                        x_depth = x_depth + torch.randn_like(x_depth) * 0.01
                    x_depth = self._as_bchw(x_depth)
                else:
                    # camera_luma_plus_depth 直输模式
                    # 依然使用倒数深度保留对近处物体的敏感性
                    inv_depth = 3 / depth_obs.clamp(0.05, 24) - 0.6
                    # 使用 Tanh 软截断，抑制近距离深度突变导致的极端值
                    x_depth = torch.tanh(inv_depth / 3.0)
                    if add_noise:
                        x_depth = x_depth + torch.randn_like(x_depth) * 0.01
                    x_depth = self._as_bchw(x_depth)
        else:
            if depth_obs is not None:
                raise ValueError(f"sensor_mode={self.sensor_mode} 不应提供深度输入")

        x_depth_pack = x_depth

        channels = []
        if x_main is not None:
            channels.append(x_main)
        if x_depth_pack is not None:
            channels.append(x_depth_pack)
        if len(channels) == 0:
            raise ValueError("preprocess_sensor_inputs 需要至少一种传感器输入")

        if self.sensor_mode == 'camera_luma_plus_depth':
            x_fused = x_main
        else:
            x_fused = torch.cat(channels, 1)
        return x_fused, x_main, x_depth_pack

    def forward(self, v, hx=None, return_intent=False,
                main_obs=None, depth_obs=None,
                add_noise=False):
        """
        前向传播函数。
        Args:
            x: 视觉观测输入（深度图 Tensor）。
            v: 物理状态观测输入（包含速度、姿态等，可能包含相机状态）。
            hx: GRU 的隐藏状态（记忆）。
        Returns:
            flight_act: 飞行控制动作。
            cam_params: 相机控制参数（增量或绝对值）。
            hx: 更新后的 GRU 隐藏状态。
        """
        # ==========================
        # A. 严格模式输入：仅接受传感器原始输入
        x, x_main, x_depth = self.preprocess_sensor_inputs(
            main_obs=main_obs,
            depth_obs=depth_obs,
            add_noise=add_noise,
        )

        # ==========================
        # B. 视觉特征提取
        # ==========================
        if self.sensor_mode == 'camera_luma_plus_depth':
            if x_main is None or x_depth is None:
                raise ValueError('sensor_mode=camera_luma_plus_depth 需要同时提供 main 与 depth 输入')
            # 分别提取空间网格特征 [B, 128, 2, 4]
            feat_main_grid = self.stem_main_spatial(x_main)
            feat_depth_grid = self.stem_depth_spatial(x_depth)
            # 空间语义引导：用 main 特征在 2x4 的网格上为深度分支生成空间+通道 mask
            depth_grid_refined = feat_depth_grid * self.depth_gate(feat_main_grid)
            # 融合并降维到 192 维
            img_feat = self.fuse_spatial(torch.cat([feat_main_grid, depth_grid_refined], dim=1))
        else:
            img_feat = self.stem(x)
        
        # ==========================
        # C. 多模态融合 + 时序建模
        # ==========================
        # 视觉特征 + 状态向量投影后做非线性激活
        x = self.act(img_feat + self.v_proj(v))
        
        # GRU 累积时序上下文（POMDP 下尤为关键）
        hx = self.gru(x, hx)
        
        # 输出动作头原始值
        raw = self.fc(self.act(hx))

        act = raw
        cam_params = None
        if self.enable_camera_head:
            cam_raw = self.fc_cam(self.act(hx))
            cam_params = torch.sigmoid(cam_raw)  # (Batch, 3)
        if return_intent and self.use_policy_intent:
            intent_raw = self.fc_intent(self.act(hx))
            return act, cam_params, hx, intent_raw
        return act, cam_params, hx

if __name__ == '__main__':
    # 简单的测试代码，确保模型可以被实例化
    Model()