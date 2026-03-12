import torch
from torch import nn

# 这是一个自定义的梯度衰减函数。
# 它的作用是在反向传播时，将梯度乘以一个衰减系数 alpha。
# 前向传播时，它的值就是 x 本身 (x * alpha + x * (1 - alpha) = x)。
# 这在长序列的 RNN/GRU 训练中常用于缓解梯度爆炸，或在教师-学生训练中控制梯度流。
def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4,
                 camera_action_mode='off', include_camera_state_in_obs=False,
                 in_channels=1, use_policy_intent=False, intent_dim=9,
                 main_in_channels=1,
                 use_tof_conf=False,
                 tof_nn_width=16,
                 tof_nn_height=12,
                 active_depth_use_pipeline=True,
                 sensor_mode='camera_luma_plus_passive_depth') -> None:
        """
        初始化无人机的策略网络模型 (Policy Network)。
        Args:
            dim_obs: 基础物理观测维度（通常是7维无里程计，或10维带里程计）。
            dim_action: 飞行控制动作维度（默认6维：3维加速度 + 3维速度预测）。
            camera_action_mode: 相机动作模式，off|absolute|incremental。
            include_camera_state_in_obs: 是否将当前相机状态加入到观测向量中。
        """
        super().__init__()
        # 保存配置标志，用于在 forward 中决定走哪条计算路径
        self.camera_action_mode = str(camera_action_mode).strip().lower()
        if self.camera_action_mode not in ('off', 'absolute', 'incremental'):
            raise ValueError(f"camera_action_mode must be one of off|absolute|incremental, got: {camera_action_mode}")
        self.include_camera_state_in_obs = bool(include_camera_state_in_obs)
        self.use_policy_intent = use_policy_intent
        self.intent_dim = intent_dim
        self.main_in_channels = main_in_channels
        self.use_tof_conf = use_tof_conf
        self.tof_nn_width = max(int(tof_nn_width), 1)
        self.tof_nn_height = max(int(tof_nn_height), 1)
        self.active_depth_use_pipeline = bool(active_depth_use_pipeline)
        self.sensor_mode = self._normalize_sensor_mode(sensor_mode)

        def make_stem(cin: int, feat_dim: int):
            return nn.Sequential(
                nn.Conv2d(cin, 32, 2, 2, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(32, 64, 3, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(64, 128, 3, bias=False),
                nn.LeakyReLU(0.05),
                nn.AdaptiveAvgPool2d((2, 4)),
                nn.Flatten(),
                nn.Linear(128 * 2 * 4, feat_dim, bias=False),
            )

        # 如果启用了相机状态观测，物理观测维度需要增加 3 维（FOV, 曝光, ISO）
        actual_obs_dim = dim_obs + (3 if self.include_camera_state_in_obs else 0)

        # 视觉特征提取主干网络 (CNN Stem)
        # 按 sensor_mode 固定网络结构：
        #   passive_depth: 单分支
        #   camera_luma: 单分支
        #   camera_luma_plus_passive_depth: 双分支 + 特征融合
        #   active_depth: 单分支（只用主动深度/置信度）
        if self.sensor_mode == 'camera_luma_plus_passive_depth':
            self.main_feat_dim = 96
            self.tof_feat_dim = 96
            self.stem_main = make_stem(self.main_in_channels, self.main_feat_dim)
            tof_in = 1 + (1 if self.use_tof_conf else 0)
            self.stem_tof = make_stem(tof_in, self.tof_feat_dim)
            self.fuse = nn.Linear(self.main_feat_dim + self.tof_feat_dim, 192, bias=False)
            self.fuse.weight.data.mul_(0.5)
        elif self.sensor_mode in ('passive_depth', 'camera_luma', 'active_depth'):
            stem_in_channels = in_channels
            if self.sensor_mode == 'active_depth':
                stem_in_channels = 1 + (1 if self.use_tof_conf else 0)
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
        if self.camera_action_mode == 'incremental':
            # 统一控制模式：动作维度 = 飞行控制维度 + 3维相机增量控制(FOV, Exposure, ISO)
            total_action_dim = dim_action + 3
            self.fc = nn.Linear(192, total_action_dim, bias=False)
            self.fc.weight.data.mul_(0.01) # 权重初始化极小，防止初始动作过大导致无人机直接坠毁
            self._flight_dim = dim_action  # 记录飞行控制占用的维度索引
        else:
            # 普通模式：只输出飞行控制动作
            self.fc = nn.Linear(192, dim_action, bias=False)
            self.fc.weight.data.mul_(0.01)
            self._flight_dim = dim_action

        # 可选的意图输出头（用于意图域训练 + dLQR）
        if self.use_policy_intent:
            self.fc_intent = nn.Linear(192, intent_dim, bias=True)
            self.fc_intent.weight.data.mul_(0.01)
            self.fc_intent.bias.data.zero_()

        # 传统的独立可微相机头（如果启用且未启用统一控制）
        if self.camera_action_mode == 'absolute':
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
            'passive_depth',
            'camera_luma',
            'camera_luma_plus_passive_depth',
            'active_depth',
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

    def _active_depth_pipeline(self, depth_like: torch.Tensor):
        """
        active_depth 输入流水线：
        1) 深度反转并归一化（近处值更大）
        2) 最近邻下采样到 (2*H_nn, 2*W_nn)
        3) 2x2 最大池化到 (H_nn, W_nn)
        """
        # 支持 (B,H,W) 或 (B,1,H,W)
        d = depth_like
        if d.dim() == 4 and d.shape[1] == 1:
            d = d[:, 0]
        elif d.dim() != 3:
            raise ValueError(f'active_depth pipeline 期望输入为 (B,H,W) 或 (B,1,H,W)，实际: {tuple(d.shape)}')

        # 反转 + 归一化：把 0.05~24m 映射到 [0,1]，近处更亮
        d = d.clamp(0.05, 24.0)
        inv = 1.0 / d
        inv_min = 1.0 / 24.0
        inv_max = 1.0 / 0.05
        x = (inv - inv_min) / (inv_max - inv_min)
        x = x.clamp(0.0, 1.0)

        # 先最近邻到 2x 目标尺寸，再 max-pooling 到最终输入尺寸
        h2 = max(self.tof_nn_height * 2, 2)
        w2 = max(self.tof_nn_width * 2, 2)
        x = torch.nn.functional.interpolate(
            x[:, None],
            size=(h2, w2),
            mode='nearest',
        )
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def preprocess_sensor_inputs(self, main_obs=None, tof_depth=None, tof_conf=None,
                                 add_noise=False):
        """
        将“原始传感器观测”转换成模型可直接消费的张量。

        这一步做两类事情：
        1) 数值域映射（例如把深度映射到更适合网络学习的范围）
        2) 形状整理（确保是 B,C,H,W）

        返回:
            x_fused: 单张量融合输入（兼容单输入路径）
            x_main:  主传感器分支输入（给 dual-encoder 的 main stem）
            x_tof_pack: ToF 分支输入（ToF depth + 可选 conf 拼接）
        """
        x_main = None
        x_tof = None

        if self.sensor_mode in ('passive_depth', 'camera_luma', 'camera_luma_plus_passive_depth'):
            if main_obs is None:
                raise ValueError(f"sensor_mode={self.sensor_mode} 需要 main_obs 输入")
            if self.sensor_mode == 'passive_depth':
                x_main = 3 / main_obs.clamp(0.3, 24) - 0.6
                if add_noise:
                    x_main = x_main + torch.randn_like(x_main) * 0.02
            else:
                x_main = main_obs.clamp(0.0, 1.0)
                if add_noise:
                    x_main = (x_main + torch.randn_like(x_main) * 0.01).clamp(0.0, 1.0)
                x_main = x_main * 2.0 - 1.0
            x_main = self._as_bchw(x_main)
        elif self.sensor_mode == 'active_depth':
            pass
        else:
            raise ValueError(f"unsupported sensor_mode: {self.sensor_mode}")

        if self.sensor_mode in ('camera_luma_plus_passive_depth', 'active_depth'):
            if tof_depth is None:
                raise ValueError(f"sensor_mode={self.sensor_mode} 需要 tof_depth 输入")
            if self.sensor_mode == 'active_depth':
                if self.active_depth_use_pipeline:
                    x_tof = self._active_depth_pipeline(tof_depth)
                    if add_noise:
                        x_tof = (x_tof + torch.randn_like(x_tof) * 0.01).clamp(0.0, 1.0)
                    x_tof = x_tof * 2.0 - 1.0
                else:
                    # active_tof（直输模式）：回退到原逻辑，不做分辨率流水线处理
                    x_tof = 3 / tof_depth.clamp(0.05, 24) - 0.6
                    if add_noise:
                        x_tof = x_tof + torch.randn_like(x_tof) * 0.01
                    x_tof = self._as_bchw(x_tof)
            else:
                # x_tof = 3 / tof_depth.clamp(0.05, 24) - 0.6

                # 依然使用倒数深度保留对近处物体的敏感性
                inv_depth = 3 / tof_depth.clamp(0.05, 24) - 0.6

                # 【修改点1】：使用 Tanh 进行软截断，将无穷大的突变平滑压缩到 [-1.0, 1.0] 附近
                # 除以 3.0 是为了让 1 米 (算出来是 2.4) 的距离刚好在 tanh(0.8) 左右，处于敏感区
                x_tof = torch.tanh(inv_depth / 3.0)
                if add_noise:
                    x_tof = x_tof + torch.randn_like(x_tof) * 0.01
                x_tof = self._as_bchw(x_tof)
        else:
            if tof_depth is not None or tof_conf is not None:
                raise ValueError(f"sensor_mode={self.sensor_mode} 不应提供 ToF 输入")

        x_tof_pack = x_tof
        if self.sensor_mode in ('camera_luma_plus_passive_depth', 'active_depth') and self.use_tof_conf and tof_conf is not None:
            c = tof_conf
            if self.sensor_mode == 'active_depth':
                if self.active_depth_use_pipeline:
                    c = self._active_depth_pipeline(c)
                    if add_noise:
                        c = (c + torch.randn_like(c) * 0.01).clamp(0.0, 1.0)
                    c = c * 2.0 - 1.0
                else:
                    if add_noise:
                        c = (c + torch.randn_like(c) * 0.01).clamp(0.0, 1.0)
                    c = self._as_bchw(c * 2.0 - 1.0)
            else:
                if add_noise:
                    c = (c + torch.randn_like(c) * 0.01).clamp(0.0, 1.0)
                c = self._as_bchw(c * 2.0 - 1.0)
            assert c is not None
            if x_tof_pack is not None and (x_tof_pack.shape[-2:] != c.shape[-2:]):
                raise ValueError('tof_depth 与 tof_conf 尺寸必须一致')
            x_tof_pack = c if x_tof_pack is None else torch.cat([x_tof_pack, c], 1)

        if self.sensor_mode in ('camera_luma_plus_passive_depth', 'active_depth') and self.use_tof_conf and tof_conf is None:
            raise ValueError(f"sensor_mode={self.sensor_mode} 且 use_tof_conf=True 时必须提供 tof_conf")

        channels = []
        if x_main is not None:
            channels.append(x_main)
        if x_tof_pack is not None:
            channels.append(x_tof_pack)
        if len(channels) == 0:
            raise ValueError("preprocess_sensor_inputs 需要至少一种传感器输入")

        if self.sensor_mode == 'camera_luma_plus_passive_depth':
            x_fused = x_main
        else:
            x_fused = torch.cat(channels, 1)
        return x_fused, x_main, x_tof_pack

    def forward(self, v, hx=None, return_intent=False,
                main_obs=None, tof_depth=None, tof_conf=None,
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
        x, x_main, x_tof = self.preprocess_sensor_inputs(
            main_obs=main_obs,
            tof_depth=tof_depth,
            tof_conf=tof_conf,
            add_noise=add_noise,
        )

        # ==========================
        # B. 视觉特征提取
        # ==========================
        if self.sensor_mode == 'camera_luma_plus_passive_depth':
            if x_main is None or x_tof is None:
                raise ValueError('sensor_mode=camera_luma_plus_passive_depth 需要同时提供 main 与 tof 输入')
            feat_main = self.stem_main(x_main)
            feat_tof = self.stem_tof(x_tof)
            img_feat = self.fuse(torch.cat([feat_main, feat_tof], -1))
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

        if self.camera_action_mode == 'incremental':
            flight_act = raw[:, :self._flight_dim]
            cam_deltas = torch.tanh(raw[:, self._flight_dim:])  # (Batch, 3)
            if return_intent and self.use_policy_intent:
                # 可选意图头（给 dLQR/dMPC 使用）
                intent_raw = self.fc_intent(self.act(hx))
                return flight_act, cam_deltas, hx, intent_raw
            return flight_act, cam_deltas, hx

        act = raw
        cam_params = None
        if self.camera_action_mode == 'absolute':
            cam_raw = self.fc_cam(self.act(hx))
            cam_params = torch.sigmoid(cam_raw)  # (Batch, 3)
        if return_intent and self.use_policy_intent:
            intent_raw = self.fc_intent(self.act(hx))
            return act, cam_params, hx, intent_raw
        return act, cam_params, hx

if __name__ == '__main__':
    # 简单的测试代码，确保模型可以被实例化
    Model()