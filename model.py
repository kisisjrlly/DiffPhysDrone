import torch
from torch import nn

# 这是一个自定义的梯度衰减函数。
# 它的作用是在反向传播时，将梯度乘以一个衰减系数 alpha。
# 前向传播时，它的值就是 x 本身 (x * alpha + x * (1 - alpha) = x)。
# 这在长序列的 RNN/GRU 训练中常用于缓解梯度爆炸，或者在 G-DAC 算法中控制梯度流。
def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4, use_diff_cam=False,
                 use_unified_control=False, use_cam_obs=False,
                 in_channels=1, use_policy_intent=False, intent_dim=9,
                 main_in_channels=1,
                 use_tof_conf=False,
                 vision_mode='depth') -> None:
        """
        初始化无人机的策略网络模型 (Policy Network)。
        Args:
            dim_obs: 基础物理观测维度（通常是7维无里程计，或10维带里程计）。
            dim_action: 飞行控制动作维度（默认6维：3维加速度 + 3维速度预测）。
            use_diff_cam: 是否使用传统的独立可微相机头（输出绝对的 sigmoid 参数）。
            use_unified_control: 论文 §2.1 提出的统一控制空间。相机增量作为动作输出的一部分。
            use_cam_obs: 论文 §2.1 提出的将当前相机状态加入到观测向量中。
        """
        super().__init__()
        # 保存配置标志，用于在 forward 中决定走哪条计算路径
        self.use_diff_cam = use_diff_cam
        self.use_unified_control = use_unified_control
        self.use_cam_obs = use_cam_obs
        self.use_policy_intent = use_policy_intent
        self.intent_dim = intent_dim
        self.main_in_channels = main_in_channels
        self.use_tof_conf = use_tof_conf
        self.vision_mode = vision_mode

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
        actual_obs_dim = dim_obs + (3 if use_cam_obs else 0)

        # 视觉特征提取主干网络 (CNN Stem)
        # 按 vision_mode 固定网络结构：
        #   depth/yuv: 单分支
        #   yuv_tof  : 双分支 + 特征融合
        if self.vision_mode == 'yuv_tof':
            self.main_feat_dim = 96
            self.tof_feat_dim = 96
            self.stem_main = make_stem(self.main_in_channels, self.main_feat_dim)
            tof_in = 1 + (1 if self.use_tof_conf else 0)
            self.stem_tof = make_stem(tof_in, self.tof_feat_dim)
            self.fuse = nn.Linear(self.main_feat_dim + self.tof_feat_dim, 192, bias=False)
            self.fuse.weight.data.mul_(0.5)
        elif self.vision_mode in ('depth', 'yuv'):
            self.stem = make_stem(in_channels, 192)
        else:
            raise ValueError(f'unsupported vision_mode: {self.vision_mode}')

        # 状态向量投影层：将无人机的物理状态（速度、姿态、目标距离等）映射到 192 维
        self.v_proj = nn.Linear(actual_obs_dim, 192)
        # 初始化权重，乘以 0.5 使其初始输出较小，防止物理状态特征在初期淹没视觉特征
        self.v_proj.weight.data.mul_(0.5)

        # 门控循环单元 (GRU)：用于处理时序信息，赋予无人机记忆能力
        # 因为无人机只能看到当前的深度图（部分可观测环境 POMDP），需要记忆来推断自身速度和环境结构
        self.gru = nn.GRUCell(192, 192)

        # 动作输出头 (Action Head)
        if use_unified_control:
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
        if use_diff_cam and not use_unified_control:
            # 输出 3 个相机参数的绝对值: FOV, Exposure, ISO
            self.fc_cam = nn.Linear(192, 3, bias=True)
            self.fc_cam.weight.data.mul_(0.01)
            # 初始化偏置为 0，这样经过后续的 sigmoid 激活后，初始输出在 0.5 附近（即默认的居中参数）
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

    @staticmethod
    def _finite(x: torch.Tensor, nan=0.0, pos=1.0, neg=-1.0):
        return torch.nan_to_num(x, nan=nan, posinf=pos, neginf=neg)

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

        if self.vision_mode in ('depth', 'yuv', 'yuv_tof'):
            if main_obs is None:
                raise ValueError(f"vision_mode={self.vision_mode} 需要 main_obs 输入")
            if self.vision_mode == 'depth':
                # depth 模式：把深度值映射到一个更紧凑的数值范围，增强近距离分辨能力
                x_main = 3 / main_obs.clamp(0.3, 24) - 0.6
                if add_noise:
                    # 训练时可注入小噪声，提升鲁棒性
                    x_main = x_main + torch.randn_like(x_main) * 0.02
            else:
                # yuv / y 模式：主输入被视为亮度图，先裁剪到 [0,1]
                x_main = main_obs.clamp(0.0, 1.0)
                if add_noise:
                    x_main = (x_main + torch.randn_like(x_main) * 0.01).clamp(0.0, 1.0)
                # 再映射到 [-1, 1]，更符合后续网络初始化分布
                x_main = x_main * 2.0 - 1.0
            # 统一为 B,C,H,W
            x_main = self._as_bchw(x_main)
        else:
            raise ValueError(f"unsupported vision_mode: {self.vision_mode}")

        if self.vision_mode == 'yuv_tof':
            if tof_depth is None:
                raise ValueError("vision_mode=yuv_tof 需要 tof_depth 输入")
            # ToF 深度同样做非线性映射，突出近场几何变化
            x_tof = 3 / tof_depth.clamp(0.05, 24) - 0.6
            if add_noise:
                x_tof = x_tof + torch.randn_like(x_tof) * 0.01
            x_tof = self._as_bchw(x_tof)
        else:
            # 非 yuv_tof 模式不允许 ToF 输入，避免模式语义漂移
            if tof_depth is not None or tof_conf is not None:
                raise ValueError(f"vision_mode={self.vision_mode} 不应提供 ToF 输入")

        x_tof_pack = x_tof
        if self.vision_mode == 'yuv_tof' and self.use_tof_conf and tof_conf is not None:
            # 可选 ToF 置信度分支：映射到 [-1,1] 后与 ToF depth 在通道维拼接
            c = tof_conf
            if add_noise:
                c = (c + torch.randn_like(c) * 0.01).clamp(0.0, 1.0)
            c = self._as_bchw(c * 2.0 - 1.0)
            if x_tof_pack is not None and (x_tof_pack.shape[-2:] != c.shape[-2:]):
                raise ValueError('tof_depth 与 tof_conf 尺寸必须一致')
            x_tof_pack = c if x_tof_pack is None else torch.cat([x_tof_pack, c], 1)

        if self.vision_mode == 'yuv_tof' and self.use_tof_conf and tof_conf is None:
            raise ValueError("vision_mode=yuv_tof 且 use_tof_conf=True 时必须提供 tof_conf")

        channels = []
        if x_main is not None:
            channels.append(x_main)
        if x_tof_pack is not None:
            channels.append(x_tof_pack)
        if len(channels) == 0:
            raise ValueError("preprocess_sensor_inputs 需要至少一种传感器输入")

        # depth/yuv 为单分支输入，可直接使用通道拼接结果（通常只有1路）
        # yuv_tof 模式下主/ToF 分辨率允许不同，不能强制拼接；
        # 这里返回 x_main 作为占位 fused 输入（forward 在该模式不会消费 x_fused）
        if self.vision_mode == 'yuv_tof':
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

        if x is not None:
            x = self._finite(x, nan=0.0, pos=5.0, neg=-5.0)
        if x_main is not None:
            x_main = self._finite(x_main, nan=0.0, pos=5.0, neg=-5.0)
        if x_tof is not None:
            x_tof = self._finite(x_tof, nan=0.0, pos=5.0, neg=-5.0)
        v = self._finite(v, nan=0.0, pos=50.0, neg=-50.0)
        if hx is not None:
            hx = self._finite(hx, nan=0.0, pos=10.0, neg=-10.0)

        # ==========================
        # B. 视觉特征提取
        # ==========================
        if self.vision_mode == 'yuv_tof':
            # 固定双输入结构：main/tof 两路特征后融合
            if x_main is None or x_tof is None:
                raise ValueError('vision_mode=yuv_tof 需要同时提供 main 与 tof 输入')
            feat_main = self.stem_main(x_main)
            feat_tof = self.stem_tof(x_tof)
            img_feat = self.fuse(torch.cat([feat_main, feat_tof], -1))
        else:
            # depth/yuv 固定单输入结构
            img_feat = self.stem(x)
        img_feat = self._finite(img_feat, nan=0.0, pos=10.0, neg=-10.0)
        
        # ==========================
        # C. 多模态融合 + 时序建模
        # ==========================
        # 视觉特征 + 状态向量投影后做非线性激活
        x = self.act(img_feat + self.v_proj(v))
        x = self._finite(x, nan=0.0, pos=10.0, neg=-10.0)
        
        # GRU 累积时序上下文（POMDP 下尤为关键）
        hx = self.gru(x, hx)
        hx = self._finite(hx, nan=0.0, pos=10.0, neg=-10.0)
        
        # 输出动作头原始值
        raw = self.fc(self.act(hx))
        raw = self._finite(raw, nan=0.0, pos=10.0, neg=-10.0)

        if self.use_unified_control:
            # 统一控制：输出 = 飞行动作 + 相机增量动作
            flight_act = raw[:, :self._flight_dim]
            # 相机增量限制到 [-1,1]
            cam_deltas = torch.tanh(raw[:, self._flight_dim:])  # (Batch, 3)
            cam_deltas = self._finite(cam_deltas, nan=0.0, pos=1.0, neg=-1.0)
            if return_intent and self.use_policy_intent:
                # 可选意图头（给 dLQR/dMPC 使用）
                intent_raw = self.fc_intent(self.act(hx))
                intent_raw = self._finite(intent_raw, nan=0.0, pos=10.0, neg=-10.0)
                return flight_act, cam_deltas, hx, intent_raw
            return flight_act, cam_deltas, hx

        # 传统模式：只输出飞行动作；若启用 diff_cam 再给一组绝对相机参数
        act = raw
        cam_params = None
        if self.use_diff_cam:
            # 独立相机头输出绝对值参数，范围 [0,1]
            cam_raw = self.fc_cam(self.act(hx))
            cam_raw = self._finite(cam_raw, nan=0.0, pos=10.0, neg=-10.0)
            cam_params = torch.sigmoid(cam_raw)  # (Batch, 3)
        if return_intent and self.use_policy_intent:
            intent_raw = self.fc_intent(self.act(hx))
            intent_raw = self._finite(intent_raw, nan=0.0, pos=10.0, neg=-10.0)
            return act, cam_params, hx, intent_raw
        return act, cam_params, hx

if __name__ == '__main__':
    # 简单的测试代码，确保模型可以被实例化
    Model()