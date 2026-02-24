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
                 use_unified_control=False, use_cam_obs=False) -> None:
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

        # 如果启用了相机状态观测，物理观测维度需要增加 4 维（FOV, 曝光, ISO, 对焦）
        actual_obs_dim = dim_obs + (4 if use_cam_obs else 0)

        # 视觉特征提取主干网络 (CNN Stem)
        # 输入是深度图 (通常是 1 通道, 经过 max_pool 降采样后的低分辨率图像)
        self.stem = nn.Sequential(
            # 第一层卷积：输入通道1，输出通道32，卷积核2x2，步长2。起到降采样作用。
            nn.Conv2d(1, 32, 2, 2, bias=False),  
            nn.LeakyReLU(0.05), # 使用 LeakyReLU 激活函数，负半轴斜率 0.05
            # 第二层卷积：输入32，输出64，卷积核3x3
            nn.Conv2d(32, 64, 3, bias=False), 
            nn.LeakyReLU(0.05),
            # 第三层卷积：输入64，输出128，卷积核3x3
            nn.Conv2d(64, 128, 3, bias=False), 
            nn.LeakyReLU(0.05),
            nn.Flatten(), # 将多维的特征图展平为一维向量
            # 线性层：将展平后的特征映射到 192 维的隐层空间
            nn.Linear(128*2*4, 192, bias=False),
        )

        # 状态向量投影层：将无人机的物理状态（速度、姿态、目标距离等）映射到 192 维
        self.v_proj = nn.Linear(actual_obs_dim, 192)
        # 初始化权重，乘以 0.5 使其初始输出较小，防止物理状态特征在初期淹没视觉特征
        self.v_proj.weight.data.mul_(0.5)

        # 门控循环单元 (GRU)：用于处理时序信息，赋予无人机记忆能力
        # 因为无人机只能看到当前的深度图（部分可观测环境 POMDP），需要记忆来推断自身速度和环境结构
        self.gru = nn.GRUCell(192, 192)

        # 动作输出头 (Action Head)
        if use_unified_control:
            # 统一控制模式：动作维度 = 飞行控制维度 + 4维相机增量控制
            total_action_dim = dim_action + 4
            self.fc = nn.Linear(192, total_action_dim, bias=False)
            self.fc.weight.data.mul_(0.01) # 权重初始化极小，防止初始动作过大导致无人机直接坠毁
            self._flight_dim = dim_action  # 记录飞行控制占用的维度索引
        else:
            # 普通模式：只输出飞行控制动作
            self.fc = nn.Linear(192, dim_action, bias=False)
            self.fc.weight.data.mul_(0.01)
            self._flight_dim = dim_action

        # 传统的独立可微相机头（如果启用且未启用统一控制）
        if use_diff_cam and not use_unified_control:
            # 输出 4 个相机参数的绝对值
            self.fc_cam = nn.Linear(192, 4, bias=True)
            self.fc_cam.weight.data.mul_(0.01)
            # 初始化偏置为 0，这样经过后续的 sigmoid 激活后，初始输出在 0.5 附近（即默认的居中参数）
            self.fc_cam.bias.data.zero_()

        # 全局使用的激活函数
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        # 重置函数，当前为空。
        # 因为 GRU 的隐藏状态 hx 是在外部循环 (main_cuda.py) 中手动维护并传入的，而不是存在模型内部。
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
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
        # 1. 提取视觉特征 (CNN)
        img_feat = self.stem(x)
        
        # 2. 将视觉特征与物理状态特征相加，并激活。这是多模态融合的一步。
        x = self.act(img_feat + self.v_proj(v))
        
        # 3. 更新 GRU 隐藏状态（结合当前融合特征和历史记忆）
        hx = self.gru(x, hx)
        
        # 4. 通过全连接层输出原始动作向量
        raw = self.fc(self.act(hx))

        if self.use_unified_control:
            # 如果是统一控制模式，将输出拆分为飞行控制和相机增量控制
            flight_act = raw[:, :self._flight_dim]
            # 相机增量使用 tanh 激活，将其限制在 [-1, 1] 范围内，表示正向或负向的微调 (Delta)
            cam_deltas = torch.tanh(raw[:, self._flight_dim:])  # (Batch, 4)
            return flight_act, cam_deltas, hx

        # 传统模式路径
        act = raw
        cam_params = None
        if self.use_diff_cam:
            # 如果启用了传统可微相机，通过独立的头输出，并使用 sigmoid 限制在 [0, 1] 范围内（绝对值）
            cam_raw = self.fc_cam(self.act(hx))
            cam_params = torch.sigmoid(cam_raw)  # (Batch, 4)

        return act, cam_params, hx

if __name__ == '__main__':
    # 简单的测试代码，确保模型可以被实例化
    Model()