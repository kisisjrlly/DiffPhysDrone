import torch


class GDecay(torch.autograd.Function):
    """
    梯度衰减函数 (Gradient Decay)。
    在强化学习/轨迹优化中，长序列的反向传播容易导致梯度爆炸。
    这个函数在前向传播时保持值不变，但在反向传播时，将梯度乘以一个衰减系数 alpha。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


g_decay = GDecay.apply
