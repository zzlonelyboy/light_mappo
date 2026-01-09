import copy
import torch.nn as nn
import math
import torch
import numpy as np
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def get_shape_from_obs_space(obs_space):
    """
    从 obs_space 中获取观测输入的形状 (tuple)。
    支持 Box (连续/图像), Discrete (离散), 以及 list 列表形式。
    """
    # 如果传入的是列表（多智能体场景有时会把所有 agent 的 space 放在列表里）
    if isinstance(obs_space, list):
        obs_space = obs_space[0]

    # Gym Box (连续空间 或 图像)
    if hasattr(obs_space, 'shape'):
        return obs_space.shape

    # Gym Discrete (离散空间)
    # 神经网络输入维度通常是 one-hot 后的长度，即 n
    if hasattr(obs_space, 'n'):
        return (obs_space.n,)

    raise NotImplementedError(f"Unsupported observation space type: {type(obs_space)}")


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """
    线性衰减学习率。
    lr = init_lr * (1 - current_epoch / total_epochs)
    """
    # 防止除以0
    if total_num_epochs == 0:
        return

    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))

    # 确保 lr 不小于 0
    lr = max(0.0, lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# =============================================================================
# 以下是你代码 Trainer 中引用的其他辅助函数，建议一并补充，防止报错
# =============================================================================

def get_gard_norm(it):
    """
    计算梯度的 L2 范数 (用于 log 打印)
    :param it: parameters 迭代器 (model.parameters())
    """
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    """
    Huber Loss 实现
    :param e: error (diff)
    :param d: delta 阈值
    """
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    """
    MSE Loss 实现
    :param e: error
    """
    return e ** 2 / 2


def check(input):
    """
    将 numpy 数组转换为 tensor，用于 algorithms.utils.util 中的引用
    """
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
    return input

