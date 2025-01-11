import math

import numpy as np


def epsilon_R_subsampling(alpha, q, epsilon_R):
    if alpha <= 1:
        raise ValueError("alpha must biggger than 1")

    exp_term = np.exp((alpha - 1) * epsilon_R)

    inner = 1 - q + q * exp_term

    epsilon_R_prime_val = (1.0 / (alpha - 1)) * np.log(inner)

    return epsilon_R_prime_val

def rdp_to_epsilon_dp(L, epsilon_R, lambd, m, N, B):
    """
    根据您提供的公式(20)，将给定的RDP参数转换为ε-DP。

    参数：
    L : float
        与算法中灵敏度相关的参数（公式中出现的L）。
    epsilon_R : float
        Rényi差分隐私参数(RDP)中给定的 ε_R。
    lambd : float
        公式中的 λ (lambda) 参数。
    m : float
        批次数量或相关迭代次数参数，视具体应用场景定义。
    N : float
        与数据规模或训练参数相关的数量(来自公式中 (N+1)L + 2|B| 一项)。
    B : float
        |B| 的大小

    返回：
    float
        转换得到的 ε 值，使机制满足 (ε)-DP。
    """
    numerator = L * epsilon_R
    denominator = 2 * (lambd ** 2) * m * ((N + 1) * L + 2 * B)
    epsilon = math.sqrt(numerator / denominator)
    return epsilon







