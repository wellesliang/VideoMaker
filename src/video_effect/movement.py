# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
get commedity information, such as catogory, name, image, ...

Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/01 09:01:01
"""

import math
import numpy as np


def linear(x1, y1, x2, y2, T):
    """
    线性移动
    :param x1:起点坐标
    :param y1:
    :param x2:终点坐标
    :param y2:
    :param T:总时间
    :return:
    """
    rate = lambda r: 1.0 if r >= 1.0 else r
    return lambda t: ((x2 - x1) * rate(t / T) + x1, (y2 - y1) * rate(t / T) + y1)


def decel(x1, y1, x2, y2, T, str='pow'):
    """
    减速运动，
    :param x1:起点坐标
    :param y1:
    :param x2:终点坐标
    :param y2:
    :param T:总时间
    :param str: 减速函数：circle: 圆形衰减，sin: 正弦衰减，default：指数衰减
    :return:
    """
    if str == 'circle':
        rate = lambda r: 1.0 if r >= 1.0 else math.sqrt(1 - (r - 1) * (r - 1))
    elif str == 'sin':
        rate = lambda r: 1.0 if r >= 1.0 else math.sin(r * math.pi / 2)
    else:
        rate = lambda r: 1.0 if r >= 1.0 else 1.0 - (1.0 - r) ** 6
    return lambda t: ((x2 - x1) * rate(t / T) + x1, (y2 - y1) * rate(t / T) + y1)


def accel(x1, y1, x2, y2, T, str='pow'):
    """
    加速运动
    :param x1:起点坐标
    :param y1:
    :param x2:终点坐标
    :param y2:
    :param T:
    :param str:加速策略 circle: 圆形衰减，sin: 正弦衰减，default：指数衰减
    :return:
    """
    if str == 'circle':
        rate = lambda r: 1.0 if r >= 1.0 else 1 - math.sqrt(1 - r * r)
    elif str == 'sin':
        rate = lambda r: 1.0 if r >= 1.0 else math.sin(r * math.pi / 2 + math.pi * 1.5) + 1.0
    else:
        rate = lambda r: 1.0 if r >= 1.0 else r ** 6
    return lambda t: ((x2 - x1) * rate(t / T) + x1, (y2 - y1) * rate(t / T) + y1)


rotMatrix = lambda a: np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def vortex_in(x2, y2, i, n):
    """
    从屏幕外漩涡轨迹飞入x2 y2位置
    :param x2: 目标点位置
    :param y2:
    :param i: i/n 控制了飞入角度， 0 <= i < n
    :param n:
    :return:
    """
    d = lambda t: 1.0 / (0.3 + t ** 6)  # damping
    a = i * np.pi / n  # angle of the movement
    v = rotMatrix(a).dot([-1, 0])
    if i % 2:
        v[1] = -v[1]
    return lambda t: (x2, y2) + 600 * d(t) * rotMatrix(0.5 * d(t) * a).dot(v)


def cascade(x1, y1, x2, y2, T):
    """
    模拟物品坠落的运动轨迹，坠落后有小幅反弹，想象皮球落地的情况，但实际弹性远低于皮球。可观察的幅度大概弹跳3次
    :param x1:起点坐标
    :param y1:
    :param x2:终点坐标
    :param y2:
    :param T: 时间
    :return:
    """
    # v = np.array([0, -1])
    d = lambda t: 1 if t < 0 else abs(np.sinc(t / T) / (1 + (t / T) ** 3))
    return lambda t: (x2 + (x1 - x2) * d(t), y2 + (y1 - y2) * d(t))
