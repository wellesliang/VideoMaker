# -*- coding: utf-8 -*-
"""
Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/08/15 09:24:01
"""
import math


def distance_pt_seg_3d(pt, p, q, ret_pow=False):
    """
    计算3D空间中点到线段的距离，参考：https://blog.csdn.net/u012138730/article/details/79779996
    :param pt:  point
    :param p: 线段端点之一
    :param q: 线段端点之一
    :return:
    """
    pqx = q[0] - p[0]
    pqy = q[1] - p[1]
    pqz = q[2] - p[2]
    dx = pt[0] - p[0]
    dy = pt[1] - p[1]
    dz = pt[2] - p[2]
    d = pqx ** 2 + pqy ** 2 + pqz ** 2
    t = pqx * dx + pqy * dy + pqz * dz
    if d > 0:
        t = t / d
    if t < 0:
        t = 0
    elif t > 1:
        t = 1

    dx = p[0] + t * pqx - pt[0]
    dy = p[1] + t * pqy - pt[1]
    dz = p[2] + t * pqz - pt[2]
    if ret_pow:
        return dx ** 2 + dy ** 2 + dz ** 2
    else:
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def distance_pt_seg(pt, p, q, ret_pow=False):
    """
    计算2D空间中点到线段的距离，参考：https://blog.csdn.net/u012138730/article/details/79779996
    :param pt:  point
    :param p: 线段端点之一
    :param q: 线段端点之一
    :return:
    """
    pqx = float(q[0]) - p[0]
    pqy = float(q[1]) - p[1]
    dx = float(pt[0]) - p[0]
    dy = float(pt[1]) - p[1]
    d = pqx ** 2 + pqy ** 2
    t = pqx * dx + pqy * dy
    if d > 0:
        t = t / d
    if t < 0:
        t = 0.0
    elif t > 1:
        t = 1.0

    dx = p[0] + t * pqx - pt[0]
    dy = p[1] + t * pqy - pt[1]

    if ret_pow:
        return dx ** 2 + dy ** 2
    else:
        return math.sqrt(dx ** 2 + dy ** 2)


if __name__ == '__main__':
    pt = (0, 5)
    p = (0, 0)
    q = (10, 10)
    print(distance_pt_seg(pt, p, q))

