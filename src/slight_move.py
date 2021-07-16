# -*- coding: utf-8 -*-

"""
微动视频制作

Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/01 09:10:06
"""


import os
import sys
import numpy as np
import math
import time
import cv2 as cv
import pickle
import moviepy.editor as mpy
from tools.flow import Flow


def get_mask_rect(w, h):
    """
    return a mask with gauss blur
    """
    margin = max(12, (w + h) / 6)
    nw = w + margin * 2
    nh = h + margin * 2
    m = np.ndarray(shape = (nh, nw), dtype = 'uint8')
    m[0:, 0:] = 255
    m[margin: -margin, margin: -margin] = 0
    ksize = margin / 2 * 2 + 1
    m = cv.GaussianBlur(m, (ksize, ksize), 0) / 255.0
    mask = cv.merge((m, m, m))
    return mask


def get_mask_circle(r):
    """
    :param r: 半径
    :return:
    """
    margin = max(12, r / 3)
    w, h = 2 * (r + margin), 2 * (r + margin)
    m = np.ndarray(shape = (h, w), dtype = 'uint8')
    m[0:, 0:] = 255
    cv.circle(m, (h / 2, w / 2), r, 0, -1)
    ksize = margin / 2 * 2 + 1
    m = cv.boxFilter(m, -1, (ksize, ksize)) / 255.0
    mask = cv.merge((m, m, m))
    return mask


def bilinear_interpolation(im, x, y):
    """
    :param im:
    :param x:
    :param y:
    :return:
    """
    x = max(0, x)
    y = max(0, y)
    x = min(im.shape[1] - 2, x)
    y = min(im.shape[0] - 2, y)

    u = x - int(x)
    v = y - int(y)
    x = int(x)
    y = int(y)
    c = (1 - v) * (1 - u) * im[y, x] + \
        (1 - v) * u * im[y, x + 1] + \
        v * (1 - u) * im[y + 1, x] + \
        v * u * im[y + 1, x + 1]
    return c


def slight_move_demo(clip):
    """
    低效demo版本，但方便理解原理
    :return:
    """
    T = 2.0
    LEN = 25
    def mfl(gf, t):
        """
        低效版本，但方便理解原理
        :return:
        """
        t0 = time.time()
        r = t / T - int(t / T)
        im = gf(t)
        im2 = np.copy(im)
        mask = np.zeros(shape=im.shape[0:2], dtype='uint8')

        for x1 in range(200, 400, 1):
            for y1 in range(70, 200, 1):
                t1 = time.time()
                x0 = x1 + r * LEN
                #y0 = y1 + r * LEN
                y0 = y1

                x0 = max(0, x0)
                y0 = max(0, y0)
                x0 = min(im.shape[1] - 2, x0)
                y0 = min(im.shape[0] - 2, y0)

                u = x0 - int(x0)
                v = y0 - int(y0)
                x = int(x0)
                y = int(y0)
                c = (1 - v) * (1 - u) * im[y, x] + \
                    (1 - v) * u * im[y, x + 1] + \
                    v * (1 - u) * im[y + 1, x] + \
                    v * u * im[y + 1, x + 1]
                im2[y1, x1] = c
                mask[y1, x1] = 255

        mask = cv.boxFilter(mask, -1, (7, 7))
        mask = mask / 255.0
        if r < 0.3:
            mask = mask * r / 0.3
        elif r > 0.7:
            mask = mask * (1 - r) / 0.3
        mask = cv.merge((mask, mask, mask))
        im2 = im * (1 - mask) + im2 * mask
        print('all', time.time() - t0, r)
        return im2
    return clip.fl(mfl)


def calc_flow(im, mxi, myi, mxd, myd, width, height, r):
    """
    所有输入矩阵已经reshape到一维了
    :param im: 原始图像
    :param mxi: x轴坐标矩阵
    :param myi: y轴坐标矩阵
    :param mxd: x轴方向矩阵
    :param myd: y轴方向矩阵
    :param width: 宽
    :param height: 高
    :param r: 移动系数，0-1的百分比
    :return: 新图像
    """
    if r <= 0.0001:
        return im.reshape((height, width, 3))

    mx0 = mxi - mxd * r
    mx0 = np.minimum(width - 2, mx0)
    mx = mx0.astype('int')
    mu = mx0 - mx
    mu = np.concatenate([mu[:, np.newaxis], mu[:, np.newaxis], mu[:, np.newaxis]], axis=1)

    my0 = myi - myd * r
    my0 = np.minimum(height - 2, my0)
    my = my0.astype('int')
    mv = my0 - my
    mv = np.concatenate([mv[:, np.newaxis], mv[:, np.newaxis], mv[:, np.newaxis]], axis=1)

    im2 = (1 - mv) * (1 - mu) * im[my * width + mx] + \
          mv * (1 - mu) * im[(my + 1) * width + mx] + \
          (1 - mv) * mu * im[my * width + mx + 1] + \
          mv * mu * im[(my + 1) * width + mx + 1]
    im2 = im2.astype('uint8').reshape((height, width, 3))
    return im2


def slight_move_matrix(clip, mxd, myd, T_a, T_b):
    """
    高效实现版本
    核心思路为利用坐标矩阵和移动向量矩阵计算出新的坐标矩阵。
    再用新坐标矩阵为下标，得到新图像像素。在其基础上，用各种差值算法优化图像。
    :param clip:
    :param mxd: x轴的移动向量矩阵
    :param myd: y轴的移动向量矩阵
    :param T_a: 总的循环周期时长
    :param T_b: 两个循环周期重叠的时长
    :return:
    """
    width = mxd.shape[1]
    height = mxd.shape[0]

    # mxi: x轴的坐标矩阵，
    mxi = np.ndarray(shape=(height, width), dtype='int16')
    for i in range(width):
        mxi[0:, i] = i
    mxi = mxi.reshape(width * height)
    # myi: y轴的坐标矩阵，
    myi = np.ndarray(shape=(height, width), dtype='int16')
    for i in range(height):
        myi[i, 0:] = i
    myi = myi.reshape(width * height)

    #
    mask0 = np.zeros(shape=(height, width), dtype='float')
    mask0 = np.where(np.abs(mxd) > 0.01, 1.0, mask0)
    mask0 = np.where(np.abs(myd) > 0.01, 1.0, mask0)
    mask0 = cv.boxFilter(mask0, -1, (9, 9)).reshape((height, width, 1))
    mask0 = np.concatenate([mask0, mask0, mask0], axis=2)

    # reshape为一维矩阵，用于图像下标索引
    mxd = mxd.reshape(width * height)
    myd = myd.reshape(width * height)

    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        im = im.reshape((width * height), 3)

        t = t - int(t / (T_a - T_b)) * (T_a - T_b)
        r1, mask_r1 = t / T_a, 1.0
        r2, mask_r2 = 0.0, 0.0
        if t < T_b:
            r2 = (t + T_a - T_b) / T_a
            mask_r1 = t / T_b
            mask_r2 = 1.0 - mask_r1
        # print t, r1, r2, mask_r1, mask_r2
        im1 = calc_flow(im, mxi, myi, mxd, myd, width, height, r1)
        im2 = calc_flow(im, mxi, myi, mxd, myd, width, height, r2)
        im2 = im1 * mask_r1 + im2 * mask_r2

        im = im.reshape(im2.shape)
        im2 = im * (1 - mask0) + im2 * mask0
        return im2

    return clip.fl(mfl)


def get_straight_line_move(x1, y1, x2, y2, tlen):
    """
    return a function desc straight move
    """
    t_rate = lambda t: t / tlen * 1.0 - int(t / tlen * 1.0)
    return lambda t: (int(t_rate(t) * (x2 - x1) + x1), int(t_rate(t) * (y2 - y1) + y1))


def demo():
    """
    =使用微动视频demo，需加载 mx，my两个移动向量矩阵，通过 draw flow绘制获得
    """
    input_path = '../output/weidong/demo1.jpg'
    input_name = input_path.split('/')[-1].split('.')[0]
    output_path = '../output/weidong/' + input_name + '.mp4'
    output_name = output_path.split('/')[-1]
    tmp_name = output_path[0: -4] + '_lop.mp4'

    flow = Flow(input_path)
    mx, my = flow.get_flow_matrix()

    T_a = 3.0  # 总的循环周期秒数
    T_b = 1.2  # 两个循环周期交叠的秒数
    clip0 = mpy.ImageClip(input_path)
    clip0 = clip0.set_duration(T_a - T_b)

    clip0 = clip0.fx(slight_move_matrix, mx, my, T_a, T_b)
    clip0.write_videofile(tmp_name, fps=36)

    clip0 = mpy.VideoFileClip(tmp_name)
    clip0 = mpy.concatenate([clip0, clip0, clip0, clip0, clip0, clip0])
    clip0.write_videofile(output_path)
    os.remove(tmp_name)


if __name__ == '__main__':
    demo()
