# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/07/18 11:55:06
一些换场动画的demo展示：http://bos.qasandbox.bcetest.baidu.com/pa-test/ppt_demo1
"""

from moviepy.video.tools import drawing
import math


def block_V_in(clip, duration):
    """
    网格状从上往下换场
    :param clip:
    :param duration:
    :return:
    """
    hnum = 15
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if t >= duration:
            im[0:, 0:] = 1.0
            return im

        wnum = int(hnum * im.shape[1] / im.shape[0] + 0.5)
        hidx = int(hnum * t / duration)
        mrate = (t - hidx * duration * 1.0 / hnum) / (duration * 1.0 / hnum)
        y0 = int(im.shape[0] / hnum * (hidx - 1))
        y1 = int(im.shape[0] / hnum * hidx)
        y2 = int(im.shape[0] / hnum * (hidx + 1))
        im[0:, 0:] = 0.0
        if y0 > 0:
            im[0: y0, 0:] = 1.0
        y0 = max(y0, 0)
        y2 = min(y2, im.shape[0])

        for i in range(wnum):
            x1 = int(max(im.shape[1] / wnum * i, 0))
            x2 = int(min(im.shape[1] / wnum * (i + 1), im.shape[1]))
            if i % 2 == 0:
                im[y0: y1, x1: x2] = mrate
            else:
                im[y0: y1, x1: x2] = 1.0
        for i in range(wnum):
            x1 = int(max(im.shape[1] / wnum * i, 0))
            x2 = int(min(im.shape[1] / wnum * (i + 1), im.shape[1]))
            if i % 2 == 1:
                im[y1: y2, x1: x2] = mrate
            else:
                im[y1: y2, x1: x2] = 0.0
        return im

    return clip.fl(mfl)


def scroll_D_in(clip, duration):
    """
    从左下角往右上角翻页
    :param clip:
    :param duration:
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if t >= duration:
            im[0:, 0:] = 1.0
            return im
        size = (im.shape[1], im.shape[0])
        x1 = t * 1.0 / duration * (im.shape[0] + im.shape[1]) - im.shape[0]
        im2 = drawing.color_split(size, p1=[x1, 0], vector=(1.0, 1.0), grad_width=15)
        return im2
    return clip.fl(mfl)


def strip_V_in(clip, duration):
    """
    垂直方向的，条状渗透
    :param clip:
    :param duration:
    :return:
    """
    wnum = 8
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if t >= duration:
            im[0:, 0:] = 1.0
            return im

        y = int(t * 1.0 / duration * im.shape[0])
        wwidth = im.shape[1] * 1.0 / wnum
        for i in range(wnum):
            x1 = int(i * wwidth + 0.5)
            x2 = int((i + 1) * wwidth + 0.5)
            if i % 2 == 0:
                im[0: y, x1: x2] = 1.0
                im[y:, x1: x2] = 0.0
            else:
                yy = im.shape[0] - y
                im[0: yy, x1: x2] = 0.0
                im[yy:, x1: x2] = 1.0
        return im
    return clip.fl(mfl)


def clock180(clip, duration):
    """
    中心点180度顺时针扫描
    :param clip:
    :param duration:
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if t >= duration:
            im[0:, 0:] = 1.0
            return im

        ang = math.pi * t / duration
        vec = (-math.sin(ang), math.cos(ang))
        xc, yc = (int(im.shape[1] / 2), int(im.shape[0] / 2))
        size = (im.shape[1], im.shape[0])
        im = drawing.color_split(size, p1=(xc, yc), vector=vec, grad_width=10)
        im2 = 1.0 - im
        if ang < math.pi / 2:
            im[0: yc, 0: xc] = 0.0
            im[yc:, xc:] = 0.0
            im[yc:, 0: xc] = im2[yc:, 0: xc]
        else:
            im[0: yc, xc:] = 1.0
            im[yc:, 0: xc] = 1.0
            im[0: yc, 0: xc] = im2[0: yc, 0: xc]
        return im
    return clip.fl(mfl)


def clock360(clip, duration):
    """
    中心点360度扫描
    :param clip:
    :param duration:
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if t >= duration:
            im[0:, 0:] = 1.0
            return im

        ang = math.pi * 2.0 * t / duration
        vec = (-math.sin(ang), math.cos(ang))
        xc, yc = (int(im.shape[1] / 2), int(im.shape[0] / 2))
        size = (im.shape[1], im.shape[0])
        im = drawing.color_split(size, p1=(xc, yc), vector=vec, grad_width=10)
        if ang <= math.pi * 0.5:
            im[0:, 0: xc] = 0.0
            im[yc:, xc:] = 0.0
        elif ang <= math.pi:
            im[0:, 0: xc] = 0.0
        elif ang < math.pi * 1.5:
            im[0:, xc:] = 1.0
            im[0: yc, 0: xc] = 0.0
        else:
            im[0:, xc:] = 1.0
        return im
    return clip.fl(mfl)


def circle_I(clip, duration):
    """
    从中心向外围圆圈装扩大
    :param clip:
    :param duration:
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if t >= duration:
            im[0:, 0:] = 1.0
            return im

        r = math.sqrt(im.shape[0] ** 2 + im.shape[1] ** 2) / 2
        r = r * t / duration
        size = (im.shape[1], im.shape[0])
        center = (im.shape[1] / 2, im.shape[0] / 2)
        im = drawing.circle(size, center, r, blur=10)
        return im
    return clip.fl(mfl)


def circle_O(clip, duration):
    """
    从四周向圆心圆圈中缩小
    :param clip:
    :param duration:
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if t >= duration:
            im[0:, 0:] = 1.0
            return im

        r = math.sqrt(im.shape[0] ** 2 + im.shape[1] ** 2) / 2
        r = r * (1 - t / duration)
        size = (im.shape[1], im.shape[0])
        center = (im.shape[1] / 2, im.shape[0] / 2)
        im = drawing.circle(size, center, r, col1=0, col2=1.0, blur=10)
        return im
    return clip.fl(mfl)


def scroll_V_in(clip, duration):
    """
    从上往下翻页
    :param clip:
    :param duration:
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if t >= duration:
            im[0:, 0:] = 1.0
            return im

        size = (im.shape[1], im.shape[0])
        y2 = int((t * 1.0 / duration) * im.shape[0])
        im2 = drawing.color_split(size, y=y2, grad_width=15, col1=1.0, col2=0)
        return im2
    return clip.fl(mfl)
