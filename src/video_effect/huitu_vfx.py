# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/21 11:55:06
"""
import sys
import copy
from video_effect import huitu_fade
import cv2 as cv
import time
import numpy as np
import moviepy.video.fx.all as vfx
import moviepy.editor as mpy
from video_effect import video_effect_highlight_mask
sys.path.append('../')
from utils import text_utils


get_prop_section = lambda r, s: (s[1] - s[0]) * r + s[0]


def demo(clip):
    """
    制作特效函数的示例
    :param clip:
    :return:
    """
    # demo可以带入控制参数，在这里设置一些局部变量，可以作用到mfl内部，比如运动速度等
    def mfl(gf, t):
        """
        mfl只能有gf， t这两个参数，接口此接口不要变
        :return:
        """
        im = gf(t)  # 固定写法
        new_im = np.copy(im)  # 固定写法
        # 对new_im做一些操作, 注意: 一般不要修改im本身，因为如果是image clip，修改im本身会影响后续的frame
        return new_im
    return clip.fl(mfl)


def blur(clip, gauss_blur=False, ksize=7):
    """
    Returns a clip blur all the img, default box blur
    :param clip:
    :param gauss_blur: 默认box blur
    :param ksize: 必须为奇数
    :return:
    """

    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        if gauss_blur:
            new_im = cv.GaussianBlur(im, (ksize, ksize), 0)
        else:
            new_im = cv.boxFilter(im, -1, (ksize, ksize))
        return new_im
    return clip.fl(mfl)


def delux_text(clip, cyc=4):
    """
    多彩特效，在cyc时间段内hue值循环一个周期
    :param clip:
    :param cyc: secs of color transformation
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        hsv = np.copy(im)
        hsv[0:, 0:, 0] = int(t / cyc * 180) % 180
        hsv[0:, 0:, 1] = 255
        hsv[0:, 0:, 2] = 255
        im2 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return im2
    return clip.fl(mfl)


def delux_image(clip, cyc=4):
    """
    多彩特效，在cyc时间段内hue值循环一个周期
    :param clip:
    :param cyc: secs of color transformation
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        # important: moviepy's image is RGB, not BGR
        hsv = cv.cvtColor(im, cv.COLOR_RGB2HSV)
        hsv[0:, 0:, 0] = (hsv[0:, 0:, 0].astype(np.uint16) + int(t / cyc * 180) % 180) % 180
        im2 = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        return im2
    return clip.fl(mfl)


def blink(clip, duration=0.3):
    """
    快闪特效，在a和b两种颜色之间闪烁的间隔为duration，a为原色，b为255-a
    :param clip:
    :param duration:
    :return:
    """
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        im2 = np.copy(im)
        if int(t / duration) % 2 != 0:
            im2 = 255 - im
        return im2
    return clip.fl(mfl)


def typing(clip, speed, text, font, size, color, pos, paddle=''):
    """
    模拟打字特效
    :param clip:
    :param speed: 每个字的输入速度，单位秒
    :param text:
    :param font: 字体。当前默认字体文件在src/fonts目录下
    :param size: 以pix为单位的文字大小
    :param color:
    :param pos: 根据paddle决定
    :param paddle: 如果指定“center”，模拟doc下居中的打字效果，否则模拟左对齐的打字效果
    :return:
    """
    txt_painters = text_utils.get_text_painters('../fonts')
    if font[-4:] == '.ttf':
        font = font[0: -4]
    txt_paint = txt_painters[font]
    text = text.decode('utf-8')

    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        im2 = np.copy(im)
        txt_num = int(t / speed)
        if txt_num == 0:
            return im

        txt_im = txt_paint.get_text_img(text[0: txt_num], size, color)
        x1, y1 = pos
        x1 = im.shape[1] / 2 if x1 == 'center' else x1
        y1 = im.shape[0] / 2 if y1 == 'center' else y1
        if paddle == 'center':
            x1, y1 = x1 - txt_im.shape[1] / 2, y1 - txt_im.shape[0] / 2
        x2, y2 = x1 + txt_im.shape[1], y1 + txt_im.shape[0]

        if x1 < 0:
            txt_im = txt_im[0:, -x1:]
            x1 = 0
        if y1 < 0:
            txt_im = txt_im[-y1:, 0:]
            y1 = 0
        if x2 > im2.shape[1]:
            txt_im = txt_im[0:, 0: im2.shape[1] - x2]
            x2 = im2.shape[1]
        if y2 > im2.shape[0]:
            txt_im = txt_im[0: im2.shape[0] - y2, 0:]
            y2 = im2.shape[0]

        mask = txt_im[0:, 0:, 3, np.newaxis].astype('float') / 255.0
        mask = np.concatenate((mask, mask, mask), axis=2)

        txt_im = txt_im[0:, 0:, 0: 3]
        im2[y1: y2, x1: x2] = im2[y1: y2, x1: x2] * (1 - mask) + txt_im * mask
        return im2
    return clip.fl(mfl)


def crosscollapsein(clip, duration, style='fade'):
    """
    转场动画特效，特效类型由style指定，特效描述见huitu_fade.py
    :param clip:
    :param duration:
    :param style:
    :return:
    """
    if clip.mask is None:
        clip = clip.add_mask()

    # example: huitu_fade.block_V_in
    mask_fx_name = 'huitu_fade.' + style
    try:
        mask_fx_func = eval(mask_fx_name)
    except:
        mask_fx_func = vfx.fadein

    clip.mask = clip.mask.fx(mask_fx_func, duration)

    return clip


def phantom1(clip, cyc=0.55, enlarge_rates=(1.0, 3.0)):
    """
    模拟抖音的 灵魂出窍 特效
    :param clip:
    :param cyc:
    :param enlarge_rates:
    :return:
    """
    alpha_rates = [0.5, 0.0]

    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        r = t * 1.0 / cyc
        r = cyc * (r - int(r))
        alpha_rate = get_prop_section(r, alpha_rates)
        r = 1.0 - (1.0 - r) ** 4
        enlarge_rate = get_prop_section(r, enlarge_rates)
        im2 = cv.resize(im, (0, 0),
                        fx=enlarge_rate,
                        fy=enlarge_rate,
                        interpolation=cv.INTER_LINEAR)
        x1, y1 = (im2.shape[1] - im.shape[1]) / 2, (im2.shape[0] - im.shape[0]) / 2
        x2, y2 = x1 + im.shape[1], y1 + im.shape[0]
        im2 = im2[y1: y2, x1: x2]
        im2 = im * (1 - alpha_rate) + im2 * alpha_rate
        return im2
    return clip.fl(mfl)


def phantom2(clip, cyc=0.45, enlarge_rates=(1.0, 1.1)):
    """
    模拟抖音的 抖动 效果
    :param clip:
    :param cyc:
    :param enlarge_rates:
    :return:
    """
    rgb_rates = [0.8, 0.5, 0.2]

    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        r = t * 1.0 / cyc
        r = cyc * (r - int(r))
        r = 1.0 - (1.0 - r) ** 4
        enlarge_rate = get_prop_section(r, enlarge_rates)
        im2 = cv.resize(im, (0, 0),
                        fx=enlarge_rate,
                        fy=enlarge_rate,
                        interpolation=cv.INTER_LINEAR)
        x1, y1 = (im2.shape[1] - im.shape[1]) / 2, (im2.shape[0] - im.shape[0]) / 2
        x2, y2 = x1 + im.shape[1], y1 + im.shape[0]
        im3 = im2[y1: y2, x1: x2]

        x1, y1 = (im2.shape[1] - im.shape[1]) / 4 * 3, (im2.shape[0] - im.shape[0]) / 4 * 3
        x2, y2 = x1 + im.shape[1], y1 + im.shape[0]
        im2 = im2[y1: y2, x1: x2]
        for i in range(im2.shape[2]):
            im2[0:, 0:, i] = im2[0:, 0:, i] * rgb_rates[i] + im3[0:, 0:, i] * (1 - rgb_rates[i])
        return im2
    return clip.fl(mfl)


def hist_equalization(clip):
    """
    亮度饱和度均衡
    :param clip:
    :return:
    """
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    def mfl(gf, t):
        """
        :return:
        """
        im = gf(t)
        hsv = cv.cvtColor(im, cv.COLOR_RGB2HSV)
        hsv[0:, 0:, 1] = clahe.apply(hsv[0:, 0:, 1])
        hsv[0:, 0:, 2] = clahe.apply(hsv[0:, 0:, 2])
        new_im = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        return new_im
    return clip.fl(mfl)


def text_clip_vertical(clip, text, font, size=0, color1=(255, 255, 255), color2=(0, 0, 0)):
    """
    竖向排列的等距文字，背景色为黑白间隔，文字色为背景的反向颜色
    :param clip:
    :param text:
    :param font:
    :param size:
    :param color1:
    :param color2:
    :return:
    """
    txt_painters = text_utils.get_text_painters('../fonts')
    if font == '':
        font = 'PingFang Bold'
    elif font[-4:] == '.ttf':
        font = font[0: -4]
    txt_paint = txt_painters[font]
    text = text.decode('utf-8')

    im = np.ndarray(shape=(clip.size[1], clip.size[0], 3), dtype='uint8')
    height = im.shape[0] * 1.0 / len(text)
    width = im.shape[1]
    if size <= 0:
        size = int(min(height, width) / 3 * 2)

    for i in range(len(text)):
        if i % 2 == 0:
            text_color, bkg_color = color1, color2
        else:
            text_color, bkg_color = color2, color1

        y1 = int(i * height)
        y2 = int(y1 + height)
        x1 = 0
        x2 = width
        im[y1: y2, x1: x2] = bkg_color
        txt_paint.draw_text(im[y1: y2, x1: x2, 0:], 'center', text[i], size, text_color)
    im[y2:, 0:] = bkg_color

    def mfl(gf, t):
        """
        mfl只能有gf， t这两个参数，接口此接口不要变
        :return:
        """
        return im
    return clip.fl(mfl)


def float_layer(clip, x1, x2, y1, y2, transparent, text, font, size, color):
    """
    浮层
    :param clip:
    :param x1: -1表示从0开始
    :param x2: -1表示到最大宽度
    :param y1: 同上
    :param y2: 同上
    :param transparent: 透明比例
    :param text:
    :param font:
    :param size:
    :param color:
    :return:
    """
    x1 = max(0, x1)
    x2 = min(clip.w, x2)
    y1 = max(0, y1)
    y2 = min(clip.h, y2)
    x2 = clip.w if x2 < 0 else x2
    y2 = clip.h if y2 < 0 else y2

    if text != '':
        txt_painters = text_utils.get_text_painters('../fonts')
        if font == '':
            font = 'PingFang Bold'
        elif font[-4:] == '.ttf':
            font = font[0: -4]
        txt_paint = txt_painters[font]
        # text = text.decode('utf-8')
        text_img = txt_paint.get_text_img(text, -1, color, bgcolor='', image_size=(y2-y1, x2-x1))
        text_mask = text_img[0:, 0:, 3].astype('float') / 255.0
        text_mask = cv.merge((text_mask, text_mask, text_mask))
        text_img = text_img[0:, 0:, 0: 3]

    def mfl(gf, t):
        """
        mfl只能有gf， t这两个参数，接口此接口不要变
        :return:
        """
        im = gf(t)  # 固定写法
        new_im = np.copy(im)  # 固定写法
        if abs(transparent - 1.0) > 0.01:
            new_im[y1: y2, x1: x2] = new_im[y1: y2, x1: x2] * transparent
        if text != '':
            new_im[y1: y2, x1: x2] = new_im[y1: y2, x1: x2] * (1 - text_mask) \
                            + text_img * text_mask
        return new_im
    return clip.fl(mfl)


def rolling_title(clip, x1, x2, y1, y2, speed, text, font, size, color):
    """
    重复滚动标题功能，支持长标题首位衔接滚动
    :param clip:
    :param x1: -1表示从0开始
    :param x2: -1表示到最大宽度
    :param y1: 同上
    :param y2: 同上
    :param speed: 滚动速度，单位: 像素/秒
    :param text:
    :param font:
    :param size:
    :param color:
    :return:
    """
    x1 = max(0, x1)
    x2 = min(clip.w, x2)
    y1 = max(0, y1)
    y2 = min(clip.h, y2)
    x2 = clip.w if x2 < 0 else x2
    y2 = clip.h if y2 < 0 else y2

    assert(text != '')
    txt_painters = text_utils.get_text_painters('../fonts')
    if font == '':
        font = 'PingFang Bold'
    elif font[-4:] == '.ttf' or font[-4:] == '.TTF':
        font = font[0: -4]
    font = font.decode('utf-8').encode('gb18030')
    txt_paint = txt_painters[font]
    text = text.decode('utf-8')
    text_img = txt_paint.get_text_img(text, size, color)

    # 处理文字图片超过设定高度的情况
    if text_img.shape[0] > (y2 - y1):
        y00 = (text_img.shape[0] - (y2 - y1)) / 2
        text_img = text_img[y00: y00 + (y2 - y1), 0:, 0:]
    # 对text_img左右扩宽，有利于体验
    margin = text_img.shape[0]
    new_img = np.zeros(shape=(text_img.shape[0], text_img.shape[1] + margin * 2, 4), dtype=np.uint8)
    new_img[0:, margin: margin + text_img.shape[1], 0:] = text_img
    text_img = new_img

    # 将text img 放置到目标尺寸中，若宽度小于目标尺寸则居中，若大于目标尺寸则扩展并左对齐
    height = y2 - y1
    width = max(x2 - x1, text_img.shape[1])
    new_img = np.zeros(shape=(height, width, 4), dtype=np.uint8)
    y00 = (height - text_img.shape[0]) / 2
    x00 = (width - text_img.shape[1]) / 2
    new_img[y00: y00 + text_img.shape[0], x00: x00 + text_img.shape[1], 0:] = text_img
    text_img = np.zeros(shape=(new_img.shape[0], new_img.shape[1] * 2, 4), dtype=np.uint8)
    text_img[0:, 0: new_img.shape[1], 0:] = new_img
    text_img[0:, new_img.shape[1]:, 0:] = new_img

    text_mask = text_img[0:, 0:, 3].astype('float') / 255.0
    text_mask = cv.merge((text_mask, text_mask, text_mask))
    text_img = text_img[0:, 0:, 0: 3]

    def mfl(gf, t):
        """
        mfl只能有gf， t这两个参数，接口此接口不要变
        :return:
        """
        im = gf(t)  # 固定写法
        new_im = np.copy(im)  # 固定写法
        width = text_img.shape[1] / 2
        pos1 = int(speed * t % width)
        pos2 = pos1 + (x2 - x1)

        new_im[y1: y2, x1: x2] = new_im[y1: y2, x1: x2] * (1.0 - text_mask[0:, pos1: pos2, 0:]) \
                    + text_img[0:, pos1: pos2, 0:] * text_mask[0:, pos1: pos2, 0:]
        return new_im
    return clip.fl(mfl)


def resize_overlap_blur(clip, width, height, edge, size, ksize=30):
    """
    将当前图像，resize到两张图像，以中间为锚点叠加
    图像1，resize拉升到width，height，并且用ksize做平均模糊的宽高，
    图像2，以edge（宽或高）为目标，等比例拉升到size，居中叠加图像1
    :param clip:
    :param width:
    :param height:
    :param edge: 'width' or 'height'
    :param size:
    :param ksize:
    :return:
    """

    def resize_overlap_blur_img(img, width, height, edge, size, ksize):
        """
        resize_overlap_blur_img
        :return:
        """
        ori_height, ori_width = img.shape[0: 2]
        new_im = cv.resize(img, (width, height))
        new_im = cv.boxFilter(new_im, -1, (ksize, ksize))
        lr = np.average(new_im, axis=(0, 1, 2)) / 255.0
        lr = lr * 0.8
        new_im = new_im * lr + 255.0 * (1 - lr)
        if edge == 'width':
            width2 = size
            height2 = int(ori_height * 1.0 * width2 / ori_width)
            new_im2 = cv.resize(img, (width2, height2))
            if height2 > height:
                y = (height2 - height) / 2
                new_im2 = new_im2[y: y + height, 0:]
                height2 = new_im2.shape[0]
        elif edge == 'height':
            height2 = size
            width2 = int(ori_width * 1.0 * height2 / ori_height)
            new_im2 = cv.resize(img, (width2, height2))
            if width2 > width:
                x = (width2 - width) / 2
                new_im2 = new_im2[0:, x: x + width]
                width2 = new_im2.shape[1]
        else:
            print('unknown resize_overlap_blur edge %s' % edge)
            assert (1 == 2)

        x0 = int((width - width2) / 2)
        y0 = int((height - height2) / 2)
        new_im[y0: y0 + height2, x0: x0 + width2] = new_im2
        return new_im

    if isinstance(clip, mpy.ImageClip):
        d_img = resize_overlap_blur_img(clip.img, width, height, edge, size, ksize)
    else:
        d_img = None

    # demo可以带入控制参数，在这里设置一些局部变量，可以作用到mfl内部，比如运动速度等
    def mfl(gf, t):
        """
        mfl只能有gf， t这两个参数，接口此接口不要变
        :return:
        """
        im = gf(t)  # 固定写法
        if d_img is not None:
            new_img = copy.copy(d_img)
        else:
            new_img = resize_overlap_blur_img(im, width, height, edge, size, ksize)
        return new_img
    return clip.fl(mfl)


def highlight(clip, begin, end, anchor='topleft', highlight=1.10, resize=1.0, win_len=100):
    """
    高亮区域扫动
    :param clip:
    :param begin: 开始时间
    :param end: 结束时间
    :param anchor: 高亮区域为平行四边形，定点为anchor
    :param highlight: 高亮度为原亮度 highlight倍
    :param resize: 高亮图像为原图像 highlight倍
    :param win_len: 高亮块的宽度
    :return:
    """
    def process_img(img, highlight, resize):
        """
        highlight and resize img
        :param img:
        :return:
        """
        new_im = copy.deepcopy(img)
        if highlight > 1.0:
            highlight_r = highlight - 1.0
            new_im = new_im * (1.0 - highlight_r) + 255.0 * highlight_r
        elif highlight < 1.0:
            new_im = new_im * highlight

        if resize != 1.0:
            new_im = cv.resize(new_im, dsize=(0, 0), fx=resize, fy=resize)
            x, y = int((new_im.shape[1] - img.shape[1]) / 2), int((new_im.shape[0] - img.shape[0]) / 2)
            new_im = new_im[y: y + img.shape[0], x: x + img.shape[1]]

        return new_im

    mask_proxy = video_effect_highlight_mask.RhombusMask(end - begin, anchor, win_len)

    if isinstance(clip, mpy.ImageClip):
        d_img = process_img(clip.img, highlight, resize)
    else:
        d_img = None

    def mfl(gf, t):
        """
        mfl只能有gf， t这两个参数，接口此接口不要变
        :return:
        """
        im = gf(t)  # 固定写法
        if t < begin or t > end:
            return im
        t = t - begin

        if d_img is None:
            new_im = process_img(im, highlight, resize)
        else:
            new_im = np.copy(d_img)

        mask = mask_proxy.get_mask(t, im.shape[0], im.shape[1])
        new_im = im * (1.0 - mask) + new_im * mask
        # 对new_im做一些操作, 注意: 一般不要修改im本身，因为如果是image clip，修改im本身会影响后续的frame
        return new_im
    return clip.fl(mfl)
