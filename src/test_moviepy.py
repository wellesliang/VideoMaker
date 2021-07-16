# -*- coding: utf-8 -*-

"""
video maker

Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/01 09:10:06
"""


import os
import sys
import numpy as np
import math
import cv2 as cv
from utils import text_utils
from utils import image_utils

import moviepy.editor as mpy
import moviepy.video.fx.all as vfx


SCROLL_IN_TIME = 2.0
SCROLL_SHOW_TIME = 4.0
SCROLL_SHOW_R = 1.08

tmp = cv.imread('../resource/vfx/show_detail_mask_rhombus.png', cv.IMREAD_UNCHANGED)[0:, 0:, 3]
SCROLL_MASK = cv.merge((tmp, tmp, tmp))
tmp = cv.imread('../resource/vfx/show_detail_mask_circle.png', cv.IMREAD_UNCHANGED)[0:, 0:, 3]
SCROLL_MASK_CIRCLE = cv.merge((tmp, tmp, tmp))

decel_circle = lambda x: math.sqrt(1 - (x - 1) * (x - 1))
decel_sin = lambda x: math.sin(x * math.pi / 2)


def scroll_img_in_horizontal(gf, t):
    """
    scroll img from left to right in n seconds
    """
    frame0 = gf(t)
    if t >= SCROLL_IN_TIME:
        return frame0
    
    #x_rate = decel_circle(t / scroll_time)
    x_rate = decel_sin(t / SCROLL_IN_TIME)
    x = int(x_rate * frame0.shape[1])
    frame1 = np.ndarray(shape = frame0.shape, dtype = 'uint8')
    frame1[0:, 0:] = (255, 255, 255)
    
    if x == 0:
        return frame1
    
    frame1[0:, 0: x, 0:] = frame0[0:, -int(x):, 0:]
    return frame1


def scroll_img_in_vertical(gf, t):
    """
    scroll img from up to down in n seconds
    """
    frame0 = gf(t)
    if t >= SCROLL_IN_TIME:
        return frame0
    
    y_rate = decel_circle(t / SCROLL_IN_TIME)
    y = int(y_rate * frame0.shape[0])

    frame1 = np.ndarray(shape = frame0.shape, dtype = 'uint8')
    frame1[0:, 0:] = (255, 255, 255)
    
    if y == 0:
        return frame1
    
    frame1[0: y, 0: , 0:] = frame0[-y:, 0:, 0:]
    return frame1


def show_detail_horizontal(gf, t):
    """
    scroll show detail from left to right
    """
    frame0 = gf(t)
    if t >= SCROLL_SHOW_TIME:
        return frame0
    
    frame1 = np.copy(frame0)
    r = SCROLL_SHOW_R
    frame2 = cv.resize(frame1, None, fx = r, fy = r, interpolation = cv.INTER_LINEAR)
    dy = (frame2.shape[0] - frame1.shape[0]) / 2
    dx = (frame2.shape[1] - frame1.shape[1]) / 2
    frame2 = frame2[dy: dy + frame1.shape[0], dx: dx + frame1.shape[1]]
    frame2 = image_utils.hist_equalization(frame2)
    
    mask_r = frame1.shape[0] / SCROLL_MASK.shape[0]
    mask_dsize = (mask_r * SCROLL_MASK.shape[1], frame1.shape[0])
    scroll_mask = cv.resize(SCROLL_MASK, mask_dsize, interpolation = cv.INTER_LINEAR)
    
    win_width = scroll_mask.shape[1]
    x_rate = t / SCROLL_SHOW_TIME
    x1 = int(x_rate * (win_width + frame1.shape[1]) - win_width)
    x2 = int(x1 + win_width)

    if x1 < 0:
        scroll_mask = scroll_mask[0:, -x2:, 0:] / 255.0
    elif x2 > frame1.shape[1]:
        scroll_mask = scroll_mask[0:, 0: frame1.shape[1] - x1, 0:] / 255.0
    else:
        scroll_mask = scroll_mask / 255.0
        
    x1 = max(0, x1)
    x2 = min(frame1.shape[1], x2)
    if x2 - x1 == 0:
        return frame0
    
    frame1[0:, x1: x2] = frame1[0:, x1: x2, 0:] * scroll_mask + frame2[0:, x1: x2, 0:] * (1 - scroll_mask)
    return frame1


def show_detail_spin(gf, t):
    """
    a等距螺旋轨迹显示商品细节
    """
    frame0 = gf(t)
    if t > SCROLL_SHOW_TIME:
        return frame0
    
    frame1 = np.copy(frame0)
    frame_w = frame1.shape[1]
    frame_h = frame1.shape[0]
    r = SCROLL_SHOW_R
    frame2 = cv.resize(frame1, None, fx = r, fy = r, interpolation = cv.INTER_LINEAR)
    dy = (frame2.shape[0] - frame1.shape[0]) / 2
    dx = (frame2.shape[1] - frame1.shape[1]) / 2
    frame2 = frame2[dy: dy + frame1.shape[0], dx: dx + frame1.shape[1]]
    frame2 = image_utils.hist_equalization(frame2)
    
    scroll_mask = SCROLL_MASK_CIRCLE / 255.0
    mask_w = scroll_mask.shape[1]
    mask_h = scroll_mask.shape[0]
    ang = t / SCROLL_SHOW_TIME * 2 * math.pi * 2
    r = t / SCROLL_SHOW_TIME * min(frame0.shape[0], frame0.shape[0]) / 2
    xc = math.cos(ang) * r + frame0.shape[1] / 2
    yc = math.sin(ang) * r + frame0.shape[0] / 2
    
    x1 = int(xc - mask_w / 2)
    x2 = int(x1 + mask_w)
    y1 = int(yc - mask_h / 2)
    y2 = int(y1 + mask_h)
    if x2 <= 0 or x1 >= frame_w or y2 <= 0 or y1 >= frame_h:
        return frame0
    
    if x1 < 0:
        scroll_mask = scroll_mask[0:, -x2:]
        x1 = 0
    if x2 > frame_w:
        scroll_mask = scroll_mask[0:, 0: frame_w - x1]
    if y1 < 0:
        scroll_mask = scroll_mask[-y2:, 0:]
        y1 = 0
    if y2 > frame_h:
        scroll_mask = scroll_mask[0: frame_h - y1, 0:]
        y2 = frame_h
    frame1[y1: y2, x1: x2] = frame1[y1: y2, x1: x2] * scroll_mask + frame2[y1: y2, x1: x2] * (1 - scroll_mask) 
    return frame1


def change_color(gf, t):
    """
    change color by t
    """
    frame0 = gf(t)
    hsv = np.copy(frame0)
    hsv[0:, 0:, 0] = int(t / 4.0 * 180) % 180
    hsv[0:, 0:, 1] = 255
    hsv[0:, 0:, 2] = 255
    frame1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    #frame1 = np.concatenate((frame1, frame0[0:, 0:, 3, np.newaxis]), axis = 2)
    return frame1

    
def test3():
    txt_painter = text_utils.PutChineseText('../fonts/PingFang Bold.ttf')
    
    img_dir = '../img/'
    clip0 = mpy.ImageClip('../resource/vfx/white.jpg')
    clip0 = clip0.set_duration(10)
    
    clips = []
    start_t = 0

    clip = mpy.ImageClip(img_dir + '/01.jpg').set_position('center')

    txt_dur = 6
    txt = '2400万AI摄影。会变色，更潮美！'.decode('utf-8')
    txt_img = txt_painter.get_text_img(txt, 40, (255, 0, 0))
    txt_clip = mpy.ImageClip(txt_img).set_duration(txt_dur) \
                    .set_pos(('center', 0.8), relative = True) \
                    .set_start(start_t)
    txt_clip = txt_clip.fl(change_color).fl(scroll_img_in_horizontal).crossfadeout(1.5)
    clips.append(txt_clip)
    
    dur1 = SCROLL_IN_TIME
    clip1 = clip.set_duration(dur1).fl(scroll_img_in_vertical)
    clips.append(clip1.set_start(start_t))
    start_t += dur1
    
    dur2 = SCROLL_SHOW_TIME
    clip2 = clip.set_duration(SCROLL_SHOW_TIME) \
            .fx(vfx.resize, newsize = lambda t: 1 + 0.015 * t) \
            .fl(show_detail_horizontal)
    clips.append(clip2.set_start(start_t))
    start_t += dur2
    
    dur_last = 1.5
    clips[-1] = clips[-1].set_end(start_t + dur_last).crossfadeout(dur_last)
    start_t += dur_last
    
    
    clip = mpy.ImageClip(img_dir + '/02.jpg').set_pos('center')

    txt_dur = 6
    txt = '超清全面屏设计,色彩显示更加细腻'.decode('utf-8')
    txt_img = txt_painter.get_text_img(txt, 40, (0, 255, 255))
    txt_clip = mpy.ImageClip(txt_img).set_duration(txt_dur) \
                    .set_pos(('center', 0.8), relative = True) \
                    .set_start(start_t)
    txt_clip = txt_clip.crossfadein(2.0).crossfadeout(1.5)
    clips.append(txt_clip)
    
    dur1 = SCROLL_IN_TIME
    clip1 = clip.set_duration(dur1).fl(scroll_img_in_horizontal)
    clips.append(clip1.set_start(start_t))
    start_t += dur1
    
    dur2 = SCROLL_SHOW_TIME
    clip2 = clip.set_duration(dur2) \
            .fx(vfx.resize, newsize = lambda t: 1 + 0.015 * t) \
            .fl(show_detail_spin)
    clips.append(clip2.set_start(start_t))
    start_t += dur2
    
    dur_last = 1.5
    clips[-1] = clips[-1].set_end(start_t + dur_last).crossfadeout(dur_last)
    start_t += dur_last
    
    
    clip = mpy.ImageClip(img_dir + '/03.jpg').set_pos('center')

    txt_dur = 6
    txt = '智能柔光双摄,让照片更显立体感'.decode('utf-8')
    txt_img = txt_painter.get_text_img(txt, 40, (0, 255, 255))
    txt_clip = mpy.ImageClip(txt_img).set_duration(txt_dur) \
                    .set_pos(('center', 0.8), relative = True) \
                    .set_start(start_t)
    txt_clip = txt_clip.fl(scroll_img_in_horizontal).crossfadeout(1.5)
    clips.append(txt_clip)
    
    dur1 = SCROLL_IN_TIME
    clip1 = clip.set_duration(dur1).crossfadein(1.5)
    clips.append(clip1.set_start(start_t))
    start_t += dur1
    
    dur2 = SCROLL_SHOW_TIME
    clip2 = clip.set_duration(dur2) \
            .fl(show_detail_horizontal)
    clips.append(clip2.set_start(start_t))
    start_t += dur2
    
    dur_last = 1.5
    clips[-1] = clips[-1].set_end(start_t + dur_last).crossfadeout(dur_last)
    start_t += dur_last
    
    
    clip0 = clip0.set_duration(start_t).set_start(0)
    clip0 = mpy.CompositeVideoClip([clip0] + clips)
    
    #clip0.write_videofile('../output/output.mp4', fps = 36)
    #clip0.write_gif('../output/output.gif', program = 'imageio', opt = 'wu', loop = 0, fps = 18)
    #bos_client = BosClient(bos_conf.config)
    #bos_client.put_object_from_file('pa-test', 'liangweimingtest3', '../output/output.mp4')

    print('finished')


def make_letter_clips(texts, tdur, tsize, text_pos, screen_size, tcolor):
    """
    make clips per word
    Args:
        scn_w, scn_y: screen size
        tx, ty: text pos, normaly ty is used, and x 'center'
    """
    tx, ty = text_pos
    scn_w, scn_y = screen_size
    txt_painter = text_utils.PutChineseText('../fonts/PingFang Bold.ttf')
    x, y = 0, ty
    poss = []
    clips = []
    for word in texts:
        txt_img = txt_painter.get_text_img(word, tsize, tcolor)
        txt_clip = mpy.ImageClip(txt_img).set_duration(tdur)
        clips.append(txt_clip)
        poss.append((x, y + tsize - txt_img.shape[0]))
        x += txt_img.shape[1]
    dx = (scn_w - x) / 2
    
    for i, clip in enumerate(clips):
        clips[i] = clip.set_position((poss[i][0] + dx, poss[i][1]))
    return clips


def test4():
    """
    text animation
    """    
    img_dir = '../img/'
    clip0 = mpy.ImageClip('../resource/vfx/white.jpg')
    clip0 = clip0.set_duration(10)
    
    clips = []
    start_t = 0

    clip = mpy.ImageClip(img_dir + '/01.jpg').set_pos('center')

    screensize = (800, 600)
    txt = '2400万AI摄影。会变色，更潮美！'.decode('utf-8')
    letters = make_letter_clips(txt, 6, 50, (-1, 450), screensize, (0, 255, 255))
    rotMatrix = lambda a: np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])

    def vortex(screenpos, i, nletters):
        """
        vortex
        """
        d = lambda t: 1.0 / (0.3 + t ** 6) #damping
        a = i * np.pi / nletters # angle of the movement
        v = rotMatrix(a).dot([-1, 0])
        if i % 2:
            v[1] = -v[1]
        return lambda t: screenpos + 600 * d(t) * rotMatrix(0.5 * d(t) * a).dot(v)
    
    def cascade(screenpos, i, nletters):
        """
        cascade
        """
        v = np.array([0, -1])
        d = lambda t: 1 if t < 0 else abs(np.sinc(t) / (1 + t ** 4))
        return lambda t: screenpos + v * 600 * d(t - 0.15 * i)

    def arrive(screenpos, i, nletters):
        """
        arrive
        """
        v = np.array([-1, 0])
        d = lambda t: max(0, 3 - 3 * t)
        return lambda t: screenpos - 600 * v * d(t - 0.2 * i)
    
    def vortexout(screenpos, i, nletters):
        """
        vortexout
        """
        d = lambda t: max(0, t) #damping
        a = i * np.pi / nletters # angle of the movement
        v = rotMatrix(a).dot([-1, 0])
        if i % 2: v[1] = -v[1]
        return lambda t: screenpos + 600 * d(t - 0.1 * i) * rotMatrix(-0.2 * d(t) * a).dot(v)
    
    
    def move_letters(letters, funcpos):
        """
        move letters
        """
        return [letter.set_pos(funcpos(letter.pos(0), i, len(letters))) \
                for i, letter in enumerate(letters)]
    
    txt_clip1 = mpy.CompositeVideoClip(move_letters(letters, vortex), \
                                      size = screensize).subclip(0, 5)
    txt_clip2 = mpy.CompositeVideoClip(move_letters(letters, cascade), \
                                      size = screensize).subclip(0, 5).set_start(5)
    txt_clip3 = mpy.CompositeVideoClip(move_letters(letters, arrive), \
                                      size = screensize).subclip(0, 5).set_start(10)
    txt_clip4 = mpy.CompositeVideoClip(move_letters(letters, vortexout), \
                                      size = screensize).subclip(0, 5).set_start(15)
    
    clips = [txt_clip1, txt_clip2, txt_clip3, txt_clip4]
    
    clip0 = clip0.set_duration(20).set_start(0)
    clip0 = mpy.CompositeVideoClip([clip0] + clips)
    
    clip0.write_videofile('../output/output.mp4', fps = 36)
    #clip0.write_gif('../output/output.gif', program = 'imageio', opt = 'nq', loop = 0, fps = 18)
    
    
if __name__ == '__main__':
    test4()
    #test3()