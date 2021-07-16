# -*- coding: utf-8 -*-
"""
Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/21 11:55:06
"""

import os
import random
import logging
import requests
import tempfile
import numpy as np
import cv2 as cv
import moviepy.editor as mpy
import moviepy.video.fx.all as vfx
import copy

from video_effect import huitu_vfx
from video_effect import movement
from utils import text_utils
from utils.image_utils import get_img
from utils.video_utils import get_video_local_path


class Video(object):
    """
    主要是video相关的存储功能，无他
    """
    def __init__(self, clip):
        self.clip = clip

    def save(self, path, verbose=True):
        """
        :param path:
        :param verbose:
        :return:
        """
        if path.split('.')[-1] == 'ogv':
            self.clip.write_videofile(path,
                                      threads=1,
                                      verbose=verbose,
                                      audio=True,
                                      audio_codec='libvorbis',
                                      bitrate='2400k',
                                      ffmpeg_params=[
                                          '-movflags', 'faststart'
                                        ])
        else:
            self.clip.write_videofile(path,
                                      threads=1,
                                      verbose=verbose,
                                      audio_codec='aac',
                                      audio=True,
                                      ffmpeg_params=[
                                          '-movflags', 'faststart'
                                        ])

    def save_multi(self, path1, path2, verbose=True):
        """
        :param path1:
        :param path2:
        :param verbose:
        :return:
        """
        assert(path2.split('.')[-1] == 'ogv')
        wh = '%d*%d' % (self.clip.w / 2, self.clip.h / 2)
        self.clip.write_videofile(path1,
                                  verbose=verbose,
                                  audio_codec='aac',
                                  audio=True,
                                  threads=1,
                                  ffmpeg_params=[
                                      '-movflags', 'faststart',
                                      '-vcodec', 'libtheora',
                                      '-acodec', 'libvorbis',
                                      '-s', wh,
                                      '-b', '1200k',
                                path2])


class VideoCompositer(object):
    def __init__(self, name):
        self.name = name
        self.temp_local_file = []

    def composit(self, video_template):
        """
        主刀视频合成的整个过程，大体流程是，解析clip配置，设置clip参数，调用moviepy合成
        :param vedio_template:
        :return:
        """
        vt = video_template
        bk_img = np.ndarray(shape=(vt.height, vt.width, 3), dtype='uint8')
        bk_img[0:, 0:] = vt.color
        max_duration = -1

        clips = []
        for idx, clip_template in enumerate(vt.video_clips):
            # 根据clip类型分别设置clip的初始class
            if clip_template.type == 'image':
                # 这个type下，image保持原格式，png会保留mask
                img = get_img(clip_template.url, 'rgb')
                if img is None:
                    logging.fatal('get img failed: %s', clip_template.url)
                    return None
                clip = mpy.ImageClip(img)
            elif clip_template.type == 'image_nomask':
                # 这个type下，所有的png会当作jpg处理，一般情况下，建议使用此type
                img = get_img(clip_template.url, 'rgb')
                img = img[0:, 0:, 0: 3]
                if img is None:
                    logging.fatal('get img failed: %s', clip_template.url)
                    return None
                clip = mpy.ImageClip(img)
            elif clip_template.type == 'video':
                # 视频需要先下载到本地，再load进内存，会在本地产生临时文件，需要在析构的时候删除
                tmpfile = tempfile.NamedTemporaryFile(delete=False)
                #video_local_name = self.name + '_' + str(idx)
                video_local_name = tmpfile.name
                video_local_path = get_video_local_path(clip_template.url, video_local_name)
                if video_local_path is None or video_local_path == '':
                    logging.fatal('get video local path failed: %s', clip_template.url)
                    return None
                self.temp_local_file.append(video_local_path)
                clip = mpy.VideoFileClip(video_local_path)
                #os.remove(tmpfile.name)
            elif clip_template.type == 'text':
                # 文本类text的，主要是通过调用text painter，将文本画成图像
                if clip_template.text != '':
                    txt_font = clip_template.font
                    txt_size = clip_template.size
                    txt_color = clip_template.color
                    txt_painters = text_utils.get_text_painters('../fonts')
                    text_painter = txt_painters[txt_font]
                    img = text_painter.get_text_img(clip_template.text, txt_size, txt_color)
                else:
                    img = get_img(clip_template.url, 'rgb')
                if img is None:
                    logging.fatal('get img failed: %s', clip_template.url)
                    return None
                clip = mpy.ImageClip(img)
            else:
                logging.fatal('unknown clip type %s', clip_template.type)
                return None

            # 这个函数实现的是clip特效，mask等等相关的具象化
            deco_clips = decorate_video_clip(clip, clips, clip_template, vt)
            if deco_clips is None:
                logging.fatal('decorate video clip failed in %s' % clip_template.name)
                return None
            for clip in deco_clips:
                clips.append(clip)
                if max_duration < clip.end:
                    max_duration = clip.end

        # duration相关的实现
        if vt.duration < 0:
            all_duration = max_duration
            for i, clip in enumerate(clips):
                if clip.duration is None:
                    clips[i] = clip.set_duration(all_duration)
        else:
            all_duration = vt.duration
        clip0 = mpy.ImageClip(bk_img).set_duration(all_duration).set_start(0)
        final_clip = mpy.CompositeVideoClip([clip0] + clips).set_fps(vt.fps)

        # 音频相关
        audio_clip = None
        for clip_template in vt.audio_clips:
            audio_clip = mpy.AudioFileClip(clip_template.url).\
                audio_loop(duration=final_clip.duration)
        if audio_clip is not None:
            final_clip = final_clip.set_audio(audio_clip)

        return Video(final_clip)

    def __del__(self):
        # 删除临时文件
        for fname in self.temp_local_file:
            try:
                os.remove(fname)
            except:
                logging.warning('cannot remove temp local file: %s' % fname)
                continue
        self.temp_local_file = []


def decorate_video_clip(clip, clips, clip_template, video_clip_template):
    """
    这个函数实现的是clip特效，mask等等相关的具象化
    :param clip:
    :param clips:
    :param clip_template:
    :param video_clip_template
    :return: None, if decorate failed
    """
    tpl = clip_template

    if tpl.subclip is not None:
        clip = clip.subclip(tpl.subclip[0], tpl.subclip[1])

    # 开始 结束相关的实现
    if tpl.duration is not None:
        if tpl.duration[0] in ['last_begin', 'last_end']\
                or tpl.duration[1] in ['last_begin', 'last_end']:
            begin = tpl.duration[0]
            if begin == 'last_begin':
                begin = clips[-1].start
            elif begin == 'last_end':
                begin = clips[-1].end
            assert(begin is not None)

            delta = tpl.duration[2]
            if delta is not None:
                begin += delta

            end = tpl.duration[1]
            if end == 'last_begin':
                end = clips[-1].begin
            elif end == 'last_end':
                end = clips[-1].end

            dura = tpl.duration[3]
            if dura is not None:
                end = begin + dura

            if end is None:
                end = clips[-1].end
        else:
            begin = tpl.duration[0]
            end = tpl.duration[1]
        # check sure, begin and end is "int" instance not "string"
        assert(type(begin) == int or type(begin) == float)
        assert(type(end) == int or type(end) == float)
        clip = clip.set_start(begin).set_end(end)
    elif video_clip_template.duration > 0:
        clip = clip.set_start(0).set_end(video_clip_template.duration)
    else:
        clip = clip.set_start(0).set_end(clip.duration)

    # 裁剪相关实现
    if tpl.crop is not None:
        clip = clip.fx(vfx.crop,
                       x1=tpl.crop[0],
                       y1=tpl.crop[1],
                       x2=tpl.crop[2],
                       y2=tpl.crop[3])

    if tpl.mask is not None:
        mask_img = get_img(tpl.mask.url, 'rgb')
        if mask_img is None:
            logging.fatal('load mask img failed, %s' % tpl.mask.url)
            return None
        if tpl.mask.crop is not None:
            x1, y1, x2, y2 = tpl.mask.crop
            mask_img = mask_img[x1: x2, y1: y2]
        if tpl.mask.position is not None:
            x1, y1, width, height = tpl.mask.position
        else:
            x1, y1, width, height = 0, 0, clip.w, clip.h
        mask_img = cv.resize(mask_img,
                             (width, height),
                             fx=0,
                             fy=0,
                             interpolation=cv.INTER_CUBIC)
        tmp = mask_img[0:, 0:, 3]
        mask_img = cv.merge((tmp, tmp, tmp))
        mask_img2 = np.ndarray(shape=(clip.h, clip.w, 3), dtype='uint8')
        mask_img2[0:, 0:] = (255, 255, 255)
        mask_img2[y1: y1 + height, x1: x1 + width] = mask_img
        mask_clip = mpy.ImageClip(mask_img2, ismask=True)
        clip = clip.set_mask(mask_clip)

    # 特效相关的实现，在特效函数中，这里只是调用
    for fx_param in tpl.fx.fxs:
        clip = eval('clip.%s' % fx_param)

    # 位置函数相关的实现
    pos_proxy = PositionProxy()
    for start, func, paddle in tpl.position.positions:
        pos_proxy.add(start, func, paddle)
    pos_proxy.set_clip(clip)
    clip = clip.set_position(lambda t: pos_proxy.get_pos_func(t))

    return [clip]


class PosFunc(object):
    """
    posfunc
    """
    def __init__(self, start, func, paddle):
        self.start = start
        self.func = func
        self.paddle = paddle


class PositionProxy(object):
    """
    clip位置相关的函数具象化
    """
    def __init__(self):
        self.positions_tpl = []
        self.positions = []
        self.is_sorted = False
        self.last_idx = 0
        self.clip = None

    def add(self, t, pos_func, paddle):
        """
        :param t:
        :param pos_func:
        :return:
        """
        self.positions_tpl.append(PosFunc(t, pos_func, paddle))
        self.is_sorted = False
        self.last_idx = 0

    def set_clip(self, clip):
        """
        :param clip:
        :return:
        """
        self.clip = clip

    def get_pos_func(self, t):
        """
        如果位置是一个数组的话，根据时间排序，分别实现位置函数
        :param t:
        :return:
        """
        height, width = self.clip.get_frame(t).shape[0: 2]
        if not self.is_sorted:
            self.positions_tpl = sorted(self.positions_tpl, key=lambda x: x.start)
            self.is_sorted = True
            self.positions = copy.deepcopy(self.positions_tpl)
            self.positions.append(PosFunc(99999, None, ''))

            for i, position in enumerate(self.positions[0: -1]):
                f_str = position.func
                position.func = eval(f_str)

        if self.positions[0].start != 0:
            self.positions[0].start = 0
            logging.warning('clip position should start from 0')

        if self.positions[self.last_idx].start <= t < self.positions[self.last_idx + 1].start:
            func = self.positions[self.last_idx].func
            start = self.positions[self.last_idx].start
            x, y = func(t - start)
            if self.positions[self.last_idx].paddle == 'center':
                x, y = x - width / 2, y - height / 2
            return x, y

        for i, pos in enumerate(self.positions):
            if pos.start <= t < self.positions[i + 1].start:
                self.last_idx = i
                func = pos.func
                start = pos.start
                x, y = func(t - start)
                if pos.paddle == 'center':
                    x, y = x - width / 2, y - height / 2
                return x, y
