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

import json
import logging
import video_clip
import audio_clip


class VideoTemplate(object):
    """
    video tempalte, include video clip, image_clip, audio clip,
    模板上层配置。
    """
    def __init__(self, fpath, jsn_str=None):
        self.invalid = True
        if jsn_str is None:
            with open(fpath) as f:
                jsn_str = f.read()
        try:
            jsn = json.loads(jsn_str, encoding='utf-8')
        except:
            logging.fatal(jsn_str)
            return

        self.size = jsn.get('size', None)
        if self.size is None:
            logging.fatal('expect size')
            return

        self.height = self.size.get('height', None)
        if self.height is None:
            return
        self.width = self.size.get('width', None)
        if self.width is None:
            return

        self.color = jsn.get('color', [0, 0, 0])

        self.video_aspect_threshold = jsn.get('video_aspect_threshold', None)

        self.fps = jsn.get('fps', 24)
        self.brate = jsn.get('brate', 2400)

        self.duration = jsn.get('duration', None)
        if self.duration is None:
            logging.fatal('expect duration')
            return

        self.video_clips = []
        last_vc = None
        for jsn_vc in jsn.get('videoclips', []):
            vc = video_clip.VideoClip(jsn_vc, last_vc)
            if vc.is_invalid():
                logging.fatal('load video clip failed')
                return
            self.video_clips.append(vc)
            last_vc = vc
        if len(self.video_clips) == 0:
            return

        self.audio_clips = []
        for jsn_ac in jsn.get('audioclips', []):
            ac = audio_clip.AudioClip(jsn_ac)
            if ac.is_invalid():
                logging.fatal('load audio clip failed')
                return
            self.audio_clips.append(ac)

        self.invalid = False

    def is_invalid(self):
        """
        :return:
        """
        return self.invalid
