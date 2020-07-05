# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: liangweiming(liangweiming@baidu.com)
Date:    2019/05/16 17:43:06
"""
import sys
import math
import cv2 as cv
import numpy as np
from video_effect import movement
import time
from moviepy.video.tools import drawing


class RhombusMask(object):
    """
    rhombus mask
    """
    def __init__(self, duration, anchor, win_len):
        self.full_mask = None
        self.width = 0
        self.height = 0
        self.start_pos = None  # (x, y)
        self.end_pos = None  # (x, y)
        self.move_func = None
        self.duration = duration
        self.anchor = anchor
        self.win_len = win_len

    def create_full_mask(self, height, width):
        """
        create full mask
        :param height:
        :param width:
        :return:
        """
        size = (width, height)
        ang = math.pi * 0.40
        if self.anchor == 'A-left':
            xc1, yc1 = (0, 0)
            vec1 = (math.cos(ang), math.sin(ang))  # (x_vec, y_vec)
            # Region 1 is then the region on the left when starting
            # in position p1 and going in the direction given by vector.
            im1 = drawing.color_split(size, p1=(xc1, yc1), vector=vec1, col1=1.0, col2=0.0)
            xc2, yc2 = (self.win_len, 0)
            vec2 = (math.cos(ang), math.sin(ang))  # (x_vec, y_vec)
            im2 = drawing.color_split(size, p1=(xc2, yc2), vector=vec2, col1=0.0, col2=1.0)
            self.start_pos = (xc2 + height * 1.0 / vec2[1] * vec2[0] + width, 0)
            self.end_pos = (0, 0)
        elif self.anchor == 'A-right':
            xc1, yc1 = (width, 0)
            vec1 = (-math.cos(ang), math.sin(ang))  # (x_vec, y_vec)
            im1 = drawing.color_split(size, p1=(xc1, yc1), vector=vec1, col1=0.0, col2=1.0)
            xc2, yc2 = (width - self.win_len, 0)
            vec2 = (-math.cos(ang), math.sin(ang))  # (x_vec, y_vec)
            im2 = drawing.color_split(size, p1=(xc2, yc2), vector=vec2, col1=1.0, col2=0.0)
            self.start_pos = (width - height * math.cos(ang) / math.sin(ang) - self.win_len,
                              0)
            self.end_pos = (width * 2, 0)
        elif self.anchor == 'B-left':
            xc1, yc1 = (0, height)
            vec1 = (math.cos(ang), -math.sin(ang))  # (x_vec, y_vec)
            im1 = drawing.color_split(size, p1=(xc1, yc1), vector=vec1, col1=0.0, col2=1.0)
            xc2, yc2 = (self.win_len, height)
            vec2 = (math.cos(ang), -math.sin(ang))  # (x_vec, y_vec)
            im2 = drawing.color_split(size, p1=(xc2, yc2), vector=vec2, col1=1.0, col2=0.0)
            self.start_pos = (self.win_len + height / vec2[1] * vec2[0] + width, 0)
            self.end_pos = (0, 0)
        elif self.anchor == 'B-right':
            xc1, yc1 = (width, height)
            vec1 = (-math.cos(ang), -math.sin(ang))  # (x_vec, y_vec)
            im1 = drawing.color_split(size, p1=(xc1, yc1), vector=vec1, col1=1.0, col2=0.0)
            xc2, yc2 = (width - self.win_len, height)
            vec2 = (-math.cos(ang), -math.sin(ang))  # (x_vec, y_vec)
            im2 = drawing.color_split(size, p1=(xc2, yc2), vector=vec2, col1=0.0, col2=1.0)
            self.start_pos = (width - height * math.cos(ang) / math.sin(ang) - self.win_len, 0)
            self.end_pos = (width * 2, 0)

        self.move_func = movement.decel(self.start_pos[0],
                                        self.start_pos[1],
                                        self.end_pos[0],
                                        self.end_pos[1],
                                        self.duration,
                                        str='circle')

        mask = np.zeros(shape=(height, width * 3), dtype=np.float)
        mask[0: height, width: width * 2] = im1 * im2
        self.full_mask = cv.merge((mask, mask, mask))
        self.height = height
        self.width = width
        return None

    def resize_full_mask(self, height, width):
        assert(1 == 0)
        return None

    def get_mask(self, t, height, width):
        """
        get mask by time
        :param t:
        :param height:
        :param width:
        :return:
        """
        if self.full_mask is None:
            self.create_full_mask(height, width)
        if self.width != width or self.height != height:
            self.resize_full_mask(height, width)

        x, y = self.move_func(t)
        x, y = int(x), int(y)

        return self.full_mask[y: y + height, x: x + width]
