# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
辅助画微动轨迹

Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/01 09:10:06
"""

import os
import sys
import math
import pickle
import cv2 as cv
import numpy as np
sys.path.append('../')
from utils import math_util


weighted_avg = lambda v1, r1, v2, r2: v1 * r1 / (r1 + r2) + v2 * r2 / (r1 + r2)


class Flow(object):
    """
    移动指示线
    """
    matrix_suffix = '.matrix'
    flow_suffix = '.flow'

    def __init__(self, img_path):
        self.invalid = True
        self.img_path = img_path

        self.img = img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        if img is None:
            print 'load img failed %s' % img_path
            return

        self.flows = set()
        self.flows_dict = dict()
        self.load_flows()
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.mx = np.zeros(shape=(self.height, self.width), dtype='float')
        self.my = np.zeros(shape=(self.height, self.width), dtype='float')
        self.m_move_len = np.zeros(shape=(self.height, self.width), dtype='float')
        self.m_move_r = np.zeros(shape=(self.height, self.width), dtype='float')
        self.dist_pow_thre = 35 * 35
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.invalid = False

    def is_invalid(self):
        """
        :return:
        """
        return self.invalid

    def get_img(self):
        """
        :return:
        """
        return self.img

    def add_flow(self, x0, y0, x1, y1):
        """
        :param x0:
        :param y0:
        :param x1:
        :param y1:
        :return:
        """
        self.flows.add((x0, y0, x1, y1))
        for dx, dy in [(-1, -1), (0, -1), (1, -1),
                       (-1, 0), (0, 0), (1, 0),
                       (-1, 1), (0, 1), (1, 1)]:
            nx = dx + x1
            ny = dy + y1
            self.flows_dict[(nx, ny)] = (x0, y0, x1, y1)

    def load_flows(self):
        """
        :return:
        """
        line_color = (255, 0, 0)
        line_thick = 3
        point_color = (0, 0, 255)
        point_radius = 5
        if not os.path.exists(self.img_path + Flow.flow_suffix):
            return
        with open(self.img_path + Flow.flow_suffix) as f:
            for line in f:
                x0, y0, x1, y1 = line.strip().split('\t')
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                self.add_flow(x0, y0, x1, y1)
                cv.line(self.img, (x0, y0), (x1, y1), line_color, line_thick)
                self.img[y0, x0] = [0, 0, 0]
                self.img[y1 - point_radius: y1 + point_radius + 1, \
                        x1 - point_radius: x1 + point_radius + 1] = point_color

    def calc_flow_matrix(self):
        """
        :return:
        """
        bit = np.zeros(shape=(self.height, self.width), dtype='uint8')
        for tmp_i, (x0, y0, x1, y1) in enumerate(list(self.flows)):
            print '\rprocessing %d/%d' % (tmp_i + 1, len(self.flows)),
            sys.stdout.flush()
            MOVE_LEN = l2 = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            xl = (x1 - x0) / l2
            yl = (y1 - y0) / l2

            bit[0:, 0:] = 0
            bfs = [(x0, y0)]
            bit[y0, x0] = 1
            idx = 0
            while idx < len(bfs):
                x, y = bfs[idx]
                idx += 1

                dist_pow = math_util.distance_pt_seg((x, y), (x0, y0), (x1, y1), True)
                if dist_pow > self.dist_pow_thre:
                    continue
                # r = 1 / (r ** 2 + 0.1)
                r = 1.0 / (dist_pow / self.dist_pow_thre + 0.1)
                self.mx[y, x] += xl * r
                self.my[y, x] += yl * r

                # r = 1 - (r ** 4)
                move_len_r = 1.0 - ((dist_pow / self.dist_pow_thre) ** 2)
                move_len = MOVE_LEN * move_len_r
                self.m_move_len[y, x] = \
                    weighted_avg(self.m_move_len[y, x], self.m_move_r[y, x], move_len, r)
                self.m_move_r[y, x] += r

                for dx, dy in self.directions:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and bit[ny, nx] == 0:
                        bit[ny, nx] = 1
                        bfs.append((nx, ny))
        print 'all finished'
        ml2 = np.sqrt(self.mx * self.mx + self.my * self.my)
        ml2 = np.where(ml2 < 0.01, 999999999.0, ml2)
        self.mx = self.mx / ml2 * self.m_move_len
        self.my = self.my / ml2 * self.m_move_len

    def get_flow_matrix(self):
        """
        :return:
        """
        self.calc_flow_matrix()
        return self.mx, self.my

    def save_flows(self, out_name):
        """
        :param out_name:
        :return:
        """
        with open(out_name, 'w') as f:
            for x0, y0, x1, y1 in list(self.flows):
                f.write('%d\t%d\t%d\t%d\n' % (x0, y0, x1, y1))

    def save_matrix(self, out_name):
        """
        :param out_name:
        :return:
        """
        self.calc_flow_matrix()
        with open(out_name, 'wb') as f:
            pickle.dump((self.mx, self.my), f)

    def save(self, out_prefix=None):
        """
        :param out_prefix:
        :return:
        """
        if out_prefix is None:
            out_prefix = self.img_path
        self.save_flows(out_prefix + Flow.flow_suffix)
        #self.save_matrix(out_prefix + Flow.matrix_suffix)

    def remove_by_endpoint(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        res = self.flows_dict.get((x, y), None)
        if res is None:
            return None
        x0, y0, x1, y1 = res
        self.flows.remove((x0, y0, x1, y1))
        return x0, y0, x1, y1
