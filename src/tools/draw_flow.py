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
import cv2 as cv
from flow import Flow

#events = [i for i in dir(cv) if 'EVENT' in i]
#print events

rec_x0 = rec_y0 = rec_x1 = rec_y1 = 0

def draw_flow(event, x, y, flags, param):
    """
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    global rec_x0, rec_y0, rec_x1, rec_y1
    global img
    if event == cv.EVENT_LBUTTONDOWN:
        rec_x0 = x
        rec_y0 = y
        img[y, x] = [0, 0, 0]
    if event == cv.EVENT_LBUTTONUP:
        rec_x1 = x
        rec_y1 = y
        img[y - 1: y + 2, x - 1: x + 2] = [0, 0, 0]
        if rec_x0 == rec_x1 and rec_y0 == rec_y1:
            return None
        cv.line(img, (rec_x0, rec_y0), (rec_x1, rec_y1), (0, 0, 0), 1)
        flow.add_flow(rec_x0, rec_y0, rec_x1, rec_y1)
    if event == cv.EVENT_RBUTTONDOWN:
        res = flow.remove_by_endpoint(x, y)
        if res is None:
            return
        x0, y0, x1, y1 = res
        cv.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        img[y1 - 1: y1 + 2, x1 - 1: x1 + 2] = [0, 0, 255]


if __name__ == '__main__':
    img_path = '../../output/weidong/cosmestic04.jpg'
    flow = Flow(img_path)
    if flow.is_invalid():
        print 'init flow failed'
        sys.exit()

    img = flow.get_img()
    cv.namedWindow('image', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.setMouseCallback('image', draw_flow)
    while(1):
        cv.imshow('image', img)
        if cv.waitKey(10) & 0xFF == 27:
            break
    cv.destroyAllWindows()

    flow.save()
