"""
get commedity information, such as catogory, name, image, ...

Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/03/05 17:23:06
"""

import cv2 as cv
import numpy as np
import urllib
import socket
import traceback


def hist_equalization(img, blocksize=8):
    """
    histogram_equalization by hsv color space
    """
    if img.shape[0] * img.shape[1] == 0:
        return img
    
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    clahe = cv.createCLAHE(clipLimit=1.8, tileGridSize=(blocksize, blocksize))
    s_equ = clahe.apply(hsv[0:, 0:, 1])
    v_equ = clahe.apply(hsv[0:, 0:, 2])
    #s_equ = cv.equalizeHist(hsv[0:, 0:, 1])
    #v_equ = cv.equalizeHist(hsv[0:, 0:, 2])
    new_hsv = cv.merge((hsv[0:, 0:, 0], s_equ, v_equ))
    new_img = cv.cvtColor(new_hsv, cv.COLOR_HSV2BGR)
    return new_img


def get_img(path, o='bgr'):
    """
    :param path:
    :return: load img from local disk or url by path
    """
    if path is None:
        return None

    if path[0: 4] == 'http':
        img = get_img_from_url(path)
    else:
        img = get_img_from_file(path)
    if img is not None and o == 'rgb':
        alpha = None
        if img.shape[2] == 4:
            alpha = img[0:, 0:, 3]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if alpha is not None:
            img = np.concatenate((img, alpha[0:, 0:, np.newaxis]), axis=2)
    return img


def get_img_from_url(url):
    """
    :param url:
    :return: ndarrar, or None if connect error or url error
    """
    try:
        socket.setdefaulttimeout(10)
        resp = urllib.request.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv.imdecode(img, cv.IMREAD_UNCHANGED)
    except:
        img = None
        print('cannot open url %s' % url)
        traceback.print_exc()
    return img


def get_img_from_file(path):
    """
    :param path:
    :return: ndarray, or None if file not found or file error
    """
    return cv.imread(path, cv.IMREAD_UNCHANGED)
