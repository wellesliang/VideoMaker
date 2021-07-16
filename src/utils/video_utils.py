"""

Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/03/05 17:23:06
"""
import moviepy.editor as mpy
import cv2 as cv
import os
import sys
import traceback
import requests


def get_video_local_path(path_url, dst_local_name):
    """
    local path or url, if url, save to local path first
    :param path_url:
    :param dst_local_name:
    :return:
    """
    if path_url[0: 4] != 'http':
        if os.path.isfile(path_url):
            return path_url
        else:
            return None

    path_url = path_url.split('?')[0]
    ext_name = path_url.split('.')[-1]
    if ext_name not in ['mp4', 'avi', 'wmv', 'ogv', 'mpeg', 'mov']:
        ext_name = 'mp4'
    local_name = dst_local_name + '.' + ext_name
    try:
        r = requests.get(path_url, timeout=60.0)
        with open(local_name, 'wb') as f:
            f.write(r.content)
    except:
        local_name = None
        traceback.print_exc()

    return local_name


def get_frames(video, num):
    """
    :param video:
    :param num:
    :return:
    """
    if not isinstance(video, mpy.VideoClip):
        clip = mpy.VideoFileClip(video)
    else:
        clip = video
    d = clip.duration
    seg = d * 1.0 / (num + 1)
    res = []
    for i in range(num):
        img = clip.get_frame(seg * (i + 1))
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        res.append(img)
    return res


if __name__ == '__main__':
    url = 'http://vd3.bdstatic.com/mda-iavtzcd8u54vzvz6/mda-iavtzcd8u54vzvz6.mp4'
    fname = get_video_local_path(url, '../../output/tttt')
    if fname is None:
        print('load video failed')
        sys.exit(1)
    imgs = get_frames(fname, 10)
    for i, img in enumerate(imgs):
        cv.imwrite('../../output/%d.jpg' % i, img)
