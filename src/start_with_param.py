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

import re
import os
import sys
import random
import traceback
import video_template
import video_compositer


class Music(object):
    """
    music proxy
    """
    def __init__(self, music_dir):
        self.music_list = []
        for fname in os.listdir(music_dir):
            if fname[-4:] == '.mp3':
                self.music_list.append(music_dir + '/' + fname)

    def get_rand_music_file(self):
        """
        return a random music
        :return:
        """
        i = random.randint(0, len(self.music_list) - 1)
        return self.music_list[i]


class Fade(object):
    """
    fade proxy
    """
    def __init__(self, fade_file):
        self.fade_list = []
        with open(fade_file) as f:
            for line in f:
                self.fade_list.append(line.rstrip().split('\t')[0])

    def get_rand_fade_name(self):
        """
        return rand fade name
        :return:
        """
        i = random.randint(0, len(self.fade_list) - 1)
        return self.fade_list[i]


def set_demo_env_args(fname):
    """
    """
    with open(fname) as f:
        for line in f:
            name, value = line.rstrip().split('\t')[0: 2]
            os.environ[name] = value


def get_template_args(str):
    """
    find patten __ARG_*__, replaceimage，text，music
    :param str:
    :return:
    """
    pat = re.compile('__ARG_[\w]+__')
    all_pats = set(pat.findall(jsn_str))
    all_pats = [(pat_name, pat_name[2: -2]) for pat_name in all_pats]
    return all_pats


if __name__ == '__main__':
    template_dir = '../template/'
    resource_dir = '../resource/'

    if len(sys.argv) >= 2:
        template_name = sys.argv[1]
        verbose = False
    else:
        sub_dir = 'xianyu_20200704'
        tpl_fname = 'vert_1.json'
        #tpl_fname = 'vert_2_ending.json'
        template_name = sub_dir + '/' + tpl_fname
        set_demo_env_args(template_dir + sub_dir + '/demo/demo_args1.txt')
        verbose = True

    if not os.path.isfile(template_dir + template_name):
        print(f'Not found {template_dir}/{template_name}')
        sys.exit(1)
    music_pr = Music(resource_dir + '/audios/')
    fade_pr = Fade(resource_dir + '/fade/fade_list.txt')
    template_name = template_dir + template_name

    vc = video_compositer.VideoCompositer('thread1')

    # template should be utf-8
    with open(template_name) as f:
        jsn_str = f.read()
    for pat_name, arg_name in get_template_args(jsn_str):
        value = os.getenv(arg_name)
        if value is not None:
            try:
                value = value
            except:
                traceback.print_exc()
                sys.exit(3)
        if value is None:
            if arg_name[0: 9] == 'ARG_music':
                value = music_pr.get_rand_music_file()
            elif arg_name[0: 8] == 'ARG_fade':
                value = fade_pr.get_rand_fade_name()
            else:
                sys.exit(2)

        jsn_str = jsn_str.replace(pat_name, value)

    vt = video_template.VideoTemplate(None, jsn_str)
    if vt.is_invalid():
        sys.exit(4)

    out_name = os.getenv('RESULT_VIDEO_PATH')
    out_name = template_name.replace('.json', '.mp4') if out_name is None else out_name
    ogv_name = template_name.replace('.json', '.ogv')
    video = vc.composit(vt)
    if video is None:
        sys.exit(5)
    video.save(out_name, verbose=verbose)
    # video.save(ogv_name, verbose=verbose, progress_bar=progress_bar)
    # video.save_multi(out_name, ogv_name, verbose=verbose, progress_bar=progress_bar)

