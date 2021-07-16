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
import csv
import random
import yaml
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
    在本地调试的时候，模拟线上调用设置环境参数，参数从文件读取
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
    all_pats = [(pat_name, pat_name[6: -2]) for pat_name in all_pats]
    return all_pats


def load_item_from_file(filenames, filter):
    items = []
    for filename in filenames:
        with open(filename, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if 'image_num' in filter and len(row['PICTURE_URL'].split(';')) < filter['image_num']:
                    continue
                item = {'item_id': row['item_id'],
                        'title': row['AUCT_TITL'],
                        'first': row['META_CATEG_NAME'],
                        'second': row['CATEG_LVL2_NAME'],
                        'third': row['CATEG_LVL3_NAME']}
                for i, url in enumerate(row['PICTURE_URL'].split(';')):
                    item[f'image{i+1}'] = url
                items.append(item)
    return items


def load_item_from_demo(demo):
    return demo


def load_item(conf, return_num=1):
    if 'demo' in conf and 'used' in conf['demo'] and conf['demo']['used'] is True:
        items = [load_item_from_demo(conf['demo'])]
    else:
        items = load_item_from_file(conf['item']['path'], conf['item'])

    random.shuffle(items)
    return items[0: return_num]


if __name__ == '__main__':
    conf_name = '../conf/invent.yaml'
    with open(conf_name) as f:
        conf = yaml.load(f)

    resource_dir = conf['resource']['path']
    music_pr = Music(resource_dir + '/audios/')
    fade_pr = Fade(resource_dir + '/fade/fade_list.txt')

    for item in load_item(conf, return_num=1):
        for template_path in conf['templates']:
            # template should be utf-8
            with open(template_path) as f:
                jsn_str = f.read()
            for pat_name, arg_name in get_template_args(jsn_str):
                value = item.get(arg_name, None)
                if value is None:
                    if arg_name[0: 5] == 'music':
                        value = music_pr.get_rand_music_file()
                    elif arg_name[0: 4] == 'fade':
                        value = fade_pr.get_rand_fade_name()
                    else:
                        raise Exception(f'Unknown argname: {pat_name}')
                jsn_str = jsn_str.replace(pat_name, value)

            vt = video_template.VideoTemplate(None, jsn_str)
            if vt.is_invalid():
                raise Exception(f'Load template: {template_path} failed')

            # 生成视频。常规生成mp4格式；如果用于网络流播放，生成ogv格式。
            # 如果用于双路输出，调用save_multi
            out_name = template_path.replace('.json', '.mp4')
            ogv_name = template_path.replace('.json', '.ogv')
            vc = video_compositer.VideoCompositer('thread1')
            video = vc.composit(vt)
            if video is None:
                raise Exception(f'Video compositer failed, template: {vt.name}')
            video.save(out_name, verbose=True)
            # video.save(ogv_name, verbose=verbose, progress_bar=progress_bar)
            # video.save_multi(out_name, ogv_name, verbose=verbose, progress_bar=progress_bar)

