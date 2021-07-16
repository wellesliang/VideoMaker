# -*- coding: utf-8 -*-
"""
Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/21 11:55:06
"""

import logging
import traceback
import moviepy.video.fx.all as vfx
from video_effect import movement
from video_effect import huitu_vfx


class VideoClip(object):
    """
    根据配置，进行视频/图像/文字等相关的clip合成设置
    """
    def __init__(self, jsn, last_vc=None):
        self.invalid = True
        self.name = jsn.get('name', '')
        # clip类型，image，video，text
        self.type = jsn.get('type', None)
        if self.type is None:
            logging.fatal('clip %s type expected', self.name)
            return

        # 视频或者图片的url或者本地地址，会自动判断下载或加载到本地
        self.url = jsn.get('url', None)
        if self.url is None:
            logging.fatal('clip %s type expected', self.name)
            return

        self.text = jsn.get('text', '')
        self.size = jsn.get('size', '')
        self.font = jsn.get('font', '')
        self.color = jsn.get('color', '')

        # 对视频进行剪切用，如取某个视频的第2秒到第4秒
        self.subclip = jsn.get('subclip', None)
        if self.subclip is not None:
            self.subclip = [self.subclip['begin'], self.subclip['end']]

        # 当前clip持续时间，描述方法很多，可以通过begin，end描述，也可以通过begin，duration描述；
        # 也可以基于上一个视频的last_begin, last_end描述，同时可以设置delta，表示基于begin的相对时间增减
        self.duration = jsn.get('duration', None)
        if self.duration is not None:
            if self.duration['begin'] in ['last_begin', 'last_end']\
                    or self.duration.get('end', None) in ['last_begin', 'last_end']:
                begin = self.duration['begin']
                end = self.duration.get('end', None)
                delta = self.duration.get('delta', None)
                dura = self.duration.get('duration', None)
                self.duration = [begin, end, delta, dura]
            elif 'begin' in self.duration and 'end' in self.duration:
                self.duration = [self.duration['begin'], self.duration['end']]
            elif 'begin' in self.duration and 'duration' in self.duration:
                begin = self.duration['begin']
                end = begin + self.duration['duration']
                self.duration = [begin, end]
            else:
                print(jsn['name'])
                print(self.duration)
                assert(1 == 0)

        # 裁剪相关设置
        self.crop = jsn.get('crop', None)
        if self.crop is not None:
            if 'x1' not in self.crop or 'y1' not in self.crop:
                logging.fatal('clip %s crop x1 and y1 expected', self.name)
                return
            if 'width' not in self.crop or 'height' not in self.crop:
                logging.fatal('clip %s crop width and height expected', self.name)
                return
            x1 = self.crop['x1']
            x2 = x1 + self.crop['width']
            y1 = self.crop['y1']
            y2 = y1 + self.crop['height']
            self.crop = [x1, y1, x2, y2]

        # 位置控制，位置是一个数组，每个数组元素，描述了一个位置动作，注意是动作。
        # 对一个元素：你可以设置在begin到end之间，位置基于t的函数
        self.position = VideoClipPosition(jsn.get('position', []))
        if self.position.is_invalid():
            logging.fatal('clip %s position is invalid', self.name)
            return

        # 特效设置，特效是一个数组，依次进行渲染，可以配置vfx.XXXX，表示调用系统特效，
        # 也可以配置vfx_huitu.XXXX表示调用自定义设置，详细请见特效class
        self.fx = VideoClipFx(jsn.get('fx', []))
        if self.fx.is_invalid():
            logging.fatal('clip %s fx is invalid', self.name)
            return

        # mask相关设置
        if 'mask' in jsn:
            self.mask = VideoClipMask(jsn['mask'])
            if self.mask.is_invalid():
                logging.fatal('clip %s mask is invalid', self.name)
                return
        else:
            self.mask = None

        self.invalid = False

    def is_invalid(self):
        """
        :return:
        """
        return self.invalid


class VideoClipPosition(object):
    """
    clip 位置相关的设置，需要注意：位置可以设置center，top，bottom，等默认的设置，位置也可以是基于t的函数。
    建议参看典型配置，理解此段代码
    """
    def __init__(self, jsn):
        self.invalid = True
        self.positions = []
        for pos in jsn:
            if 'start' not in pos:
                logging.fatal('parse failed in VideoClipPosition: start')
                return
            if 'value' not in pos:
                logging.fatal('parse failed in VideoClipPosition: value')
                return
            try:
                # 如果配置中带括号，可能是函数类似"move(a,b,c)"，也可能是(center,center)，要区分对待
                func_name = pos['value'].strip().split('(')[0]
                if func_name == '':
                    s = pos['value'].replace('center', '"center"').\
                        replace('top', '"top"').replace('bottom', '"bottom"').\
                        replace('left', '"left"').replace('right', '"right"')
                    func = 'lambda t: %s' % s
                else:
                    func = pos['value']
            except:
                logging.fatal('parse failed in VideoClipPosition: eval value')
                traceback.print_exc()
                return

            # 对齐方式有两种，中心对齐和左上角对齐
            paddle = pos.get('paddle', '')
            self.positions.append((pos['start'], func, paddle))

        if len(self.positions) <= 0:
            return

        self.invalid = False

    def is_invalid(self):
        """
        :return:
        """
        return self.invalid


class VideoClipFx(object):
    """
    视频相关特效的配置解析，需要注意的是，vfx.XXX表示系统自定义特效，
    vfx_huitu.XXX表示自定义特效，自定义特效在vdieo_effect下
    """
    def __init__(self, jsn):
        self.invalid = True
        self.fxs = []

        for fx in jsn:
            if 'func' not in fx:
                logging.fatal('parse failed in VideoClipFx: func')
                return

            func_name = fx['func']
            if func_name in ['vfx.crossfadein', 'vfx.crossfadeout']:
                duration = 0
                for param in fx.get('params', []):
                    if param['key'] == 'duration':
                        duration = param['value']
                func_name = func_name.split('.')[1]
                params = func_name + '(%s)' % duration
                self.fxs.append(params)
            else:
                try:
                    func = eval(func_name)
                except:
                    logging.fatal('parse failed in VideoClipFx: eval func, %s' % fx['func'])
                    return
                params = [func_name]
                # key value对参数解析成相应的值，主要是处理字符串和lambda等函数
                for param in fx.get('params', []):
                    if isinstance(param['value'], str) \
                            and param['value'][0: 6] != 'lambda' \
                            and param['value'][0: 1] != '(':
                        v = '"' + param['value'] + '"'
                    elif isinstance(param['value'], str):
                        v = param['value'].replace('center', '"center"')
                    else:
                        v = str(param['value'])
                    params.append(str(param['key']) + ' = ' + v)
                params = 'fx(%s)' % ', '.join(params)
                self.fxs.append(params)

        self.invalid = False

    def is_invalid(self):
        return self.invalid


class VideoClipMask(object):
    """
    clip mask相关配置及解析
    """
    def __init__(self, jsn):
        self.invalid = True
        self.url = jsn.get('url', None)
        if self.url is None:
            logging.fatal('parse failed in VideoClipMask: url')
            return

        self.crop = jsn.get('crop', None)
        if self.crop is not None:
            if 'x1' not in self.crop or 'y1' not in self.crop:
                logging.fatal('clip mask %s crop x1 and y1 expected', self.name)
                return
            if 'width' not in self.crop or 'height' not in self.crop:
                logging.fatal('clip mask %s crop width and height expected', self.name)
                return
            x1 = self.crop['x1']
            x2 = x1 + self.crop['width']
            y1 = self.crop['y1']
            y2 = y1 + self.crop['height']
            self.crop = [x1, y1, x2, y2]

        self.position = jsn.get('position', None)
        if self.position is not None:
            if 'x1' not in self.position or 'y1' not in self.position:
                logging.fatal('clip mask %s position x1 and y1 expected', self.name)
                return
            if 'width' not in self.position or 'height' not in self.position:
                logging.fatal('clip mask %s position width and height expected', self.name)
                return
            x1 = self.position['x1']
            x2 = x1 + self.position['width']
            y1 = self.position['y1']
            y2 = y1 + self.position['height']
            self.position = [x1, y1, x2, y2]

        self.invalid = False

    def is_invalid(self):
        """
        :return:
        """
        return self.invalid
