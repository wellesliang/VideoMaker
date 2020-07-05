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

class AudioClip(object):
    """
    audio clip
    """
    def __init__(self, jsn):
        self.invalid = True
        self.name = ''
        self.type = jsn.get('type', None)
        if self.type is None:
            return

        self.url = jsn.get('url', None)
        if self.url is None:
            return

        self.invalid = False

    def is_invalid(self):
        """
        :return:
        """
        return self.invalid
