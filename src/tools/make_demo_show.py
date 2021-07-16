# -*- coding: utf-8 -*-
"""
上传到testbos，并制作简易展示页面

Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/01 09:10:06
"""

import os
import sys
sys.path.append('../')
from utils import bos_conf


if __name__ == '__main__':
    root_dir = '../../demo'
    dir_list = [root_dir]
    file_list = []
    idx = 0
    cate_name = {'huitu_fade': '换场动画',
                 'huitu_vfx': "特效",
                 'movement': '运动方案',
                 'product': '产品示例',
                 'weidong': '微动特效'}
    while idx < len(dir_list):
        cur_dir = dir_list[idx]
        idx += 1
        for fname in os.listdir(cur_dir):
            cur_path = cur_dir + '/' + fname
            if os.path.isdir(cur_path):
                dir_list.append(cur_path)
            else:
                file_list.append(cur_path)

    name_url = {}
    for fname in file_list:
        show_name = fname[len(root_dir) + 1:]
        # prepare prefix
        prefix = 'to be continue'
        show_url = prefix + show_name
        name_url[show_name] = show_url

    video_content = ''
    with open('demo_show_video_template') as f:
        video_content = f.read()

    content = ''
    num = 0
    table_name = ''
    idx = 0
    with open('../../demo/list.txt') as f:
        for line in f:
            num += 1
            name, desc = line.strip().replace('\t', ' ').split(' ')
            desc = desc.decode('gb18030').encode('utf-8')
            url = name_url[name]

            tname = name.split('/')[0]
            if tname != table_name:
                if table_name != '':
                    content += '\n</tr>'
                    content += '\n</table></p>'
                content += ('<h1>' + cate_name[tname] + '</h1>\n</p><table border="4">')
                idx = 0
                table_name = tname

            if idx % 3 == 0:
                if idx > 0:
                    content += '\n</tr>'
                content += '\n<tr>'
            vc = video_content
            vc = vc.replace('__url1__', url)
            vc = vc.replace('__url2__', url)
            vc = vc.replace('__desc__', desc)
            vc = vc.replace('__id__', str(num))
            content += ('\n<td>' + vc + '</td>')
            idx += 1
        content += '\n</tr>'
        content += '\n</table></p>'

    with open('demo_show_template.html') as f:
        html = f.read()

    html = html.replace('__content__', content)
    with open('../../demo/video_demo.html', 'w') as f:
        f.write(html)
