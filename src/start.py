# -*- coding: utf-8 -*-
"""
Authors: liangweiming(liangweiming@baidu.com)
Date:    2018/06/21 11:55:06
"""

import sys
import video_template
import video_compositer


if __name__ == '__main__':
    vc = video_compositer.VideoCompositer('thread1')
    #conf_name = '../output/junjie_jd_shouji/horz2vert.json'
    #conf_name = '../template/test/vert2horz.json'
    conf_name = '../output/qiuyan_qiche/ppt_qiche.json'
    #conf_name = '../template/test/fast_flash_demo1.json'
    #conf_name = '../template/test/qiuyan_vip_hufu.json'
    #conf_name = '../template/test/text.json'
    #conf_name = '../template/test/conf.json'

    conf_name_list = [conf_name]
    for conf_name in conf_name_list:
        vt = video_template.VideoTemplate(conf_name)
        if vt.is_invalid():
            sys.exit(1)

        video = vc.composit(vt)
        if video is not None:
            out_name = conf_name.split('/')[-1].split('.')[0] + '.mp4'
            out_file = conf_name[0: -5] + '.mp4'
            # out_file = '../output/' + out_name
            video.save(out_file)

            # if False:
            #     bos_client = BosClient(bos_conf.config)
            #     bos_client.put_object_from_file('pa-test', out_name, out_file)
            #     print('http://bos.qasandbox.bcetest.baidu.com/pa-test/' + out_name)
