#! -*- coding: utf-8 -*-


# 导入Python标准日志模块
import logging

#从Python SDK导入BOS配置管理模块以及安全认证模块
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials

# 设置BosClient的Host，Access Key ID和Secret Access Key
bos_host = "bos.qasandbox.bcetest.baidu.com"
access_key_id = "61e13d99eee7412aa9ad76d81d783b22"
secret_access_key = "3e18d91aecad45dabb02ac6fb6bd99d6"

# bos_host = "su.bcebos.com"
# access_key_id = "d64517808fed4709a06a5b3336bdd411"
# secret_access_key = "61a123d8d8f343d1972c55184a4afecd"

# 设置日志文件的句柄和日志级别
# logger = logging.getLogger('baidubce.http.bce_http_client')
# fh = logging.FileHandler("../log/bos.log")
# fh.setLevel(logging.FATAL)

# 设置日志文件输出的顺序�1�7�结构和内容
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# logger.setLevel(logging.FATAL)
# logger.addHandler(fh)

# 创建BceClientConfiguration
config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)
