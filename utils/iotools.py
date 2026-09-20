# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from PIL import Image, ImageFile#图像读取、处理
import errno #标准错误码
import json
import pickle as pkl#二进制文件序列反序列化
import os
import os.path as osp
import yaml
from easydict import EasyDict as edict #字典的key可以通过属性访问

ImageFile.LOAD_TRUNCATED_IMAGES = True#允许 PIL 读取截断 / 不完整的图片文件


def read_image(img_path):#鲁棒读取图像
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def mkdir_if_missing(directory):#安全创建目录
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):#验证指向的文件是否存在
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)


def save_train_configs(path, args):#将训练的所有配置参数保存到YAML文件中。
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def load_train_configs(path):#从YAML文件加载配置。
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)