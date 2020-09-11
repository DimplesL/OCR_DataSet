# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@license: (C) Copyright 2017-2025, Personal Projects Limited.
@contact: qiuyurui21@163.com
@software: Pycharm
@file: labelme2json.py
@time: 2020-09-11 16:36
@desc:
"""
import pathlib
import os
import sys
import random
import json

sys.path.append(os.getcwd())
from tqdm import tqdm
import cv2
from convert.utils import load, save, get_file_list, rec_rotate, polygon_area


def LabelMe2cvt(gt_dict, data_root, gt_folder):
    """
    将labelme格式的gt转换为json格式
    :param gt_folder:
    :param gt_dict:
    :return:
    """
    json_files = get_file_list(os.path.join(data_root, gt_folder), p_postfix=['.json'])
    random.shuffle(json_files)

    data_list = []
    for json_file in tqdm(json_files):
        img_path = pathlib.Path(json_file)
        cur_gt = {'img_name': gt_folder + '/' + img_path.name.replace('.json', '.jpg'), 'annotations': []}
        content = json.load(open(json_file, 'r'))
        for shape in content['shapes']:
            cur_line_gt = {'polygon': [], 'text': '', 'illegibility': False, 'language': 'Latin'}
            chars_gt = [{'polygon': [], 'char': '', 'illegibility': False, 'language': 'Latin'}]
            cur_line_gt['chars'] = chars_gt
            # 字符串级别的信息
            cur_line_gt['polygon'] = shape['points']
            cur_line_gt['text'] = "###"
            cur_line_gt['illegibility'] = True if cur_line_gt['text'] == '*' or cur_line_gt['text'] == '###' else False
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict['data_list'].extend(data_list)


if __name__ == '__main__':
    data_root = '/Users/qiuyurui/Desktop/Text-Detect-Data/'
    save_path = '/Users/qiuyurui/Desktop/Text-Detect-Data/train_labelme.json'
    save_path2 = '/home/vip/qyr/.data/val_labelme.json'
    gt_dict = {'data_root': data_root, 'data_list': []}
    gt_dict2 = {'data_root': data_root, 'data_list': []}

    LabelMe2cvt(gt_dict, data_root, '12')
    save(gt_dict, save_path)

