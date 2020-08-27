# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@license: (C) Copyright 2017-2025, Personal Projects Limited.
@contact: qiuyurui21@163.com
@software: Pycharm
@file: MSRA2json.py
@time: 2020-08-25 14:33
@desc:
"""
"""
将MSRA-TD500数据集转换为统一格式
"""
import pathlib
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from convert.utils import load, save, get_file_list, rec_rotate


def cvt(save_path, gt_folder, img_folder):
    """
    将icdar2015格式的gt转换为json格式
    :param gt_folder:
    :param save_path:
    :return:
    """
    gt_dict = {'data_root': img_folder}
    data_list = []
    for img_path in tqdm(get_file_list(img_folder, p_postfix=['.JPG'])):
        img_path = pathlib.Path(img_path)
        gt_path = pathlib.Path(gt_folder) / img_path.name.replace('.JPG', '.gt')
        content = load(gt_path)
        cur_gt = {'img_name': img_path.name, 'annotations': []}
        for line in content:
            cur_line_gt = {'polygon': [], 'text': '', 'illegibility': False, 'language': 'Latin'}
            chars_gt = [{'polygon': [], 'char': '', 'illegibility': False, 'language': 'Latin'}]
            cur_line_gt['chars'] = chars_gt
            # 字符串级别的信息
            line = line.split(' ')
            difficult = line[1]
            line = list(map(float, line))
            x, y = line[2], line[3]
            w, h = line[4], line[5]
            pointsrotate = rec_rotate(x, y, w, h, line[-1])
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, pointsrotate[:8]))
            cur_line_gt['polygon'] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            cur_line_gt['text'] = "###"
            cur_line_gt['illegibility'] = True if difficult == '1' else False
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict['data_list'] = data_list
    save(gt_dict, save_path)


if __name__ == '__main__':
    img_folder = '/Users/qiuyurui/Desktop/Text-Detect-Data/MSRA-TD500/test'
    gt_folder = '/Users/qiuyurui/Desktop/Text-Detect-Data/MSRA-TD500/test'
    save_path = '/Users/qiuyurui/Desktop/Text-Detect-Data/MSRA-TD500/test_new.json'

    cvt(save_path, gt_folder, img_folder)
