# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@license: (C) Copyright 2017-2025, Personal Projects Limited.
@contact: qiuyurui21@163.com
@software: Pycharm
@file: getMixData.py
@time: 2020-09-03 16:29
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


def LSVTcvt(gt_dict, gt_dict2, data_root, gt_path, img_folder):
    """
    将icdar2015格式的gt转换为json格式
    :param gt_path:
    :param save_path:
    :return:
    """
    data_list = []
    origin_gt = load(gt_path)
    origin_gt_item = list(origin_gt.items())
    random.shuffle(origin_gt_item)
    for img_name, gt in tqdm(origin_gt_item[:-3000]):
        img = cv2.imread(os.path.join(data_root, img_folder + '/' + img_name + '.jpg'))
        h_o, w_o = img.shape[:2]
        cur_gt = {'img_name': img_folder + '/' + img_name + '.jpg', 'annotations': []}
        for line in gt:
            cur_line_gt = {'polygon': [], 'text': '', 'illegibility': False, 'language': 'Latin'}
            chars_gt = [{'polygon': [], 'char': '', 'illegibility': False, 'language': 'Latin'}]
            cur_line_gt['chars'] = chars_gt
            # 字符串级别的信息
            cur_line_gt['polygon'] = line['points']
            cur_line_gt['text'] = line['transcription']

            ratio = round(abs(polygon_area(line['points'])) / (h_o * w_o), 8)
            illegibility = line['illegibility']
            if illegibility is False:
                if ratio <= 0.00045:
                    illegibility = True

            cur_line_gt['illegibility'] = illegibility
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict['data_list'].extend(data_list)

    data_list = []
    for img_name, gt in tqdm(origin_gt_item[-3000:]):
        img = cv2.imread(os.path.join(data_root, img_folder + '/' + img_name + '.jpg'))
        h_o, w_o = img.shape[:2]
        cur_gt = {'img_name': img_folder + '/' + img_name + '.jpg', 'annotations': []}
        for line in gt:
            cur_line_gt = {'polygon': [], 'text': '', 'illegibility': False, 'language': 'Latin'}
            chars_gt = [{'polygon': [], 'char': '', 'illegibility': False, 'language': 'Latin'}]
            cur_line_gt['chars'] = chars_gt
            # 字符串级别的信息
            cur_line_gt['polygon'] = line['points']
            cur_line_gt['text'] = line['transcription']

            ratio = round(abs(polygon_area(line['points'])) / (h_o * w_o), 8)
            illegibility = line['illegibility']
            if illegibility is False:
                if ratio <= 0.00045:
                    illegibility = True

            cur_line_gt['illegibility'] = illegibility
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict2['data_list'] = data_list


def MSRAcvt(gt_dict, data_root, gt_folder, img_root):
    """
    将icdar2015格式的gt转换为json格式
    :param gt_folder:
    :param save_path:
    :return:
    """
    data_list = []
    img_folder = os.path.join(data_root, img_root)
    for img_path1 in tqdm(get_file_list(img_folder, p_postfix=['.JPG'])):
        img_path = pathlib.Path(img_path1)
        gt_path = pathlib.Path(os.path.join(data_root, gt_folder)) / img_path.name.replace('.JPG', '.gt')
        content = load(gt_path)
        cur_gt = {'img_name': img_root + '/' + img_path.name, 'annotations': []}
        img = cv2.imread(img_path)
        h_o, w_o = img.shape[:2]
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
            ratio = round(w * h / (h_o * w_o), 4)
            if difficult is '0':
                if ratio <= 0.0007:
                    difficult = '1'
            pointsrotate = rec_rotate(x, y, w, h, line[-1])
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, pointsrotate[:8]))
            cur_line_gt['polygon'] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            cur_line_gt['text'] = "###"
            cur_line_gt['illegibility'] = True if difficult == '1' else False
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict['data_list'].extend(data_list)


def icdar2015cvt(gt_dict, data_root, gt_path, img_folder):
    """
    将icdar2015格式的gt转换为json格式
    :param gt_path:
    :param save_path:
    :return:
    """
    data_list = []
    for file_path in tqdm(get_file_list(os.path.join(data_root, gt_path), p_postfix=['.txt'])):
        content = load(file_path)
        file_path = pathlib.Path(file_path)
        img_name = file_path.name.replace('gt_', '').replace('.txt', '.jpg')
        cur_gt = {'img_name': img_folder + '/' + img_name, 'annotations': []}
        for line in content:
            cur_line_gt = {'polygon': [], 'text': '', 'illegibility': False, 'language': 'Latin'}
            chars_gt = [{'polygon': [], 'char': '', 'illegibility': False, 'language': 'Latin'}]
            cur_line_gt['chars'] = chars_gt
            line = line.split(',')
            # 字符串级别的信息
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            cur_line_gt['polygon'] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            cur_line_gt['text'] = line[-1]
            cur_line_gt['illegibility'] = True if cur_line_gt['text'] == '*' or cur_line_gt['text'] == '###' else False
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict['data_list'].extend(data_list)


def LabelMe2cvt(gt_dict, gt_dict2, data_root, gt_folder):
    """
    将labelme格式的gt转换为json格式
    :param gt_folder:
    :param gt_dict:
    :return:
    """
    json_files = get_file_list(os.path.join(data_root, gt_folder), p_postfix=['.json'])
    random.shuffle(json_files)
    all_cnt = len(json_files)
    train_files = json_files[:int(all_cnt * 0.9)]
    val_files = json_files[int(all_cnt * 0.9):]
    data_list = []
    for json_file in tqdm(train_files):
        img_path = pathlib.Path(json_file)
        cur_gt = {'img_name': gt_folder + '/' + img_path.name.replace('.json', '.jpg'), 'annotations': []}
        content = json.load(open(json_file, 'r'))
        for shape in content['shapes']:
            cur_line_gt = {'polygon': [], 'text': '', 'illegibility': False, 'language': 'Latin'}
            chars_gt = [{'polygon': [], 'char': '', 'illegibility': False, 'language': 'Latin'}]
            cur_line_gt['chars'] = chars_gt
            # 字符串级别的信息
            cur_line_gt['polygon'] = shape['points']
            cur_line_gt['text'] = "**"
            cur_line_gt['illegibility'] = True if cur_line_gt['text'] == '*' or cur_line_gt['text'] == '###' else False
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict['data_list'].extend(data_list)

    data_list = []
    for json_file in tqdm(val_files):
        img_path = pathlib.Path(json_file)
        cur_gt = {'img_name': gt_folder + '/' + img_path.name.replace('.json', '.jpg'), 'annotations': []}
        content = json.load(open(json_file, 'r'))
        for shape in content['shapes']:
            cur_line_gt = {'polygon': [], 'text': '', 'illegibility': False, 'language': 'Latin'}
            chars_gt = [{'polygon': [], 'char': '', 'illegibility': False, 'language': 'Latin'}]
            cur_line_gt['chars'] = chars_gt
            # 字符串级别的信息
            cur_line_gt['polygon'] = shape['points']
            cur_line_gt['text'] = "**"
            cur_line_gt['illegibility'] = True if cur_line_gt['text'] == '*' or cur_line_gt['text'] == '###' else False
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict2['data_list'].extend(data_list)


if __name__ == '__main__':
    data_root = '/Users/qiuyurui/Desktop/Text-Detect-Data'
    save_path = '/Users/qiuyurui/Desktop/Text-Detect-Data/train_real.json'
    save_path2 = '/Users/qiuyurui/Desktop/Text-Detect-Data/val_real.json'
    gt_dict = {'data_root': data_root, 'data_list': []}
    gt_dict2 = {'data_root': data_root, 'data_list': []}
    for i in [17, 18, 19, 23]:  # 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23

        LabelMe2cvt(gt_dict, gt_dict2, data_root, '{}'.format(i))
    # LSVTcvt(gt_dict, gt_dict2, data_root,
    #         '/home/vip/qyr/.data/LSVT-detect-icdar2019/ICDAR2019LSVT/train_full_labels.json',
    #         'LSVT-detect-icdar2019/ICDAR2019LSVT/train_full_images')
    # MSRAcvt(gt_dict, data_root, 'MSRA-TD500/train', 'MSRA-TD500/train')
    # icdar2015cvt(gt_dict, data_root, 'icdar2015/detection/train/gt',
    #              'icdar2015/detection/train/imgs')
    save(gt_dict, save_path)
    # MSRAcvt(gt_dict2, data_root, 'MSRA-TD500/test', 'MSRA-TD500/test')
    # icdar2015cvt(gt_dict2, data_root, 'icdar2015/detection/test/gt',
    #              'icdar2015/detection/test/imgs')
    save(gt_dict2, save_path2)
