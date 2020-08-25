# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@license: (C) Copyright 2017-2025, Personal Projects Limited.
@contact: qiuyurui21@163.com
@software: Pycharm
@file: utils.py
@time: 2020-08-24 16:29
@desc:
"""

import math
import codecs
import cv2
import numpy as np
import json
import os
import glob
import pathlib
from natsort import natsorted

__all__ = ['load', 'save', 'get_file_list', 'show_bbox_on_image', 'load_gt']


def get_file_list(folder_path: str, p_postfix: list = None) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': load_txt, '.json': load_json, '.list': load_txt, '.gt': load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': save_txt, '.json': save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def save_txt(data, file_path):
    """
    将一个list的数组写入txt文件里
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def show_bbox_on_image(image, polygons=None, txt=None, color=None, font_path='convert/simsun.ttc'):
    """
    在图片上绘制 文本框和文本
    :param image:
    :param polygons: 文本框
    :param txt: 文本
    :param color: 绘制的颜色
    :param font_path: 字体
    :return:
    """
    from PIL import ImageDraw, ImageFont
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    if len(txt) == 0:
        txt = None
    if color is None:
        color = (255, 0, 0)
    if txt is not None:
        font = ImageFont.truetype(font_path, 20)
    for i, box in enumerate(polygons):
        if txt is not None:
            draw.text((int(box[0][0]) + 20, int(box[0][1]) - 20), str(txt[i]), fill='red', font=font)
        for j in range(len(box) - 1):
            draw.line((box[j][0], box[j][1], box[j + 1][0], box[j + 1][1]), fill=color, width=5)
        draw.line((box[-1][0], box[-1][1], box[0][0], box[0][1]), fill=color, width=5)
    return image


def load_gt(json_path):
    """
    从json文件中读取出 文本行的坐标和gt，字符的坐标和gt
    :param json_path:
    :return:
    """
    content = load(json_path)
    d = {}
    for gt in content['data_list']:
        img_path = os.path.join(content['data_root'], gt['img_name'])
        polygons = []
        texts = []
        illegibility_list = []
        language_list = []
        for annotation in gt['annotations']:
            if len(annotation['polygon']) == 0:
                continue
            polygons.append(annotation['polygon'])
            texts.append(annotation['text'])
            illegibility_list.append(annotation['illegibility'])
            language_list.append(annotation['language'])
            for char_annotation in annotation['chars']:
                if len(char_annotation['polygon']) == 0 or len(char_annotation['char']) == 0:
                    continue
                polygons.append(char_annotation['polygon'])
                texts.append(char_annotation['char'])
                illegibility_list.append(char_annotation['illegibility'])
                language_list.append(char_annotation['language'])
        d[img_path] = {'polygons': polygons, 'texts': texts, 'illegibility_list': illegibility_list,
                       'language_list': language_list}
    return d


def draw_box(img, box_str):
    points = box_str.split(',')
    if len(points) >= 9:
        point1_x = int(points[0])
        point1_y = int(points[1])
        point2_x = int(points[2])
        point2_y = int(points[3])
        point3_x = int(points[4])
        point3_y = int(points[5])
        point4_x = int(points[6])
        point4_y = int(points[7])
        label = points[8].strip()
        cv2.line(img, (point1_x, point1_y), (point2_x, point2_y), (255, 0, 0), 3)
        cv2.line(img, (point2_x, point2_y), (point3_x, point3_y), (0, 255, 0), 3)
        cv2.line(img, (point3_x, point3_y), (point4_x, point4_y), (0, 0, 255), 3)
        cv2.line(img, (point4_x, point4_y), (point1_x, point1_y), (0, 255, 255), 3)
        cv2.putText(img, label, (point1_x, point1_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (125, 125, 0), 2)
    return img


def vector_product(coord):
    coord = np.array(coord).reshape((4, 2))
    temp_det = 0
    for idx in range(3):
        temp = np.array([coord[idx], coord[idx + 1]])
        temp_det += np.linalg.det(temp)
    temp_det += np.linalg.det(np.array([coord[-1], coord[0]]))
    return temp_det * 0.5


def cal_distance(point1, point2):
    dis = np.sqrt(np.sum(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1])))
    return dis


# 基于海伦公式计算不规则四边形的面积
def helen_formula(coord):
    coord = np.array(coord).reshape((4, 2))
    # 计算各边的欧式距离
    dis_01 = cal_distance(coord[0], coord[1])
    dis_12 = cal_distance(coord[1], coord[2])
    dis_23 = cal_distance(coord[2], coord[3])
    dis_31 = cal_distance(coord[3], coord[1])
    dis_13 = cal_distance(coord[0], coord[3])
    p1 = (dis_01 + dis_12 + dis_13) * 0.5
    p2 = (dis_23 + dis_31 + dis_13) * 0.5
    # 计算两个三角形的面积
    area1 = np.sqrt(p1 * (p1 - dis_01) * (p1 - dis_12) * (p1 - dis_13))
    area2 = np.sqrt(p2 * (p2 - dis_23) * (p2 - dis_31) * (p2 - dis_13))
    return area1 + area2


def rotate(angle, x, y):
    """
    基于原点的弧度旋转
    :param angle:   弧度
    :param x:       x
    :param y:       y
    :return:
    """
    rotatex = math.cos(angle) * x - math.sin(angle) * y
    rotatey = math.cos(angle) * y + math.sin(angle) * x
    return rotatex, rotatey


def xy_rorate(theta, x, y, centerx, centery):
    """
    针对中心点进行旋转
    :param theta:
    :param x:
    :param y:
    :param centerx:
    :param centery:
    :return:
    """
    r_x, r_y = rotate(theta, x - centerx, y - centery)
    return centerx + r_x, centery + r_y


def rec_rotate(x, y, width, height, theta):
    """
    传入矩形的x,y和宽度高度，弧度，转成QUAD格式
    :param x:
    :param y:
    :param width:
    :param height:
    :param theta:
    :return:
    """
    centerx = x + width / 2
    centery = y + height / 2

    x1, y1 = xy_rorate(theta, x, y, centerx, centery)
    x2, y2 = xy_rorate(theta, x + width, y, centerx, centery)

    x3, y3 = xy_rorate(theta, x + width, y + height, centerx, centery)
    x4, y4 = xy_rorate(theta, x, y + height, centerx, centery)

    return x1, y1, x2, y2, x3, y3, x4, y4


def create_json_label(filename, path_save, img_h, img_w, shapes):
    _basename = filename.split('.')[0]
    default_json = {
        "version": "3.11.0",
        "flags": {},
        "shapes": [

        ],
        "lineColor": [
            0,
            255,
            0,
            128
        ],
        "fillColor": [
            255,
            0,
            0,
            128
        ],
        "imagePath": filename,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w
    }

    # for i in range(len(points)):
    #     if len(points[i]) == 4:
    #         shape_type = "rectangle"
    #     else:
    #         shape_type = "polygon"
    #     default_shape = {
    #         "label": labels[i],
    #         "line_color": None,
    #         "fill_color": None,
    #         "points": [
    #             [
    #                 int(points[i][0]),
    #                 int(points[i][1])
    #             ],
    #             [
    #                 int(points[i][2]),
    #                 int(points[i][3])
    #             ]
    #         ],
    #         "shape_type": shape_type,
    #     }

    default_json['shapes'].extend(shapes)
    save_name = _basename + '.json'
    save_file = os.path.join(path_save, save_name)
    fp = codecs.open(save_file, 'w', encoding='utf-8')
    json.dump(default_json, fp, ensure_ascii=False)
    fp.flush()
    fp.close()


def polygon_area(points):
    """返回多边形面积

    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2
