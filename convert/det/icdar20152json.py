# -*- coding: utf-8 -*-
# @Time    : 2020/8/18 14:12
# @Author  : Qiu Yurui
"""
将icdar2015数据集转换为统一格式
"""
import pathlib
from tqdm import tqdm
from convert.utils import load, save, get_file_list


def cvt(gt_path, save_path, img_folder):
    """
    将icdar2015格式的gt转换为json格式
    :param gt_path:
    :param save_path:
    :return:
    """
    gt_dict = {'data_root': img_folder}
    data_list = []
    for file_path in tqdm(get_file_list(gt_path, p_postfix=['.txt'])):
        content = load(file_path)
        file_path = pathlib.Path(file_path)
        img_name = file_path.name.replace('gt_', '').replace('.txt', '.jpg')
        cur_gt = {'img_name': img_name, 'annotations': []}
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
    gt_dict['data_list'] = data_list
    save(gt_dict, save_path)


if __name__ == '__main__':
    gt_path = r'D:\dataset\icdar2015\detection\test\gt'
    img_folder = r'D:\dataset\icdar2015\detection\test\imgs'
    save_path = r'D:\dataset\icdar2015\detection\test.json'
    cvt(gt_path, save_path, img_folder)
