# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@license: (C) Copyright 2017-2025, Personal Projects Limited.
@contact: qiuyurui21@163.com
@software: Pycharm
@file: labelmeSplit.py
@time: 2020-09-11 16:38
@desc:
"""
import os
import random
import shutil
import glob
import json
import argparse


def make_dir(_mk_path):
    if not os.path.exists(_mk_path):
        os.mkdir(_mk_path)


def get_arguments():
    parser_online = argparse.ArgumentParser()
    parser_online.add_argument('-i', '--path_input', type=str, help='origin data directory', default="")
    parser_online.add_argument('-o', '--path_output', type=str, help='split data directory', default="")
    parser_online.add_argument('-t', '--train_num', type=float, help='number of data to train', default=0.8)
    parser_online.add_argument('-v', '--val_num', type=float, help='number of data to val', default=0.1)
    parser_online.add_argument('-e', '--eval_num', type=float, help='number of data to eval', default=0.1)

    return parser_online.parse_args()


def data_split(file_list, cnt_left, cnt_right, dest_dir):
    for i in range(cnt_left, cnt_right):
        shutil.copy(file_list[i], dest_dir)
        try:
            shutil.copy(file_list[i].replace('.json', '.jpg'), dest_dir)
        except:
            try:
                shutil.copy(file_list[i].replace('.json', '.jpeg'), dest_dir)
            except:
                try:
                    shutil.copy(file_list[i].replace('.json', '.png'), dest_dir)
                except:
                    pass


def checkJson(json_content):
    shapes = json_content['shapes']
    if len(shapes) == 0:
        return False
    else:
        return True


if __name__ == '__main__':
    args = get_arguments()
    ori_path = args.path_input
    out_path = args.path_output

    train_number = args.train_num
    val_num = args.val_num
    eval_num = args.eval_num

    train_dir = os.path.join(out_path, 'train/')
    val_dir = os.path.join(out_path, 'val/')
    eval_dir = os.path.join(out_path, 'eval/')

    make_dir(out_path)
    make_dir(train_dir)
    make_dir(val_dir)
    all_json_files = glob.glob(ori_path + '/*.json')

    for json_file in all_json_files:
        json_content = json.load(open(json_file, 'r'))
        if not checkJson(json_content):
            os.remove(json_file)

    all_json_files = glob.glob(ori_path + '/*.json')
    cnt_all = len(all_json_files)
    cnt_train = int(cnt_all * train_number)
    cnt_val = int(cnt_all * val_num)
    cnt_eval = int(cnt_all * eval_num)

    print('data for train: {0}, val: {1}, eval: {2}'.format(cnt_train, cnt_val, cnt_eval))

    random.shuffle(all_json_files)
    data_split(all_json_files, 0, cnt_train, train_dir)

    if cnt_eval != 0.0:
        data_split(all_json_files, cnt_train, cnt_train + cnt_val, val_dir)
        make_dir(eval_dir)
        data_split(all_json_files, cnt_train + cnt_val, cnt_all, eval_dir)
    else:
        data_split(all_json_files, cnt_train, cnt_all, val_dir)
