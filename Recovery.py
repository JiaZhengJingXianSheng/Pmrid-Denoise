# -*- coding: utf-8 -*-
'''
@Time    : 2022/4/16 20:39
@Author  : LYZ
@FileName: Recovery.py
@Software: PyCharm
'''

import os
import rawpy
import numpy as np
import torch
import Unet

input_path = "testset"
save_path = "result/denoise"
model_path = "BaseLine-38.pth"
# device = "cuda:0"
device = "cpu"

def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, height, width


def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def pre(input_path):
    raw_data_expand_c, height, width = read_image(input_path)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level=1024, white_level=16383)
    raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
    return raw_data_expand_c_normal,height,width


def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
    return output_data


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


if __name__ == "__main__":
    test_files = os.listdir(input_path)
    net = Unet.Unet().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    for i, file in enumerate(test_files):
        input_path = "testset" + '/' + file
        save_path = "result/denoise" + str(i) + ".dng"
        X,height,width = pre(input_path)

        X = X.to(device)

        Y = net(X)

        result_data = Y.detach().numpy().transpose(0, 2, 3, 1)
        result_data = inv_normalization(result_data, black_level=1024, white_level=16383)
        result_write_data = write_image(result_data, height, width)
        write_back_dng(input_path, save_path, result_write_data)

        print(str(file) + "\nfinish")
        print()
