# -*- coding: utf-8 -*-
'''
@Time    : 2022/4/17 13:03
@Author  : LYZ
@FileName: main.py
@Software: PyCharm
'''

from PmridNet import PMRID
import os
import rawpy
import numpy as np
import torch
from torch import nn
from Eval import inv_normalization, write_image, write_back_dng, pre
import skimage
import pytorch_ssim

noisy_data_path = "dataset/noisy/"
origin_data_path = "dataset/ground_truth/"
NoisyFiles = os.listdir(noisy_data_path)
OriginFiles = os.listdir(origin_data_path)
NoisyFiles_len = len(NoisyFiles)
device = "cuda:1"
lr = 0.00005
loss1 = nn.L1Loss()
loss2 = nn.MSELoss()
epochs = 1000
model_path = "Pmrid-400.pth"

white_level = 16383
black_level = 1024
# def read_image(input_path):
#     raw = rawpy.imread(input_path)
#     raw_data = raw.raw_image_visible
#     height = raw_data.shape[0]
#     width = raw_data.shape[1]
#
#     raw_data_expand = np.expand_dims(raw_data, axis=2)
#     raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
#                                         raw_data_expand[0:height:2, 1:width:2, :],
#                                         raw_data_expand[1:height:2, 0:width:2, :],
#                                         raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
#     return raw_data_expand_c, height, width
#
#
# def normalization(input_data, black_level, white_level):
#     output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
#     return output_data


# def pre(input_path):
#     raw_data_expand_c, height, width = read_image(input_path)
#     raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level=1024, white_level=16383)
#     raw_data_expand_c_normal = torch.from_numpy(np.transpose(
#         raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
#     return raw_data_expand_c_normal, height, width


if __name__ == "__main__":
    net = PMRID()
    net.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.to(device)
    net.train()

    for epoch in range(epochs):
        # net.train()
        running_loss = 0.0
        for i in range(NoisyFiles_len):
            X, X_height, X_width = pre(input_path=noisy_data_path + str(i) + "_noise.dng")
            Y, Y_height, Y_width = pre(input_path=origin_data_path + str(i) + "_gt.dng")
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            Y_HAT = net(X)
            l1 = loss1(Y_HAT, Y)

            # result_data = Y.detach().cpu().numpy().transpose(0, 2, 3, 1)
            # result_data = inv_normalization(result_data, black_level=1024, white_level=16383)
            # result_write_data = write_image(result_data, X_height, X_width)
            #
            # Y_HAT_result_data = Y_HAT.detach().cpu().numpy().transpose(0, 2, 3, 1)
            # Y_HAT_result_data = inv_normalization(Y_HAT_result_data, black_level=1024, white_level=16383)
            # Y_HAT_result_write_data = write_image(Y_HAT_result_data, X_height, X_width)
            # ssim = skimage.metrics.structural_similarity(
            #     Y_HAT_result_write_data.astype(np.float64), result_write_data.astype(np.float64), channel_axis=True, data_range=white_level)
            #
            # l2 = 1- ssim

            l2 = loss2(Y_HAT, Y)
            l = l1 + l2
            l.backward()
            optimizer.step()

            running_loss += l.item()
        print("Epoch{}\tloss {}".format(epoch, running_loss / NoisyFiles_len))

        # if epoch % 5 == 0:
        #     noisy_path = "dataset/noisy/0_noise.dng"
        #     gt_path = "dataset/ground_truth/0_gt.dng"
        #     output_path = "tem/0_noise.dng"
        #     net.eval()
        #
        #     XX, height, width = pre(noisy_path)
            #
            # XX = XX.to(device)
            #
            # YY = net(XX)
            #
            # result_data = YY.detach().to("cpu").numpy().transpose(0, 2, 3, 1)
            # result_data = inv_normalization(result_data, black_level=1024, white_level=16383)
            # result_write_data = write_image(result_data, height, width)
            # write_back_dng(noisy_path, output_path, result_write_data)
            # """
            # obtain psnr and ssim
            # """
            # gt = rawpy.imread(gt_path).raw_image_visible
            # psnr = skimage.metrics.peak_signal_noise_ratio(
            #     gt.astype(np.float64), result_write_data.astype(np.float64), data_range=white_level)
            # ssim = skimage.metrics.structural_similarity(
            #     gt.astype(np.float64), result_write_data.astype(np.float64), channel_axis=True, data_range=white_level)
            # print('psnr:', psnr)
            # print('ssim:', ssim)
            # print("\n")
            # net.train()
        torch.save(net.state_dict(), 'models/Pmrid-' + str(epoch) + '.pth')
