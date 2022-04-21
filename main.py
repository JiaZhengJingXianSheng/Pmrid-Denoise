# -*- coding: utf-8 -*-
'''
@Time    : 2022/4/17 13:03
@Author  : LYZ
@FileName: main.py
@Software: PyCharm
'''

from PrmidNetOrigin import PMRID
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
lr = 0.00001
loss1 = nn.L1Loss()
# loss2 = nn.MSELoss()
epochs = 1000
model_path = "PmridOrigin-880.pth"

white_level = 16383
black_level = 1024

if __name__ == "__main__":
    net = PMRID()
    net.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.to(device)
    # net.train()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i in range(NoisyFiles_len):
            X, X_height, X_width = pre(input_path=noisy_data_path + str(i) + "_noise.dng")
            Y, Y_height, Y_width = pre(input_path=origin_data_path + str(i) + "_gt.dng")
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            Y_HAT = net(X)
            l = loss1(Y_HAT, Y)

            # l2 = loss2(Y_HAT, Y)
            # l = l1 + l2
            l.backward()
            optimizer.step()

            running_loss += l.item()
        print("Epoch{}\tloss {}".format(epoch, running_loss / NoisyFiles_len))


        if epoch % 5 == 0:
            with torch.no_grad():
                noisy_path = "dataset/noisy/0_noise.dng"
                gt_path = "dataset/ground_truth/0_gt.dng"
                output_path = "tem/0_noise.dng"
                net.eval()

                XX, height, width = pre(noisy_path)
                #
                XX = XX.to(device)

                YY = net(XX)

                result_data = YY.detach().to("cpu").numpy().transpose(0, 2, 3, 1)
                result_data = inv_normalization(result_data, black_level=1024, white_level=16383)
                result_write_data = write_image(result_data, height, width)
                write_back_dng(noisy_path, output_path, result_write_data)
                """
                obtain psnr and ssim
                """
                gt = rawpy.imread(gt_path).raw_image_visible
                psnr = skimage.metrics.peak_signal_noise_ratio(
                    gt.astype(np.float64), result_write_data.astype(np.float64), data_range=white_level)
                ssim = skimage.metrics.structural_similarity(
                    gt.astype(np.float64), result_write_data.astype(np.float64), channel_axis=True, data_range=white_level)
                print('psnr:', psnr)
                print('ssim:', ssim)
                print("\n")
            # net.train()
        torch.save(net.state_dict(), 'models/PmridOrigin-' + str(epoch) + '.pth')
