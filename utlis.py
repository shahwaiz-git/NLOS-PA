import os
from os.path import join

import numpy as np
import torch
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from PIL import Image
import scipy


def PSNR(y, y_hat):
    '''
    shape: batch_size, 1, height, width
    '''
    assert y.shape == y_hat.shape
    psnr_lst = [peak_signal_noise_ratio(yi.cpu().numpy()[0], yi_hat.cpu().numpy()[0], data_range=1.0)
                for yi, yi_hat in zip(y, y_hat)]
    return np.mean(psnr_lst)


def SSIM(y, y_hat):
    assert y.shape == y_hat.shape
    psnr_lst = [structural_similarity(yi.cpu().numpy()[0], yi_hat.cpu().numpy()[0], data_range=1.0, win_size=255)
                for yi, yi_hat in zip(y, y_hat)]
    return np.mean(psnr_lst)


def save_image(y_hat, base_dir, names):
    # TODO: save format
    base_dir = join(base_dir, 'target_hat')
    os.makedirs(base_dir, exist_ok=True)
    for i in range(len(names)):
        image = Image.fromarray(y_hat[i][0].cpu().numpy() * 255.).convert('L')
        image.save(join(base_dir, names[i][:-4] + '.png'))


def save_mat(signal_hat, base_dir, names):
    base_dir = join(base_dir, 'direct_signal_hat')
    os.makedirs(base_dir, exist_ok=True)
    for i in range(len(names)):
        mat = signal_hat[i].cpu().numpy()
        scipy.io.savemat(join(base_dir, names[i][:-4] + '.mat'), {'direct_signal_hat': mat})


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import ToTensor

    y = ToTensor()(np.array(Image.open(r"D:\Photo\证件照\DSC_0146.JPG")))
    y_hat = ToTensor()(np.array(Image.open(r"D:\Photo\证件照\压缩1.jpg")))
    print(PSNR(y, y_hat))
    print(SSIM(y, y_hat))
