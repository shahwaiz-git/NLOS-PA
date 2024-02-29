from os.path import join

import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from PIL import Image


def PSNR(y, y_hat):
    assert y.shape == y_hat.shape
    psnr_lst = [peak_signal_noise_ratio(yi.cpu().numpy()[0], yi_hat.cpu().numpy()[0], data_range=1.0)
                for yi, yi_hat in zip(y, y_hat)]
    return np.mean(psnr_lst)


def SSIM(y, y_hat):
    assert y.shape == y_hat.shape
    psnr_lst = [structural_similarity(yi.cpu().numpy()[0], yi_hat.cpu().numpy()[0], data_range=1.0)
                for yi, yi_hat in zip(y, y_hat)]
    return np.mean(psnr_lst)


def save_result(y_hat, base_dir, names):
    # TODO: save format
    for i in range(len(names)):
        image = Image.fromarray(y_hat[i][0].cpu().numpy() * 255.).convert('L')
        image.save(join(base_dir, names[i][:-4] + '.png'))

