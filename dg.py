# -*- coding: utf-8 -*-

import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class DataMatch(Dataset):

    def __init__(self, xs, ys):
        super(DataMatch, self).__init__()
        self.xs = xs
        self.ys = ys
        
    def __getitem__(self, index):
        batch_x = self.xs[index]
        batch_y = self.ys[index]
        return batch_y, batch_x
        
    def __len__(self):
        return self.xs.size(0)

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(source_file_name, target_file_name):

    img1 = cv2.imread(source_file_name, 0)  # gray scale
    img2 = cv2.imread(target_file_name, 0)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    patches1 = []
    patches2 = []
    for s in scales:
        h_scaled, w_scaled = int(h1*s), int(w1*s)
        img1_scaled = cv2.resize(img1, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        img2_scaled = cv2.resize(img2, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img1_scaled[i:i+patch_size, j:j+patch_size]
                y = img2_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    mode = np.random.randint(0,8)
                    x_aug = data_aug(x, mode=mode)
                    y_aug = data_aug(y, mode=mode)
                    patches1.append(x_aug)
                    patches2.append(y_aug)
    return patches1, patches2


def datagenerator(source_dir='data/target_noise', target_dir='data/target', verbose=False):
    source_list = glob.glob(source_dir+'/*.png')  # get name list of all .png files
    target_list = glob.glob(target_dir+'/*.png')
    # initrialize
    data_x = []
    data_y = []
    # generate patches
    for i in range(len(source_list)):
        patch1, patch2 = gen_patches(source_list[i], target_list[i])
        data_x.append(patch2)
        data_y.append(patch1)
        if verbose:
            print(str(i+1) + '/' + str(len(source_list)) + ' is done ^_^')
    data_x = np.array(data_x, dtype='uint8')
    data_x = data_x.reshape((data_x.shape[0]*data_x.shape[1], data_x.shape[2], data_x.shape[3], 1))
    discard_n = len(data_x)-len(data_x)//batch_size*batch_size
    data_x = np.delete(data_x, range(discard_n), axis=0)
    data_y = np.array(data_y, dtype='uint8')
    data_y = data_y.reshape((data_y.shape[0]*data_y.shape[1], data_y.shape[2], data_y.shape[3], 1))
    discard_m = len(data_y)-len(data_y)//batch_size*batch_size
    data_y = np.delete(data_y, range(discard_m), axis=0)
    print('^_^-training data finished-^_^')
    return data_x, data_y


if __name__ == '__main__': 

    data = datagenerator(data_dir='data/Train400')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       