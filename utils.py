import numpy as np
import os
from PIL import Image


def find_crop_img_label_class_index():
    with open('crop/crop_img_feature/crop_image_filename_list.txt', 'r') as f:
        crop_image_filename_list = f.read().split('\n')
    f.close()
    crop_img_filename_list = crop_image_filename_list[:-1]

    crop_img_label_list = np.load('crop/crop_img_feature/crop_img_label.npy')
    print(len(crop_img_filename_list), crop_img_label_list.shape[0])

    class_dict = {}
    for i in range(20):
        class_idx_list = np.where(crop_img_label_list == i)[0]
        class_dict[i] = class_idx_list

    return class_dict

class_dict = find_crop_img_label_class_index()
