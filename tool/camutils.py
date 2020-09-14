import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def draw_heatmap(img, hm):
    hm = (hm - np.min(hm)) / (np.max(hm) - np.min(hm))
    hm = plt.cm.hot(hm)[:, :, :3]
    hm = np.array(Image.fromarray((hm*255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(np.float)*2
    if hm.shape == np.array(img).astype(np.float).shape:
        out = (hm + np.array(img).astype(np.float)) / 3
        out = Image.fromarray((out / np.max(out) * 255).astype(np.uint8), 'RGB')
    return hm, out


def dict2npy(cam_dict, gt_label, th):
    # gt_cat = np.where(gt_label==1)[0]
    # orig_img_size = cam_dict[gt_cat[0]].shape
    orig_img_size = cam_dict[gt_label[0]].shape


    bg_score = [np.ones_like(cam_dict[gt_label[0]])*th]
    cam_npy = np.zeros((20, orig_img_size[0], orig_img_size[1]))
    # print(cam_npy.shape)      # (20, 366, 500)

    for gt in gt_label:
        cam_npy[gt] = cam_dict[gt]

    cam_npy = np.concatenate((bg_score, cam_npy), axis=0)
    # print(cam_npy.shape)     # (21, 366, 500)
    return cam_npy


def cam_npy_to_label_map(cam_npy):
    seg_map = cam_npy.transpose(1,2,0)
    seg_map = np.asarray(np.argmax(seg_map, axis=2), dtype=np.int)
    # print(seg_map.shape, np.unique(seg_map))
    return seg_map
