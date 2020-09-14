import numpy as np
import os


CAT_LIST = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor', 'meanIOU']

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist



def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def record_score(score, save_path, iou_type):
    score_list = []

    for i in range(21):
        score_list.append(score['Class IoU'][i])
        aveJ = score['Mean IoU']

    with open('{}/iou_{}.txt'.format(save_path, iou_type), 'w') as f:
        for num, cls_iou in enumerate(score_list):
            print('class {:2d} {:12} IU {:.2f}'.format(num, CAT_LIST[num], round(cls_iou, 3)))
            f.write('class {:2d} {:12} IU {:.2f}'.format(num, CAT_LIST[num], round(cls_iou, 3)) + '\n')
        print('meanIOU: ' + str(aveJ) + '\n')
        f.write('meanIOU: ' + str(aveJ) + '\n')
