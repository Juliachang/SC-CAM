import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import scipy.misc
import math



def create_class_key_in_dict(dict, cls_nb):
    for i in range(cls_nb):
        dict[i] = []
    return dict


def calculate_class_avg_iou(class_iou_dict):
    class_mean_iou_dict = {}
    for i in range(20):
        class_iou_list = class_iou_dict[i]
        if len(class_iou_list) != 0:
            class_iou_list_mean = round(sum(class_iou_list)/len(class_iou_list), 4)
        else:
            class_iou_list_mean = 0.
        class_mean_iou_dict[i] = class_iou_list_mean
    return class_mean_iou_dict


def draw_heatmap(img, hm):
    hm = plt.cm.hot(hm)[:, :, :3]
    hm = np.array(Image.fromarray((hm*255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(np.float)*2
    if hm.shape == np.array(img).astype(np.float).shape:
        out = (hm + np.array(img).astype(np.float)) / 3
        out = Image.fromarray((out / np.max(out) * 255).astype(np.uint8), 'RGB')
    return hm, out


def draw_heatmap_array(img, hm):
    hm = plt.cm.hot(hm)[:, :, :3]
    hm = np.array(Image.fromarray((hm*255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(np.float)*2
    if hm.shape == np.array(img).astype(np.float).shape:
        out = (hm + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    return hm, out


def cls200_vote(y_200, k_cluster):
    topk_subcls = np.argsort(y_200[0].detach().cpu().numpy())[-10:][::-1]
    topk_cls = np.array(topk_subcls/k_cluster, dtype=np.uint8)
    topk_vote = np.unique(topk_cls, return_counts=True)

    p_cls_sum = []
    for p_cls in topk_vote[0]:
        subcls_sum = []
        for prob in np.where(topk_cls == p_cls)[0]:
            subcls_sum.append(y_200[0][topk_subcls[prob]].item())
        p_cls_sum.append(sum(subcls_sum))

    cls200_pred_vote = topk_vote[0][np.where(np.array(p_cls_sum) >= 0.5)[0]]
    return cls200_pred_vote


def cls200_sum(y_200, k_cluster):
    cls20_prob_sum_list = []
    for rou in range(20):
        subclass_prob_sum = sum(y_200[0][rou*k_cluster:rou*k_cluster+k_cluster].detach().cpu().numpy())
        cls20_prob_sum_list.append(subclass_prob_sum/10)

    cls200_pred_max = np.where(np.array(cls20_prob_sum_list)>0.05)[0]
    return cls200_pred_max


def cam_subcls_norm(cam, cls20_gt, k_cluster):
    for gt in cls20_gt:
        subcls_cam = cam[gt*k_cluster:gt*k_cluster+k_cluster]

        norm_cam = subcls_cam / (np.max(subcls_cam, keepdims=True) + 1e-5)

        subcls_norm_cam = np.asarray(norm_cam)
        cam[gt*k_cluster:gt*k_cluster+k_cluster] = subcls_norm_cam
    return cam


def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    pred_correct_list = []
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return(acc)


def compute_iou(gt_labels, cam_np, gt_np, th, class_iou_dict):
    iou_list = []
    for label in gt_labels:
        cam = cam_np[label]
        gt = gt_np[label]

        gt_target_class = label + 1

        gt_y, gt_x = np.where(gt == gt_target_class)
        gt_pixel_nb = gt_y.shape[0]  # object

        correct_pixel_nb = 0

        cam_y, cam_x = np.where(cam >= th)
        high_response_pixel_nb = cam_y.shape[0]  # detected

        for pixel in range(gt_y.shape[0]):
            if cam[gt_y[pixel]][gt_x[pixel]] >= th:
                correct_pixel_nb += 1   # intersection
            else:
                continue

        union = gt_pixel_nb + high_response_pixel_nb - correct_pixel_nb

        if high_response_pixel_nb != 0:
            iou = round(correct_pixel_nb/union, 4)
        else:
            iou = 0.
        iou_list.append(iou)
        if high_response_pixel_nb != 0:
            precision = round(correct_pixel_nb/high_response_pixel_nb, 4)
        else:
            precision = 0.
        recall = round(correct_pixel_nb/gt_pixel_nb, 4)
        class_iou_dict[label].append(iou)
        print(label, iou)
    return class_iou_dict, iou_list




def compute_merge_iou(gt_labels, cam_nor, cam_b4_nor, gt_np, th, k, class_iou_dict):
    merged_cam_list = []
    for label in gt_labels:

        cam_b4_nor_ = cam_b4_nor[label*k:label*k+k]   # (10, 366, 500)
        cam_b4_sum = np.sum(cam_b4_nor_, axis=0)   # (366, 500)
        merge_cam = cam_b4_sum / np.amax(cam_b4_sum)   # (366, 500)   np.max(merge_cam)=1.0


        merged_cam_list.append(merge_cam)


        gt = gt_np[label]
        gt_target_class = label + 1

        gt_y, gt_x = np.where(gt == gt_target_class)
        gt_pixel_nb = gt_y.shape[0]  # object

        correct_pixel_nb = 0

        cam_y, cam_x = np.where(merge_cam >= th)
        high_response_pixel_nb = cam_y.shape[0]  # detected

        for pixel in range(gt_y.shape[0]):
            if merge_cam[gt_y[pixel]][gt_x[pixel]] >= th:
                correct_pixel_nb += 1   # intersection
            else:
                continue

        union = gt_pixel_nb + high_response_pixel_nb - correct_pixel_nb

        if high_response_pixel_nb != 0:
            iou = round(correct_pixel_nb/union, 4)
        else:
            iou = 0.
        if high_response_pixel_nb != 0:
            precision = round(correct_pixel_nb/high_response_pixel_nb, 4)
        else:
            precision = 0.
        recall = round(correct_pixel_nb/gt_pixel_nb, 4)
        class_iou_dict[label].append(iou)
    return class_iou_dict, merged_cam_list



def compute_merge_11_iou(gt_labels, cam_20, cam_200, gt_np, th, k, class_all_iou_dict):
    merged_cam_list = []
    for label in gt_labels:
        parcls_cam = np.expand_dims(cam_20[label], axis=0)
        subcls_cam = cam_200[label*k:label*k+k]

        merge_11_cam = np.concatenate((subcls_cam, parcls_cam), axis=0)
        merge_11_cam = np.amax(merge_11_cam, axis=0)
        merge_cam = merge_11_cam / np.amax(merge_11_cam)

        merged_cam_list.append(merge_cam)

        gt = gt_np[label]
        gt_target_class = label + 1

        gt_y, gt_x = np.where(gt == gt_target_class)
        gt_pixel_nb = gt_y.shape[0]

        correct_pixel_nb = 0

        cam_y, cam_x = np.where(merge_cam >= th)
        high_response_pixel_nb = cam_y.shape[0]

        for pixel in range(gt_y.shape[0]):
         if merge_cam[gt_y[pixel]][gt_x[pixel]] >= th:
             correct_pixel_nb += 1
         else:
             continue

        union = gt_pixel_nb + high_response_pixel_nb - correct_pixel_nb

        if high_response_pixel_nb != 0:
         iou = round(correct_pixel_nb/union, 4)
        else:
         iou = 0.
        if high_response_pixel_nb != 0:
         precision = round(correct_pixel_nb/high_response_pixel_nb, 4)
        else:
         precision = 0.
        recall = round(correct_pixel_nb/gt_pixel_nb, 4)
        class_all_iou_dict[label].append(iou)

    return class_all_iou_dict, merged_cam_list



def compute_ub_iou(gt_labels, cam_np, gt_np, th, k, class_iou_dict):
    iou_list = []
    all_subclass_iou_list = []
    for l_num, label in enumerate(gt_labels):
        subclass_iou_list = []
        cam = cam_np[label*k:label*k+k]
        for num, one in enumerate(cam):
            merge_cam = one
            gt = gt_np[label]

            gt_target_class = label + 1

            gt_y, gt_x = np.where(gt == gt_target_class)
            gt_pixel_nb = gt_y.shape[0]

            correct_pixel_nb = 0

            cam_y, cam_x = np.where(merge_cam >= th)
            high_response_pixel_nb = cam_y.shape[0]

            for pixel in range(gt_y.shape[0]):
                if merge_cam[gt_y[pixel]][gt_x[pixel]] >= th:
                    correct_pixel_nb += 1
                else:
                    continue

            union = gt_pixel_nb + high_response_pixel_nb - correct_pixel_nb


            if high_response_pixel_nb != 0:
                iou = round(correct_pixel_nb/union, 4)
            else:
                iou = 0.
            subclass_iou_list.append(iou)

            if high_response_pixel_nb != 0:
                precision = round(correct_pixel_nb/high_response_pixel_nb, 4)
            else:
                precision = 0.
            recall = round(correct_pixel_nb/gt_pixel_nb, 4)

        print(label, 'subcls_iou_list: {}'.format(subclass_iou_list))
        max_iou = max(subclass_iou_list)
        print(max_iou, subclass_iou_list.index(max(subclass_iou_list)))
        class_iou_dict[label].append(max_iou)
        iou_list.append(max_iou)

        all_subclass_iou_list.append(subclass_iou_list)
    return class_iou_dict, iou_list, all_subclass_iou_list




def count_maxiou_prob(y_200, cls20_gt, all_subclass_iou_list, class_20_iou_list, k_cluster, subclass_top_iou_list, class_200_ub_iou_list, class_ub_iou_dict, img_name):
    for i, gt in enumerate(cls20_gt):
        subclass_prob = y_200[0][gt*k_cluster:gt*k_cluster+k_cluster].detach().cpu().numpy()
        print('pred_score: {}'.format(subclass_prob))

        ten_subclass_iou_list = all_subclass_iou_list[i]
        ten_subclass_iou_list.append(class_20_iou_list[i])

        subclass_max_idx = ten_subclass_iou_list.index(max(ten_subclass_iou_list))
        pred_subclass = gt*k_cluster+subclass_max_idx

        if subclass_max_idx != 10:
            sort_subclass_prob_idx = np.argsort(subclass_prob)[::-1]
            top_k_best_iou = np.where(sort_subclass_prob_idx == subclass_max_idx)[0][0]
            subclass_top_iou_list[top_k_best_iou] += 1
        else:
            top_k_best_iou = 10
            subclass_top_iou_list[top_k_best_iou] += 1


        ub_iou = max(class_20_iou_list[i], class_200_ub_iou_list[i])
        class_ub_iou_dict[cls20_gt[i]].append(ub_iou)
        print(subclass_top_iou_list)

        line = '{},{},{},{},{}\n'.format(img_name, pred_subclass, top_k_best_iou, ub_iou, class_20_iou_list)

    return class_ub_iou_dict



def merge_topk_iou(y_200, gt_labels, all_subclass_iou_list, cam_np, gt_np, th, k, class_iou_dict):
    merged_cam_list = []
    for num, label in enumerate(gt_labels):
        subclass_prob = y_200[0][label*k:label*k+k].detach().cpu().numpy()
        subclass_iou_list = all_subclass_iou_list[num][:-1]

        cam = cam_np[label*k:label*k+k]
        sort_subcls_prob_idx = np.argsort(subclass_prob)[::-1]

        # print(subclass_prob)
        # print(subclass_iou_list)
        # print(sort_subcls_prob_idx)

        top_k_list = [0, 1, 2, 4, 9]
        top_k_iou_list = []
        for top in top_k_list:
            merge_k = np.zeros((top+1, cam.shape[1], cam.shape[2]))
            target_subcls_cam_idx = sort_subcls_prob_idx[:top+1]
            print(top, merge_k.shape, target_subcls_cam_idx)

            for i, idx in enumerate(target_subcls_cam_idx):
                merge_k[i] = cam[idx]

            ## norm -> max per pixel
            merge_cam = np.amax(merge_k, axis=0)

            # ## sum -> norm
            # merge_cam = np.sum(cam, axis=0) / np.amax(cam)

            merged_cam_list.append(merge_cam)


            gt = gt_np[label]
            gt_target_class = label + 1

            gt_y, gt_x = np.where(gt == gt_target_class)
            gt_pixel_nb = gt_y.shape[0]  # object

            correct_pixel_nb = 0

            cam_y, cam_x = np.where(merge_cam >= th)
            high_response_pixel_nb = cam_y.shape[0]  # detected

            for pixel in range(gt_y.shape[0]):
                if merge_cam[gt_y[pixel]][gt_x[pixel]] >= th:
                    correct_pixel_nb += 1   # intersection
                else:
                    continue

            union = gt_pixel_nb + high_response_pixel_nb - correct_pixel_nb

            if high_response_pixel_nb != 0:
                iou = round(correct_pixel_nb/union, 4)
            else:
                iou = 0.
            if high_response_pixel_nb != 0:
                precision = round(correct_pixel_nb/high_response_pixel_nb, 4)
            else:
                precision = 0.
            recall = round(correct_pixel_nb/gt_pixel_nb, 4)

            top_k_iou_list.append(iou)

    return class_iou_dict, merged_cam_list



def vrf_iou_w_distance(cls20_gt):
    cls20_w = np.load('./kmeans_subclass/c20_k10/3rd_round/weight_np/R3_cls20_w.npy')    # (20, 4096)
    cls200_w = np.load('./kmeans_subclass/c20_k10/3rd_round/weight_np/R3_cls200_w.npy')  # (200, 4096)

    bike_w = cls20_w[cls20_gt[0]]
    sub_human_w = cls200_w[cls20_gt[1]*10:cls20_gt[1]*10+10]

    sub_w_dis_list = []
    for num, sub in enumerate(sub_human_w):
        dist = np.linalg.norm(bike_w-sub)
        sub_w_dis_list.append(dist)
    print('dist_list: {}'.format(sub_w_dis_list))
    print(sub_w_dis_list.index(min(sub_w_dis_list)))


def find_200_pseudo_label(image_name, round_nb):
    filename_list_path = './kmeans_subclass/c20_k10/{}_round/train/{}_train_filename_list.txt'.format(round_nb, round_nb)
    label_20_npy = np.load( './kmeans_subclass/c20_k10/{}_round/train/{}_train_label_20.npy'.format(round_nb, round_nb))
    label_200_npy = np.load('./kmeans_subclass/c20_k10/{}_round/train/{}_train_label_200.npy'.format(round_nb, round_nb))

    with open(filename_list_path, 'r') as f:
        filename_list = f.read().split('\n')
    f.close()

    image_idx = filename_list.index(image_name)
    label_20 = label_20_npy[image_idx]
    label_200 = label_200_npy[image_idx]


    return label_20, label_200


def cam_npy_to_cam_dict(cam_np, label):
    cam_dict = {}
    idxs = np.where(label==1)[0]

    for idx in idxs:
        cam_dict[idx] = cam_np[idx]

    return cam_dict


def response_to_label(cam_npy):
    seg_map = cam_npy.transpose(1,2,0)
    seg_map = np.asarray(np.argmax(seg_map, axis=2), dtype=np.int)

    return seg_map


def get_accum_from_dict(par_cls, clust_dict):
    accum = 0
    for m in range(par_cls):
        accum += clust_dict[m]
    return accum


def cls200_cam_norm(cam_list_200, k_cluster):
    cam_200 = np.sum(cam_list_200, axis=0)
    norm_cam_200 = np.zeros((cam_200.shape[0], cam_200.shape[1], cam_200.shape[2]))

    for i in range(20):
        subcls_cam = cam_200[i*k_cluster:i*k_cluster+k_cluster]

        norm_cam = subcls_cam / (np.max(subcls_cam, keepdims=True) + 1e-5)

        subcls_norm_cam = np.asarray(norm_cam)
        norm_cam_200[i*k_cluster:i*k_cluster+k_cluster] = subcls_norm_cam
    return norm_cam_200



def cls200_cam_norm_dynamicK(cam_list_200, clust_dict):

    cam_200 = np.sum(cam_list_200, axis=0)
    norm_cam_200 = np.zeros((cam_200.shape[0], cam_200.shape[1], cam_200.shape[2]))
    for i in range(20):
        accum = get_accum_from_dict(i, clust_dict)

        subcls_cam = cam_200[accum:accum+clust_dict[i]]

        norm_cam = subcls_cam / (np.max(subcls_cam, keepdims=True) + 1e-5)

        subcls_norm_cam = np.asarray(norm_cam)
        norm_cam_200[accum:accum+clust_dict[i]] = subcls_norm_cam
    return norm_cam_200



def dict2npy(cam_dict, gt_label, th):
    gt_cat = np.where(gt_label==1)[0]

    orig_img_size = cam_dict[gt_cat[0]].shape

    bg_score = [np.ones_like(cam_dict[gt_cat[0]])*th]
    cam_npy = np.zeros((20, orig_img_size[0], orig_img_size[1]))

    for gt in gt_cat:
        cam_npy[gt] = cam_dict[gt]

    cam_npy = np.concatenate((bg_score, cam_npy), axis=0)
    return cam_npy



def merge_200_cam_dict(cam_dict_200, gt_label, th, k):
    gt_cat = np.where(gt_label==1)[0]

    orig_img_size = cam_dict_200[gt_cat[0]*k].shape

    cam_npy = np.zeros((20, orig_img_size[0], orig_img_size[1]))
    sub_cam_npy = np.zeros((k, orig_img_size[0], orig_img_size[1]))

    for gt in gt_cat:
        for i in range(k):
            sub_cam_npy[i] = cam_dict_200[gt*k+i]
        sub_cam_max_npy = np.amax(sub_cam_npy, axis=0)
        cam_npy[gt] = sub_cam_max_npy
    return cam_npy



def cam_npy_to_label_map(cam_npy):
    seg_map = cam_npy.transpose(1,2,0)
    seg_map = np.asarray(np.argmax(seg_map, axis=2), dtype=np.int)
    return seg_map



def cam_npy_to_cam_dict(cam_npy, label):
    cam_dict = {}
    for i in range(len(label)):
        if label[i] > 1e-5:
            cam_dict[i] = cam_npy[i]
    return cam_dict


def cls200_cam_to_cls20_entropy(no_norm_cam_200, k_cluster, norm_cam, save_path, img_name, orig_img, gt_label, save_entropy_heatmap):
    gt_cat = np.where(gt_label==1)[0]

    cam_200_entropy_path = '{}/entropy/cls_200/{}'.format(save_path, img_name)
    if save_entropy_heatmap == 1:
        if os.path.isdir(cam_200_entropy_path):
            shutil.rmtree(cam_200_entropy_path)
        os.mkdir(cam_200_entropy_path)

    entropy_npy = np.zeros((norm_cam.shape[0], norm_cam.shape[1], norm_cam.shape[2]))


    for i in range(20):
        sub_cams = no_norm_cam_200[i*k_cluster:i*k_cluster+k_cluster]

        sub_cams_sum = np.sum(sub_cams, axis=0)

        sub_cams_sum_10 = sub_cams_sum[np.newaxis, :]
        sub_cams_sum_10 = np.repeat(sub_cams_sum_10, k_cluster, axis=0)
        prob = sub_cams/(sub_cams_sum_10 + 1e-5)

        prob_log = np.log(prob + 1e-5) / np.log(k_cluster)

        entropy_norm = -(np.sum(prob*prob_log, axis=0))  # entropy normalization

        entropy_norm[entropy_norm<0]=0
        entropy_npy[i] = entropy_norm

        if save_entropy_heatmap == 1:
            if i in gt_cat:
                hm, heatmap = draw_heatmap(orig_img, entropy_norm)
                scipy.misc.imsave('{}/entropy/cls_200/{}/{}_{}.png'.format(save_path, img_name, img_name, i), heatmap)

    return entropy_npy


def create_folder(inference_dir_path):

    if os.path.exists(inference_dir_path) == True:
        shutil.rmtree(inference_dir_path)

    os.mkdir(inference_dir_path)
    os.mkdir(os.path.join(inference_dir_path + '/heatmap'))
    os.mkdir(os.path.join(inference_dir_path + '/heatmap/cls_20'))
    os.mkdir(os.path.join(inference_dir_path + '/heatmap/cls_200'))
    os.mkdir(os.path.join(inference_dir_path + '/output_CAM_npy'))
    os.mkdir(os.path.join(inference_dir_path + '/output_CAM_npy/cls_20'))
    os.mkdir(os.path.join(inference_dir_path + '/output_CAM_npy/cls_200'))
    os.mkdir(os.path.join(inference_dir_path + '/IOU'))
    os.mkdir(os.path.join(inference_dir_path + '/crf'))
    os.mkdir(os.path.join(inference_dir_path + '/crf/out_la_crf'))
    os.mkdir(os.path.join(inference_dir_path + '/crf/out_ha_crf'))




def draw_single_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    gt_cat = np.where(gt_label==1)[0]
    heatmap_list = []
    mask_list = []
    for i, gt in enumerate(gt_cat):
        hm, heatmap = draw_heatmap_array(orig_img, norm_cam[gt])
        cam_viz_path = os.path.join(save_path,'heatmap/cls_20', img_name + '_{}.png'.format(gt))
        scipy.misc.imsave(cam_viz_path, heatmap)

        norm_cam_gt = norm_cam[gt]
        norm_cam_gt[norm_cam_gt<=0.15]=0
        norm_cam_gt[norm_cam_gt>0.15]=255

        heatmap = np.transpose(heatmap, (2, 1, 0))
        heatmap_list.append(heatmap)
        mask_list.append(norm_cam_gt)

    return heatmap_list, mask_list




def draw_heatmap_cls200(norm_cam, gt_label, orig_img):
    gt_cat = np.where(gt_label==1)[0]
    heatmap_list = []
    for i, gt in enumerate(gt_cat):
        heatmap_cat_list = []
        for x in range(10):
            hm, heatmap = draw_heatmap_array(orig_img, norm_cam[gt*10+x])
            heatmap = np.transpose(heatmap, (2, 1, 0))
            heatmap_cat_list.append(heatmap)
        heatmap_list.append(heatmap_cat_list)

    return heatmap_list


def draw_heatmap_cls200_merge(norm_cam, gt_label, orig_img, img_name):
    gt_cat = np.where(gt_label==1)[0]
    heatmap_list = []
    for i, gt in enumerate(gt_cat):
        hm, heatmap = draw_heatmap_array(orig_img, norm_cam[gt])
        scipy.misc.imsave('/home/julia/julia_data/wsss/best/heatmap/cls_200/merge_{}.png'.format(img_name), heatmap)
        heatmap = np.transpose(heatmap, (2, 1, 0))
        heatmap_list.append(heatmap)

    return heatmap_list


def draw_heatmap_cls200_entropy(norm_cam, gt_label, orig_img):
    gt_cat = np.where(gt_label==1)[0]
    heatmap_list = []
    mask_list = []
    for i, gt in enumerate(gt_cat):
        hm, heatmap = draw_heatmap_array(orig_img, norm_cam[gt])

        norm_cam_gt = norm_cam[gt]
        norm_cam_gt[norm_cam_gt<=0.6]=0
        norm_cam_gt[norm_cam_gt>0.6]=255

        heatmap = np.transpose(heatmap, (2, 1, 0))
        heatmap_list.append(heatmap)
        mask_list.append(norm_cam_gt)

    return heatmap_list, mask_list


def combine_four_images(files, img_name, gt, save_path):
    result = Image.new("RGB", (1200, 800))

    for index, file in enumerate(files):
        img = file
        img.thumbnail((400, 400), Image.ANTIALIAS)
        x = index // 2 * 400
        y = index % 2 * 400
        w , h = img.size
        result.paste(img, (x, y, x + w, y + h))
    result.save(os.path.expanduser('./{}/combine_maps/{}_{}.jpg'.format(save_path, img_name, gt)))


def save_combine_response_maps(cam_20_heatmap, cam_200_merge_heatmap, cam_200_entropy_heatmap, cam_20_map, cam_200_entropy_map, orig_img, gt_label, img_name, save_path):
    gt_cat = np.where(gt_label==1)[0]
    orig_img_out = Image.fromarray(orig_img.astype(np.uint8), 'RGB')
    print(len(cam_20_heatmap), len(cam_200_merge_heatmap), len(cam_200_entropy_heatmap))

    for num, gt in enumerate(gt_cat):
        cls20_out = Image.fromarray(np.transpose(cam_20_heatmap[num], (2, 1, 0)).astype(np.uint8), 'RGB')
        cls200_merge_out = Image.fromarray(np.transpose(cam_200_merge_heatmap[num], (2, 1, 0)).astype(np.uint8), 'RGB')
        cls200_entropy_out = Image.fromarray(np.transpose(cam_200_entropy_heatmap[num], (2, 1, 0)).astype(np.uint8), 'RGB')
        cam_20_map_out = Image.fromarray(cam_20_map[num].astype(np.uint8))
        cam_200_entropy_map_out = Image.fromarray(cam_200_entropy_map[num].astype(np.uint8))

        image_list = [orig_img_out, cls200_merge_out, cls20_out, cls200_entropy_out, cam_20_map_out, cam_200_entropy_map_out]
        combine_four_images(image_list, img_name, gt, save_path)
