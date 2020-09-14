import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils, iouutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
import matplotlib.pyplot as plt
import cv2
from tool import infer_utils




cam_20_list = []
cam_200_list = []
gt_list = []



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str, help="the path to the testing model")
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", required=True, type=str, help="the path to the dataset folder")
    parser.add_argument("--save_crf", default=0, type=int, help="the flag to apply crf")
    parser.add_argument("--low_alpha", default=4, type=int, help="crf parameter")
    parser.add_argument("--high_alpha", default=32, type=int, help="crf parameter")
    parser.add_argument("--save_out_cam", default=0, type=int, help="the flag to save the CAM array")
    parser.add_argument("--th", default=0.15, type=float, help="the threshold for the response map")
    parser.add_argument("--save_path", required=True, default=None, type=str, help="the path to save the CAM")
    parser.add_argument("--k_cluster", required=True, default=10, type=int, help="the number of the sub-category")
    parser.add_argument("--round_nb", default=1, type=int, help="the round number of the testing model")

    args = parser.parse_args()

    infer_utils.create_folder(args.save_path)

    model = getattr(importlib.import_module(args.network), 'Net')(args.k_cluster, args.round_nb)
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=(1, 0.5, 1.5, 2.0),
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        gt_cat = np.where(label==1)[0]
        label_200 = torch.zeros((20*args.k_cluster))
        # label_200 = torch.zeros((args.k_cluster))
        for cat in gt_cat:
            label_200[cat*args.k_cluster:cat*args.k_cluster+args.k_cluster] = 1
        # print(np.where(label_200==1))    # convert cls20_label to cls200_label


        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    cam, cam_200 = model_replicas[i%n_gpus].forward_two_cam(img.cuda())

                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)

                    cam_200 = F.upsample(cam_200, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam_200 = cam_200.cpu().numpy() * label_200.clone().view(20*args.k_cluster, 1, 1).numpy()
                    # cam_200 = cam_200.cpu().numpy() * label_200.clone().view(args.k_cluster, 1, 1).numpy()    ### need to alter args.k_cluster
                    if i % 2 == 1:
                        cam_200 = np.flip(cam_200, axis=-1)

                    return cam, cam_200

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()


        cam_list_20 = [pair[0] for pair in cam_list]
        cam_list_200 = [pair[1] for pair in cam_list]

        # class 20 cam normalization
        sum_cam = np.sum(cam_list_20, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)
        # class 200 cam normalization
        norm_cam_200 = infer_utils.cls200_cam_norm(cam_list_200, args.k_cluster)


        # class 20 cam --> class 20 segmap
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        seg_map = infer_utils.cam_npy_to_label_map(infer_utils.dict2npy(cam_dict, label, args.th))
        cam_20_list.append(seg_map)
        cam_20_heatmap = infer_utils.draw_single_heatmap(norm_cam, label, orig_img, args.save_path, img_name)


        # class 200 cam --> class 200 segmap
        cam_dict_200 = infer_utils.cam_npy_to_cam_dict(norm_cam_200, label_200)
        merge_200_cam = infer_utils.merge_200_cam_dict(cam_dict_200, label, args.th, args.k_cluster)
        cam_dict_200_merge = infer_utils.cam_npy_to_cam_dict(merge_200_cam, label)
        seg_map_200 = infer_utils.cam_npy_to_label_map(infer_utils.dict2npy(cam_dict_200_merge, label, args.th))
        cam_200_list.append(seg_map_200)


        # read groundtruth map
        gt_folder_path = os.path.join(args.voc12_root, 'SegmentationClassAug')
        gt_map_path = os.path.join(gt_folder_path, img_name + '.png')
        gt_map = cv2.imread(gt_map_path, cv2.IMREAD_GRAYSCALE)
        gt_list.append(gt_map)


        if args.save_out_cam == 1:
            np.save(os.path.join(args.save_path, 'output_CAM_npy/cls_20', img_name + '.npy'), cam_dict)
            # np.save(os.path.join(args.save_path, 'output_CAM_npy/cls_200', img_name + '.npy'), cam_dict_200)


        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if args.save_crf == 1:
            crf_la = _crf_with_alpha(cam_dict, args.low_alpha)  # type(crf_la): <class 'dict'>
            # np.save(os.path.join(args.save_path, 'crf/out_la_crf', img_name + '.npy'), crf_la)
            crf_ha = _crf_with_alpha(cam_dict, args.high_alpha)
            # np.save(os.path.join(args.save_path, 'crf/out_ha_crf', img_name + '.npy'), crf_ha)

        print("k={} Round-{}| NOW ITER: {}".format(args.k_cluster, args.round_nb, iter))


    print(len(cam_20_list), len(gt_list), len(cam_200_list))
    print('NOW K: {} R{}'.format(args.k_cluster, args.round_nb))


    score = iouutils.scores(gt_list, cam_20_list, n_class=21)
    iouutils.record_score(score, os.path.join(args.save_path, 'IOU'), 'cls20')

    score_200 = iouutils.scores(gt_list, cam_200_list, n_class=21)
    iouutils.record_score(score_200, os.path.join(args.save_path, 'IOU'), 'cls200')
