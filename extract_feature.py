import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt



def make_folder(save_folder_path):
    if os.path.exists(save_folder_path) == False:
        os.mkdir(save_folder_path)
    if os.path.exists(os.path.join(save_folder_path, 'feature')) == False:
        os.mkdir(os.path.join(save_folder_path, 'feature'))
    if os.path.exists(os.path.join(save_folder_path, 'label')) == False:
        os.mkdir(os.path.join(save_folder_path, 'label'))
    if os.path.exists(os.path.join(save_folder_path, 'log')) == False:
        os.mkdir(os.path.join(save_folder_path, 'log'))
    if os.path.exists(os.path.join(save_folder_path, 'weight')) == False:
        os.mkdir(os.path.join(save_folder_path, 'weight'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, default="./weights/res38_cls.pth", type=str, help="the weight of the model")
    parser.add_argument("--network", default="network.resnet38_cls", type=str, help="the network of the classifier")
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str, help="the filename list for feature extraction")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default="/home/julia/datasets/VOC2012", type=str, help="the path to the dataset folder")
    parser.add_argument("--from_round_nb", required=True, default=None, type=int, help="the round number of the extracter, e.g., 1st round: from_round_nb=0, 2nd round: from_round_nb=1, and so on")
    parser.add_argument("--k_cluster", default=10, type=int, help="the number of the sub-category")
    parser.add_argument("--save_folder", required=True, default='./save', type=str, help="the path to save the extracted feature")

    args = parser.parse_args()

    make_folder(args.save_folder)

    model = getattr(importlib.import_module(args.network), 'Net')(args.k_cluster, args.from_round_nb)
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=(1, 0.5, 1.5, 2.0),
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW
                                                        ]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    filename_list = []
    image_feature_list = []

    print('################################ Exteacting features from Round-{} ...... ################################'.format(args.from_round_nb))

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]

        filename_list.append(img_name)

        # extract feature
        if args.from_round_nb == 0:
            tmp, feature, _ = model.forward(img_list[0].cuda(), args.from_round_nb)
        else:
            tmp, feature, y_20, x_200, y_200 = model.forward(img_list[0].cuda(), args.from_round_nb)

        feature = feature[0].cpu().detach().numpy()
        image_feature_list.append(feature)

        if iter % 500 == 0:
            print('Already extracted: {}/{}'.format(iter, len(infer_data_loader)))


    image_feature_list = np.array(image_feature_list)
    print(image_feature_list.shape)

    # save the extracted feature
    save_feature_folder_path = os.path.join(args.save_folder, 'feature')
    feature_save_path = os.path.join(save_feature_folder_path, 'R{}_feature.npy'.format(args.from_round_nb)) # R1 feature is for R2 use
    np.save(feature_save_path, image_feature_list)
