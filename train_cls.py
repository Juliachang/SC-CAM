import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
import voc12.data
from tool import pyutils, imutils, torchutils
import argparse
import importlib
import torch.nn.functional as F

import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import collections

import random
import scipy.misc
import os
from PIL import Image
from tensorboardX import SummaryWriter



def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    pred_correct_list = []
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc


def cls200_sum(y_200):
    cls20_prob_sum_list = []
    for rou in range(20):
        subclass_prob_sum = sum(y_200[rou*k_cluster:rou*k_cluster+k_cluster])
        cls20_prob_sum_list.append(subclass_prob_sum/10)

    cls200_pred_max = np.where(np.array(cls20_prob_sum_list)>0.05)[0]
    return cls200_pred_max


def get_img_path(img_name, dataset_path):
    tmp = os.path.join(dataset_path, img_name + '.jpg')
    return tmp





class Sub_Class_Dataset(Dataset):
    def __init__(self, voc12_root, crop_size, round_nb, k_cluster, save_folder, test=False):
        print('############################################## || k_{} / Round {} || ##############################################'.format(k_cluster, round_nb))

        self.voc12_root = voc12_root
        self.crop_size = crop_size

        with open('{}/label/R{}_train_filename_list.txt'.format(save_folder, round_nb), 'r') as f:
            self.filename = f.read().split('\n')
        f.close()
        self.filename = self.filename[:-1]  # 16458

        self.label = np.load('{}/label/R{}_train_label_200.npy'.format(save_folder, round_nb))
        self.label = torch.from_numpy(self.label).float()

        self.label_20 = np.load('{}/label/R{}_train_label_20.npy'.format(save_folder, round_nb)) # 16458

        self.label_20 = torch.from_numpy(self.label_20).float()

        print('=='*60)
        print('Training Data: image: {} | 20 class label: {} | 200 class label: {}'.format(len(self.filename), self.label_20.shape, self.label.shape))
        print('=='*60)


    def __getitem__(self, index):

        label_200 = self.label[index]
        label_20 = self.label_20[index]

        self.dataset_path = os.path.join(self.voc12_root, 'JPEGImages')

        filename = self.filename[index]

        img = Image.open(get_img_path(filename, self.dataset_path)).convert("RGB")
        img = imutils.ResizeLong(img, 256, 512)
        img = imutils.Flip(img)
        img = imutils.ColorJitter(img)
        img = np.array(img)
        img = imutils.NNormalize(img)
        img = imutils.Crop(img, self.crop_size)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img)

        return img, label_20, label_200, filename

    def __len__(self):
        return len(self.filename)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=61, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", required=True, type=str, help="the path to the pretrained weight ")
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--session_name", default="resnet_cls", type=str)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--voc12_root", default="/home/julia/datasets/VOC2012", type=str, help="the path to the dataset folder")
    parser.add_argument("--subcls_loss_weight", default="5", type=float, help="the weight multiply to the sub-category loss")
    parser.add_argument("--round_nb", default="0", type=int, help="the round number of the training classifier, e.g., 1st round: round_nb=1, and so on")
    parser.add_argument("--k_cluster", default="10", type=int, help="the number of the sub-category")
    parser.add_argument("--save_folder", required=True, default="./save", type=str, help="the path to save the model")

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')(args.k_cluster, args.round_nb)

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    log_path = os.path.join(args.save_folder, 'log', 'R{}'.format(args.round_nb))
    writer = SummaryWriter('{}'.format(log_path))


    train_dataset = Sub_Class_Dataset(voc12_root=args.voc12_root,
                                        crop_size = args.crop_size,
                                        round_nb=args.round_nb,
                                        k_cluster=args.k_cluster,
                                        save_folder=args.save_folder)

    train_data_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    drop_last=True)



    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches


    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)


    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    elif args.weights[-11:] == '.caffemodel':
        assert args.network == "network.vgg16_cls"
        import network.vgg16d
        weights_dict = network.vgg16d.convert_caffe_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")


    parent_labels = np.load('./voc12/cls_labels.npy').tolist()
    k_cluster = 10


    step = 0
    for ep in range(args.max_epoches):
        ep_count = 0
        ep_EM = 0
        ep_acc = 0

        ep_p_EM = 0
        ep_p_acc = 0
        ep_acc_vote = 0

        cls20_ep_EM = 0
        cls20_ep_acc = 0

        for iter, (data, label_20, label_200, filename) in tqdm(enumerate(train_data_loader)):

            img = data
            label_20 = label_20.cuda(non_blocking=True)
            label_200 = label_200.cuda(non_blocking=True)
            img_name = filename


            x_20, _, y_20, x_200, y_200 = model(img, args.round_nb)

            # compute acc for 20 classes
            cls20_prob = y_20.cpu().data.numpy()
            cls20_gt = label_20.cpu().data.numpy()
            for num, one in enumerate(cls20_prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]
                true_cls_20 = np.where(cls20_gt[num] == 1)[0]

                if np.array_equal(pass_cls, true_cls_20) == True:  # exact match
                    cls20_ep_EM += 1

                acc = compute_acc(pass_cls, true_cls_20)
                cls20_ep_acc += acc


            # compute acc for 200 classes
            tmp = y_200.cpu().data.numpy()
            tmp_label = label_200.cpu().data.numpy()
            for num, one in enumerate(tmp):
                pass_cls = np.where(one>0.5)[0]
                true_cls_200 = np.where(tmp_label[num] == 1)[0]

                parent_cls = np.where(parent_labels[img_name[num]] == 1)[0]

                cls200_pred_max = cls200_sum(one)

                if np.array_equal(pass_cls, true_cls_200) == True:
                    ep_EM += 1

                # cls200: acc for 200-->20 (sum)
                acc = compute_acc(cls200_pred_max, parent_cls)
                ep_acc += acc

                # cls200: acc for 200-->20 (top 1)
                pass_map_cls = np.unique([int(m/k_cluster) for m in np.where(one > 0.5)[0]])
                if np.array_equal(pass_map_cls, parent_cls) == True:
                    ep_p_EM += 1

                p_acc = compute_acc(pass_map_cls, parent_cls)
                ep_p_acc += p_acc



            avg_ep_EM = round(ep_EM/ep_count, 4)
            avg_ep_acc = round(ep_acc/ep_count, 4)

            avg_ep_p_EM = round(ep_p_EM/ep_count, 4)
            avg_ep_p_acc = round(ep_p_acc/ep_count, 4)

            avg_cls20_ep_EM = round(cls20_ep_EM/ep_count, 4)
            avg_cls20_ep_acc = round(cls20_ep_acc/ep_count, 4)

            cls_20_loss = F.multilabel_soft_margin_loss(x_20, label_20)
            cls_200_loss = F.multilabel_soft_margin_loss(x_200, label_200)

            loss = cls_20_loss + (args.subcls_loss_weight*cls_200_loss)

            if iter%100 ==0:
                print('k{} R{}| Ep:{} L:{} -20_LOSS:{} -200_LOSS:{} | -cls20:{} | -cls200_Top1:{} -Sum:{}'.format(args.k_cluster, args.round_nb, ep, round(loss.item(), 3), round(cls_20_loss.item(), 3), round(cls_200_loss.item(), 3), avg_cls20_ep_acc, avg_ep_p_acc, avg_ep_acc))

            avg_meter.add({'loss': cls_200_loss.item()})


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            writer.add_scalar('20 Classification Loss', cls_20_loss.item(), step)
            writer.add_scalar('200 Classification Loss', cls_200_loss.item(), step)
            writer.add_scalar('Total Loss', loss.item(), step)
            writer.add_scalar('Cls20 Accuracy', avg_cls20_ep_acc, step)
            writer.add_scalar('Cls200 Accuracy (Sum)', avg_ep_acc, step)
            writer.add_scalar('Cls200 Accuracy (Top1)', avg_ep_p_acc, step)

            step += 1

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)


        if ep % 10 == 0:
            torch.save(model.module.state_dict(), '{}/weight/k{}_R{}_'.format(args.save_folder, args.k_cluster, args.round_nb) + args.session_name + '_ep{}.pth'.format(ep))
            print('Loss: {} achieves the lowest one => Epoch {} weights are saved!'.format(loss, ep))
