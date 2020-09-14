import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self, k_cluster, from_round_nb):
        super().__init__()

        self.k = k_cluster
        self.from_round_nb = from_round_nb
        print('k_cluster: {}'.format(self.k))
        print('Round: {}'.format(self.from_round_nb))

        self.dropout7 = torch.nn.Dropout2d(0.5)

        # class 20
        if self.from_round_nb == 0:
            self.fc8 = nn.Conv2d(4096, 20, 1, bias=False)

            torch.nn.init.xavier_uniform_(self.fc8.weight)

            self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
            self.from_scratch_layers = [self.fc8]


        # class 20 + class 200
        else:
            self.fc8_20 = nn.Conv2d(4096, 20, 1, bias=False)
            self.fc8_200 = nn.Conv2d(4096, self.k*20, 1, bias=False)

            torch.nn.init.xavier_uniform_(self.fc8_20.weight)
            torch.nn.init.xavier_uniform_(self.fc8_200.weight)

            self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
            self.from_scratch_layers = [self.fc8_20, self.fc8_200]



    def forward(self, x, from_round_nb):
        x = super().forward(x)
        x = self.dropout7(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

        feature = x
        feature = feature.view(feature.size(0), -1)

        # class 20
        if from_round_nb == 0:
            x = self.fc8(x)
            x = x.view(x.size(0), -1)
            y = torch.sigmoid(x)
            return x, feature, y

        # class 20 + class 200
        else:
            x_20 = self.fc8_20(x)
            x_20 = x_20.view(x_20.size(0), -1)
            y_20 = torch.sigmoid(x_20)


            x_200 = self.fc8_200(x)
            x_200 = x_200.view(x_200.size(0), -1)
            y_200 = torch.sigmoid(x_200)

            return x_20, feature, y_20, x_200, y_200



    def multi_label(self, x):
        x = torch.sigmoid(x)
        tmp = x.cpu()
        tmp = tmp.data.numpy()
        _, cls = np.where(tmp>0.5)

        return cls, tmp


    def forward_cam(self, x):
        x = super().forward(x)
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x


    def forward_two_cam(self, x):
        x_ = super().forward(x)

        x_20 = F.conv2d(x_, self.fc8_20.weight)
        cam_20 = F.relu(x_20)

        x_200 = F.conv2d(x_, self.fc8_200.weight)
        cam_200 = F.relu(x_200)

        return cam_20, cam_200

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
