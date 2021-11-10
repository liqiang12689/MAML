8
import os
import zipfile

# root_path = './../datasets'
# processed_folder =  os.path.join(root_path)

# zip_ref = zipfile.ZipFile(os.path.join(root_path,'mini-imagenet.zip'), 'r')
# zip_ref.extractall(root_path)
# zip_ref.close()

9
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random


class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all images
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'images')  # image path

        # :return: dictLabels: {label1: [filename1, filename2, filename3, filename4,...], }
        dictLabels = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (label, imgs) in enumerate(dictLabels.items()):
            self.data.append(imgs)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[label] = i + self.startidx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)

        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


10
# spt_x_batch = []
# qry_x_batch = []

# dtrain = MiniImagenet('./../datasets/mini-imagenet/', mode='train', n_way=5, k_shot=1,
#                     k_query=15,
#                     batchsz=10000, resize=84)


# for b in range(10000):
#     selected_cls = np.random.choice(65, 5, False)
#     np.random.shuffle(selected_cls)

#     ## 构造支持集和查询集x
#     spt_x = []
#     qry_x = []
#     for cls in selected_cls:
#         selected_imgs_idx = np.random.choice(len(dtrain.data[cls]), 1 + 15, False)
#         np.random.shuffle(selected_imgs_idx)
#         indexDtrain = np.array(selected_imgs_idx[:1])
#         indexDtest = np.array(selected_imgs_idx[1:])
#         spt_x.append(np.array(dtrain.data[cls])[indexDtrain].tolist())
#         qry_x.append(np.array(Dtrain.data[cls])[indexDtest].tolist())
#     random.shuffle(spt_x)
#     random.shuffle(qry_x)

#     spt_x_batch.append(spt_x)
#     qry_x_batch.append(qry_x)

11


class Hello():
    def __init__(self):
        self.a = 1

    def __getitem__(self, key):
        print('hello')


ins = Hello()

12
from torch import nn


class Learner(nn.Module):
    """
    定义一个网络
    """

    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config  ## 对模型各个超参数的定义
        '''
        ## ParameterList可以像普通Python列表一样进行索引，
        但是它包含的参数已经被正确注册，并且将被所有的Module方法都可见。

        '''
        self.vars = nn.ParameterList()  ## 这个字典中包含了所有需要被优化的tensor
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                ## [ch_out, ch_in, kernel_size, kernel_size]
                weight = nn.Parameter(torch.ones(*param[:4]))  ## 产生*param大小的全为1的tensor
                torch.nn.init.kaiming_normal_(weight)  ## 初始化权重
                self.vars.append(weight)  ## 加到nn.ParameterList中

                bias = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)

            elif name is 'linear':
                weight = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(weight)
                self.vars.append(weight)
                bias = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)

            elif name is 'bn':
                ## 对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作,
                ## BN层在训练过程中，会将一个Batch的中的数据转变成正态分布
                weight = nn.Parameter(torch.ones(param[0]))
                self.vars.append(weight)
                bias = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)

                ###
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.zeros(param[0]), requires_grad=False)

                self.vars_bn.extend([running_mean, running_var])  ## 在列表附加参数

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue

            else:
                raise NotImplementedError

                ## self.net(x_support[i], vars=None, bn_training = True)

    ## x: torch.Size([5, 1, 28, 28])
    ## 构造模型
    def forward(self, x, vars=None, bn_training=True):
        '''
        :param bn_training: set False to not update
        :return:
        '''

        if vars is None:
            vars = self.vars

        idx = 0;
        bn_idx = 0
        for name, param in self.config:
            if name is 'conv2d':
                weight, bias = vars[idx], vars[idx + 1]
                x = F.conv2d(x, weight, bias, stride=param[4], padding=param[5])
                idx += 2

            elif name is 'linear':
                weight, bias = vars[idx], vars[idx + 1]
                x = F.linear(x, weight, bias)
                idx += 2

            elif name is 'bn':
                weight, bias = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                x = x.view(x.size(0), -1)

            elif name is 'relu':
                x = F.relu(x, inplace=[param[0]])

            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError

        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def parameters(self):

        return self.vars


13
from copy import deepcopy
from torch import nn


class Meta(nn.Module):
    """
    Meta-Learner
    """

    def __init__(self, config):
        super(Meta, self).__init__()
        self.update_lr = 0.1  ## learner中的学习率，即\alpha
        self.meta_lr = 1e-3  ## meta-learner的学习率，即\beta
        self.n_way = 5  ## 5种类型
        self.k_shot = 5  ## 一个样本
        self.k_query = 15  ## 15个查询样本
        self.task_num = 4  ## 每轮抽8个任务进行训练
        self.update_step = 5  ## task-level inner update steps
        self.update_step_test = 5  ## 用在finetunning这个函数中

        self.net = Learner(config)  ## base-learner
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_support, y_support, x_query, y_query):
        """
        :param x_spt:   torch.Size([8, 5, 1, 28, 28])
        :param y_spt:   torch.Size([8, 5])
        :param x_qry:   torch.Size([8, 75, 1, 28, 28])
        :param y_qry:   torch.Size([8, 75])
        :return:
        N-way-K-shot
        """
        task_num, ways, shots, h, w = x_support.size()
        #         print("Meta forward")
        querysz = x_query.size(1)  ## 75 = 15*5
        losses_q = [0 for _ in range(self.update_step + 1)]  ## losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            ## 第0步更新
            logits = self.net(x_support[i], vars=None, bn_training=True)  ## return 一个经过各层计算后的y
            ## logits : 5*5的tensor
            loss = F.cross_entropy(logits, y_support[i])  ## 计算Loss值
            grad = torch.autograd.grad(loss, self.net.parameters())  ##计算梯度。如果输入x，输出是y，则求y关于x的导数（梯度）
            tuples = zip(grad, self.net.parameters())  ##将梯度grad和参数\theta一一对应起来
            ## fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], tuples))

            ### 在query集上进行测试，计算准确率
            ## 这一步使用的是更新前的参数
            with torch.no_grad():
                logits_q = self.net(x_query[i], self.net.parameters(),
                                    bn_training=True)  ## logits_q :torch.Size([75, 5])
                loss_q = F.cross_entropy(logits_q, y_query[i])  ## y_query : torch.Size([75])
                losses_q[0] += loss_q  ##将loss存在数组的第一个位置
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  ## size = (75)
                correct = torch.eq(pred_q, y_query[i]).sum().item()  ## item()取出tensor中的数字
                corrects[0] += correct

            ### 在query集上进行测试，计算准确率
            ## 这一步使用的是更新后的参数
            with torch.no_grad():
                logits_q = self.net(x_query[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_query[i])
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_query[i]).sum().item()
                corrects[1] += correct

            for k in range(1, self.update_step):
                logits = self.net(x_support[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_support[i])
                grad = torch.autograd.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], tuples))

                if k < self.update_step - 1:
                    with torch.no_grad():
                        logits_q = self.net(x_query[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_query[i])
                        losses_q[k + 1] += loss_q

                else:
                    logits_q = self.net(x_query[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_query[i])
                    losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_query[i]).sum().item()
                    corrects[k + 1] += correct

        ## 在一组8个任务结束后，求一个平均的loss
        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad()  ## 梯度清零
        loss_q.backward()  ## 计算梯度
        self.meta_optim.step()  ## 用设置好的优化方法来迭代模型参数，这一步是meta步迭代

        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def finetunning(self, x_support, y_support, x_query, y_query):
        assert len(x_support.shape) == 4

        querysz = x_query.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        logits = net(x_support)
        loss = F.cross_entropy(logits, y_support)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        ## 开始训练前的准确率
        with torch.no_grad():
            logits_q = net(x_query, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_query).sum().item()
            corrects[0] += correct

        ## 训练后的准确率
        with torch.no_grad():
            logits_q = net(x_query, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_query).sum().item()
            corrects[1] += correct

        for k in range(1, self.update_step_test):
            logits = net(x_support, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_support)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_query, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_query)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_query).sum().item()
                corrects[k + 1] += correct

        del net

        accs = np.array(corrects) / querysz

        return accs


15
import torch, os
import numpy as np
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
import torch.nn.functional as F


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


n_way = 5
k_spt = 5
epochs = 1000001


def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda:1')
    maml = Meta(config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    #     print(maml)
    #     print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini_train = MiniImagenet('./datasets/mini-imagenet/', mode='train', n_way=5, k_shot=5,
                              k_query=15,
                              batchsz=10000, resize=84)
    mini_test = MiniImagenet('./datasets/mini-imagenet/', mode='test', n_way=5, k_shot=5,
                             k_query=15,
                             batchsz=100, resize=84)

    for epoch in range(epochs // 10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini_train, 4, shuffle=True, num_workers=4, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 100 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 1000 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=4, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)


main()




