import os
import yaml
import copy
from tqdm import tqdm, trange
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from networks import PointNet
from CINR import SirenNet, SirenWrapper, InvariantWrapper

from utils import seed_everything, count_parameters, match_loss, epoch, evaluate_synset, save_and_print, get_clouds, get_time, TensorDataset
import provider

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


class PointCloudDataset(Dataset):
    def __init__(self, root):
        self.ptn, self.val, self.vec, self.labels = torch.load(root)
        #self.ptn, self.labels = torch.load(root)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ptn[idx], self.val[idx], self.vec[idx], self.labels[idx]


def inner_loop(train_loader, net, optimizer, criterion, args):
    net = net.to(args.device)
    net = net.train()

    for i_batch, datum in enumerate(train_loader):
        data = datum[0].float().to(args.device)
        label = datum[-1].long().to(args.device)

        output = net(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(args):

    seed_everything(args.seed)

    args.device = 'cuda:{}'.format(args.cuda)
    args.eval_device = args.device

    '''Setting Path'''
    if args.init:
        args.save_path = os.path.join('syn_data', args.dataset, '2024-09-19_05:21:52')
    else:
        args.save_path = os.path.join('syn_data', args.dataset, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    ''' Path '''
    args.log_path = os.path.join('log', args.dataset, 'log.txt')

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.dataset == 'modelnet':
        args.Iteration = 400
        args.num_classes = 40
        args.batch_real = 32
        args.w0_inr = 100.

        if args.pca:

            if args.cpc == 1:
                args.outer_loop, args.inner_loop = 5, 5
            elif args.cpc == 10:
                args.outer_loop, args.inner_loop = 20, 10
            elif args.cpc == 50:
                args.outer_loop, args.inner_loop = 40, 5
            else:
                return
    
        else:

            if args.cpc == 1:
                args.outer_loop, args.inner_loop = 10, 1
            elif args.cpc == 10:
                args.outer_loop, args.inner_loop = 20, 1
            elif args.cpc == 50:
                args.outer_loop, args.inner_loop = 40, 5
            else:
                return

    elif args.dataset == 'shapenet':
        args.num_classes = 16

    elif args.dataset == 'scanobject':
        args.Iteration = 200
        args.num_classes = 15
        args.batch_real = 32
        args.w0_inr = 50.

        if args.pca:

            if args.cpc == 1:
                args.outer_loop, args.inner_loop = 5, 5
            elif args.cpc == 10:
                args.outer_loop, args.inner_loop = 20, 10
            elif args.cpc == 50:
                args.outer_loop, args.inner_loop = 40, 5
            else:
                return

        else: 

            if args.cpc == 1:
                args.outer_loop, args.inner_loop = 10, 1
            elif args.cpc == 10:
                args.outer_loop, args.inner_loop = 20, 10
            elif args.cpc == 50:
                args.outer_loop, args.inner_loop = 30, 5
            else:
                return

    elif args.dataset == 'mvp100full':
        args.Iteration = 400
        args.num_classes = 100
        args.batch_real = 128
        args.w0_inr = 100.

        if args.cpc == 1:
            args.outer_loop, args.inner_loop = 10, 1
        elif args.cpc == 10:
            args.outer_loop, args.inner_loop = 20, 1
        elif args.cpc == 50:
            args.outer_loop, args.inner_loop = 40, 5
        else:
            return

    else:
        print('No exitence of dataset {}'.format(args.dataset))
        return

    print(args)

    if args.pca:
        train_data = PointCloudDataset('dataset/processed/{}_train_so3.pt'.format(args.dataset))
        test_data = PointCloudDataset('dataset/processed/{}_test_so3.pt'.format(args.dataset))
        print('########## PCA Transformation ##########')
        clouds_all = [torch.mm(train_data[i][0], train_data[i][2]) for i in range(len(train_data))]
        clouds_test = [torch.mm(test_data[i][0], test_data[i][2]) for i in range(len(test_data))]
    else:
        train_data = PointCloudDataset('dataset/processed/{}_train_raw.pt'.format(args.dataset))
        test_data = PointCloudDataset('dataset/processed/{}_test_raw.pt'.format(args.dataset))
        print('########## None Transformation ##########')
        clouds_all = [train_data[i][0] for i in range(len(train_data))]
        clouds_test = [test_data[i][0] for i in range(len(test_data))]

    labels_all = [train_data[i][-1] for i in range(len(train_data))]
    labels_test = [test_data[i][-1] for i in range(len(test_data))]

    indices_class = [[] for c in range(args.num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    num_point = clouds_all[0].shape[0]
    clouds_all = torch.stack(clouds_all, dim=0)
    print(clouds_all.shape)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    clouds_test = torch.stack(clouds_test, dim=0)
    print(clouds_test.shape)
    labels_test = torch.tensor(labels_test, dtype=torch.long, device=args.device)

    test_data = TensorDataset(clouds_test, labels_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_train, shuffle=False)


    ''' Initilization '''
    syn_label = torch.tensor([[i] * args.cpc for i in range(args.num_classes)], device=args.device, requires_grad=False).long().view(-1)

    if args.init:
        ckpts = [ckpt for ckpt in os.listdir(args.save_path) if 'pth' in ckpt]
        # recent_ckpt = ckpts[-1]
        recent_ckpt = '2024-09-19_06:51:52.pth'
        wrapper = torch.load(os.path.join(args.save_path, recent_ckpt), map_location=args.device, weights_only=False)
    else:
        siren = SirenNet(dim_in=1, dim_hidden=args.hd_inr, dim_out=3, num_layers=args.layers, w0_initial=args.w0_inr, final_activation=nn.Tanh())
        wrapper = SirenWrapper(siren, args.num_classes, args.cpc, args.hd_inr, num_layers=args.layers)

    optimizer = torch.optim.Adam(wrapper.parameters(), lr=args.lr_inr, weight_decay=args.wd_inr)
    scheduler = StepLR(optimizer, step_size=args.Iteration // 2, gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    ''' Experiment '''
    for it in range(args.Iteration+1):

        if (it+1) % 10 == 0:

            # save data and model
            time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            torch.save(wrapper, os.path.join(args.save_path, time + '.pth'))

        if (it+1) % 50 == 0:

            indices = torch.tensor([indice for indice in range(syn_label.shape[0])]).long().to(args.device)
            noise = torch.rand(len(indices), args.eval_num_point, 1, device=args.device)
            syn_cloud_eval = wrapper(noise, indices).detach()
            syn_cloud_eval = torch.tensor(provider.normalize_data(syn_cloud_eval.cpu().numpy())).float().to(args.device)
            syn_label_eval = syn_label.detach()

            _, _, oa_test = evaluate_synset(it, syn_cloud_eval, syn_label_eval, test_loader, args)

            save_and_print(args.log_path, 'Iteration %d, OA_test = %.4f\n-------------------------'%(it, oa_test))

        ''' Train synthetic data '''
        net = PointNet(input_dim=args.channel, output_dim=args.num_classes)

        if args.pca:
            net = InvariantWrapper(net, hidden_dim=32, w0=args.w0_sign)
        net = net.to(args.device)
        
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        scheduler_net = StepLR(optimizer_net,
                               step_size=(args.outer_loop * args.inner_loop) // 2,
                               gamma=0.1)
        loss_avg = 0.

        for ol in range(args.outer_loop):

            net.train()
            dd_loss = torch.tensor(0.0).to(args.device)
            for c in range(args.num_classes):

                net = net.to(args.device)
                wrapper = wrapper.to(args.device)

                cld_real = get_clouds(clouds_all, indices_class, c, args.batch_real)
                lab_real = torch.ones((cld_real.shape[0],), device=args.device, dtype=torch.long) * c

                if args.train_num_point < num_point:
                    ind = torch.randperm(num_point)[:args.train_num_point]
                    cld_real = cld_real[:, ind, :]
                    cld_real = provider.normalize_data(cld_real.numpy())
                else:
                    cld_real = cld_real.numpy()

                if args.aug:
                    cld_real = provider.random_scale_point_cloud(cld_real)
                    cld_real = provider.jitter_point_cloud(cld_real)
                    cld_real = provider.rotate_perturbation_point_cloud(cld_real)

                cld_real = torch.from_numpy(cld_real).float()
                cld_real = cld_real.to(args.device)

                indices = torch.tensor([i for i in range(c * args.cpc, (c + 1) * args.cpc)]).long().to(args.device)
                noise = torch.rand(len(indices), args.train_num_point, 1, device=args.device)
                cld_syn = wrapper(noise, indices)
                lab_syn = syn_label[indices]

                opt_real  = net(cld_real)
                loss_real = criterion(opt_real, lab_real)
                gw_real   = torch.autograd.grad(loss_real, net_parameters)
                gw_real   = list((_.detach().clone() for _ in gw_real))

                opt_syn  = net(cld_syn)
                loss_syn = criterion(opt_syn, lab_syn)
                gw_syn   = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                dd_loss += match_loss(gw_syn, gw_real, args.dis_metric)

            optimizer.zero_grad()
            dd_loss.backward()
            optimizer.step()
            loss_avg += dd_loss.item()

            if ol == args.outer_loop - 1:
                break

            ''' update network '''
            indices = torch.tensor([indice for indice in range(syn_label.shape[0])]).long().to(args.device)
            noise = torch.rand(len(indices), args.train_num_point, 1, device=args.device)
            syn_cloud_eval = wrapper(noise, indices).detach()
            syn_cloud_eval = torch.tensor(provider.normalize_data(syn_cloud_eval.cpu().numpy())).float().to(args.device)
            syn_label_eval = syn_label.detach()

            train_syn_train = TensorDataset(syn_cloud_eval, syn_label_eval)
            train_syn_loader = torch.utils.data.DataLoader(train_syn_train, batch_size=256, shuffle=True)
            for il in range(args.inner_loop):
                inner_loop(train_syn_loader, net, optimizer_net, criterion, args)
                scheduler_net.step()
                
        scheduler.step()

        loss_avg /= (args.num_classes * args.outer_loop)

        if it%10 == 0:
            save_and_print(args.log_path, '%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--pca', action='store_true', help='PCA Transformation')
    parser.add_argument('--aug', action='store_true', help='Augmenting data during distillation')
    parser.add_argument('--init', action='store_true', help='Continuis training from checkpoints')
    parser.add_argument('--dis_metric', default='ours')
    parser.add_argument('--syn', default='inr')  # inr or fix
    parser.add_argument('--dataset', default='modelnet')  # modelnet scanobject
    parser.add_argument('--cpc', type=int, default=1, help='point clouds per class')

    parser.add_argument('--Iteration', type=int, default=200, help='training iterations')
    parser.add_argument('--batch_syn', type=int, default=0)
    parser.add_argument('--batch_real', type=int, default=32, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--lr_net', type=float, default=1e-3, help='learning rate for updating network parameters')
    parser.add_argument('--wd_net', type=float, default=5e-4, help='weight decay for updating network parameters')
    parser.add_argument('--dist_model', default='pointnet', help='choose a model for distillation')
    parser.add_argument('--eval_model', default='pointnet', help='choose a model for evaluation')

    # parameters for DD3D
    parser.add_argument('--lr_inr', type=float, default=1e-3, help='learning rate for inr')
    parser.add_argument('--wd_inr', type=float, default=5e-4, help='weight decay for inr')
    parser.add_argument('--layers', type=int, default=2, help='layers for SIREN')
    parser.add_argument('--hd_inr', type=int, default=256, help='hidden dim for SIREN')
    parser.add_argument('--w0_inr', type=float, default=100., help='w0_init for SIREN')
    parser.add_argument('--w0_sign', type=float, default=1., help='w0_init for Sign Predictor')
    parser.add_argument('--train_num_point', type=int, default=1024, help='points for distillation')
    parser.add_argument('--eval_num_point', type=int, default=1024, help='points for evaluation')

    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR')

    parser.add_argument('--epoch_eval_train', type=int, default=200, help='epochs to train a model with synthetic data')
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')

    args = parser.parse_args()

    main(args)

