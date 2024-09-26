import os
import time
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

from networks import PointNetSeg
from CINR import SirenNet, SirenWrapper

from utils import seed_everything, match_loss, get_time, SegDataset, evaluate_segment, get_segments
import provider

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


class PointCloudSegDataset(Dataset):
    def __init__(self, root):
        self.ptn, self.val, self.vec, self.segs, self.labels = torch.load(root)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ptn[idx], self.val[idx], self.vec[idx], self.segs[idx], self.labels[idx]


def generate_noise_and_seg_label(subset_syn_label, train_num_point):
    label = copy.deepcopy(subset_syn_label).detach().cpu()
    B, N = len(label), train_num_point 
    noise = torch.rand(B, N, 1, device='cpu')
    seg_label = torch.zeros((B, N), requires_grad=False, device='cpu').long()

    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
    proportion = [[0.45078503, 0.32637933, 0.12735743, 0.09547821],
                  [0.06712292, 0.93287708],
                  [0.73870295, 0.26129705],
                  [0.05368124, 0.07117755, 0.16026644, 0.71487476],
                  [0.34000747, 0.41298617, 0.21362449, 0.03338187],
                  [0.59840199, 0.26434659, 0.13725142],
                  [0.08631071, 0.19352421, 0.72016508],
                  [0.50879376, 0.49120624],
                  [0.16456666, 0.5967006,  0.01242581, 0.22630694],
                  [0.53473962, 0.46526038],
                  [0.04836895, 0.03854188, 0.23353102, 0.01477132, 0.00699115, 0.65779569],
                  [0.06083784, 0.93916216],
                  [0.6598784,  0.28620636, 0.05391524],
                  [0.70912905, 0.18180339, 0.10906756],
                  [0.1130593,  0.81221349, 0.07472721],
                  [0.74166941, 0.23007729, 0.0282533]]

    for shape_ind, lab in enumerate(label):
        prop = proportion[lab]
        prop_start = 0.
        for part_ind, part in enumerate(range(index_start[lab], index_start[lab]+seg_num[lab])):
            prop_end = prop_start + prop[part_ind]
            point_index = torch.where((noise[shape_ind] > prop_start) & (noise[shape_ind] <= prop_end))[0]
            seg_label[shape_ind, point_index] = part

            prop_start = prop_end

    return noise, seg_label


def inner_seg_loop(train_loader, net, optimizer, criterion, args):
    net = net.to(args.device)
    net = net.train()

    for i_batch, datum in enumerate(train_loader):
        cld = datum[0].float().to(args.device)
        seg = datum[1].long().to(args.device)
        lab = datum[2].long().to(args.device)

        output = net(cld, lab)
        loss = criterion(output.view(-1, 50), seg.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(args):

    args.device = 'cuda:{}'.format(args.cuda)

    seed_everything(args.seed)

    args.save_path = os.path.join('syn_data', args.dataset, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    ''' Path '''
    args.log_path = os.path.join('log', args.dataset, 'log.txt')

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.dataset == 'shapenet':
        args.num_classes = 16
        args.seg_classes = 50

        args.Iteration = 400
        args.batch_real = 32
        args.w0_inr = 400.

        if args.cpc == 1:
            args.outer_loop, args.inner_loop = 10, 1
        elif args.cpc == 10:
            args.outer_loop, args.inner_loop = 10, 2
        else:
            return

    print(args)

    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

    if args.pca:
        train_data = PointCloudSegDataset('dataset/processed/segment_train_so3.pt')
        test_data = PointCloudSegDataset('dataset/processed/segment_test_so3.pt')
        print('########## PCA Transformation ##########')
        clouds_all = [torch.mm(train_data[i][0], train_data[i][2]) for i in range(len(train_data))]
        clouds_test = [torch.mm(test_data[i][0], test_data[i][2]) for i in range(len(test_data))]
    else:
        train_data = PointCloudSegDataset('dataset/processed/segment_train_raw.pt')
        test_data = PointCloudSegDataset('dataset/processed/segment_test_raw.pt')
        print('########## None Transformation ##########')
        clouds_all = [train_data[i][0] for i in range(len(train_data))]
        clouds_test = [test_data[i][0] for i in range(len(test_data))]

    segs_all = [train_data[i][-2] for i in range(len(train_data))]
    segs_test = [test_data[i][-2] for i in range(len(test_data))]

    labels_all = [train_data[i][-1] for i in range(len(train_data))]
    labels_test = [test_data[i][-1] for i in range(len(test_data))]

    indices_class = [[] for c in range(args.num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    clouds_all = torch.stack(clouds_all, dim=0)
    segs_all = torch.stack(segs_all, dim=0)
    labels_all = torch.tensor(labels_all).long()

    clouds_test = torch.stack(clouds_test, dim=0)
    segs_test = torch.stack(segs_test, dim=0)
    labels_test = torch.tensor(labels_test).long()

    test_data = SegDataset(clouds_test, segs_test, labels_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_train, shuffle=False)


    ''' Initilization '''
    syn_label = torch.tensor([[i] * args.cpc for i in range(args.num_classes)], device=args.device, 
                             requires_grad=False).long().view(-1)

    siren = SirenNet(dim_in=1, dim_hidden=args.hd_inr, dim_out=3, num_layers=args.layers, w0_initial=args.w0_inr, final_activation=nn.Tanh())
    wrapper = SirenWrapper(siren, args.num_classes, args.cpc, args.hd_inr, num_layers=args.layers).to(args.device)

    optimizer = torch.optim.Adam(wrapper.parameters(), lr=args.lr_inr, weight_decay=args.wd_inr)
    scheduler = StepLR(optimizer, step_size=args.Iteration // 2, gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)


    ''' Experiment '''
    for it in range(args.Iteration+1):

        if (it+1) % 10 == 0:

            # save data and model
            time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            torch.save(wrapper, os.path.join(args.save_path, time + '.pth'))

        if it % 50 == 0:

            indices = torch.tensor([indice for indice in range(syn_label.shape[0])]).long().to(args.device)
            syn_label_eval = syn_label.detach()
            noise, syn_seg_eval = generate_noise_and_seg_label(syn_label_eval, args.train_num_point)
            noise, syn_seg_eval = noise.to(args.device), syn_seg_eval.to(args.device)
            syn_cloud_eval = wrapper(noise, indices)

            _, miou_test, oa_test = evaluate_segment(0, syn_cloud_eval, syn_seg_eval, syn_label_eval, 
                                                     test_loader, args)

            print(miou_test)


        ''' Train synthetic data '''
        net = PointNetSeg(input_dim=args.channel, output_dim=args.seg_classes)
        net = net.to(args.device)

        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        scheduler_net = StepLR(optimizer_net,
                               step_size=(args.outer_loop * args.inner_loop) // 2,
                               gamma=0.1)
        loss_avg = 0.

        for ol in range(args.outer_loop):

            BN_flag = False
            BNSizePC = 8
            for module in net.modules():
                if 'BatchNorm' in module._get_name(): #BatchNorm
                    BN_flag = True
            if BN_flag:
                cld_real = torch.cat([get_segments(clouds_all, segs_all, labels_all, indices_class, c, BNSizePC)[0] \
                                        for c in range(args.num_classes)], dim=0)
                label_real = torch.from_numpy(np.concatenate([([c]*BNSizePC) for c in range(args.num_classes)], axis=0)).long()
                cld_real = cld_real.to(args.device)
                label_real = label_real.to(args.device)

                net.train() # for updating the mu, sigma of BatchNorm
                net(cld_real, label_real) # get running mu, sigma
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  #BatchNorm
                        module.eval() # fix mu and sigma of every BatchNorm layer


            dd_loss = torch.tensor(0.0).to(args.device)

            for c in range(args.num_classes):

                net = net.to(args.device)
                wrapper = wrapper.to(args.device)

                cld_real, seg_real, lab_real = get_segments(clouds_all, segs_all, labels_all, indices_class, c, args.batch_real)

                if args.aug:
                    cld_real = cld_real.numpy()
                    cld_real = provider.random_scale_point_cloud(cld_real)
                    cld_real = provider.jitter_point_cloud(cld_real)
                    cld_real = provider.rotate_perturbation_point_cloud(cld_real)

                cld_real = torch.from_numpy(cld_real).float()
                cld_real = cld_real.to(args.device)
                seg_real = seg_real.to(args.device)
                lab_real = lab_real.to(args.device)

                indices = torch.tensor([i for i in range(c * args.cpc, (c + 1) * args.cpc)]).long().to(args.device)
                lab_syn = syn_label[indices]
                noise, seg_syn = generate_noise_and_seg_label(lab_syn, args.train_num_point)
                noise, seg_syn = noise.to(args.device), seg_syn.to(args.device)
                cld_syn = wrapper(noise, indices)

                # opt_real  = net(cld_real, lab_real)
                # loss_real = criterion(opt_real.view(-1, args.seg_classes), seg_real.view(-1))
                # gw_real   = torch.autograd.grad(loss_real, net_parameters)
                # gw_real   = list((_.detach().clone() for _ in gw_real))

                # opt_syn  = net(cld_syn, lab_syn)
                # loss_syn = criterion(opt_syn.view(-1, args.seg_classes), seg_syn.view(-1))
                # gw_syn   = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                # dd_loss += match_loss(gw_syn, gw_real, args.dis_metric)

                if ol % 2 < 1:
                    opt_real  = net(cld_real, lab_real)
                    loss_real = criterion(opt_real.view(-1, args.seg_classes), seg_real.view(-1))
                    gw_real   = torch.autograd.grad(loss_real, net_parameters)
                    gw_real   = list((_.detach().clone() for _ in gw_real))

                    opt_syn  = net(cld_syn, lab_syn)
                    loss_syn = criterion(opt_syn.view(-1, args.seg_classes), seg_syn.view(-1))
                    gw_syn   = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    dd_loss += match_loss(gw_syn, gw_real, args.dis_metric)

                else:

                    for part_class in range(index_start[c], index_start[c] + seg_num[c]):
                        seg_real_mask, seg_syn_mask = copy.deepcopy(seg_real), copy.deepcopy(seg_syn)
                        seg_real_mask[seg_real_mask != part_class] = -1
                        seg_syn_mask[seg_syn_mask != part_class] = -1

                        opt_real  = net(cld_real, lab_real)
                        loss_real = F.cross_entropy(opt_real.view(-1, args.seg_classes), 
                                                    seg_real_mask.view(-1), ignore_index=-1)
                        gw_real   = torch.autograd.grad(loss_real, net_parameters)
                        gw_real   = list((_.detach().clone() for _ in gw_real))

                        opt_syn  = net(cld_syn, lab_syn)
                        loss_syn = F.cross_entropy(opt_syn.view(-1, args.seg_classes),
                                                   seg_syn_mask.view(-1), ignore_index=-1)
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
            syn_label_eval = syn_label.detach()
            noise, syn_seg_eval = generate_noise_and_seg_label(syn_label_eval, args.train_num_point)
            noise, syn_seg_eval = noise.to(args.device), syn_seg_eval.to(args.device)
            syn_cloud_eval = wrapper(noise, indices)
            syn_cloud_eval = torch.tensor(provider.normalize_data(
                syn_cloud_eval.detach().cpu().numpy())).float().to(args.device)

            train_syn_train = SegDataset(syn_cloud_eval, syn_seg_eval, syn_label_eval)
            train_loader = torch.utils.data.DataLoader(train_syn_train, batch_size=args.batch_train, shuffle=True)
            for il in range(args.inner_loop):
                inner_seg_loop(train_loader, net, optimizer_net, criterion, args)

        loss_avg /= (args.num_classes * args.outer_loop)

        if it % 10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--pca', action='store_true', help='PCA Transformation')
    parser.add_argument('--aug', action='store_true', help='Augmenting data during distillation')
    parser.add_argument('--dis_metric', default='ours')
    parser.add_argument('--dataset', default='shapenet')
    parser.add_argument('--cpc', type=int, default=5, help='point clouds per class')

    parser.add_argument('--Iteration', type=int, default=200, help='training iterations')
    parser.add_argument('--batch_syn', type=int, default=0)
    parser.add_argument('--batch_real', type=int, default=32, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=32, help='batch size for training networks')
    parser.add_argument('--lr_net', type=float, default=1e-3, help='learning rate for updating network parameters')
    parser.add_argument('--wd_net', type=float, default=5e-4, help='weight decay for updating network parameters')

    parser.add_argument('--lr_inr', type=float, default=5e-4, help='learning rate for inr')
    parser.add_argument('--wd_inr', type=float, default=1e-5, help='weight decay for inr')
    parser.add_argument('--layers', type=int, default=2, help='layers for SIREN')
    parser.add_argument('--hd_inr', type=int, default=256, help='hidden dim for SIREN')
    parser.add_argument('--w0_inr', type=float, default=400., help='w0_init for SIREN')
    parser.add_argument('--w0_sign', type=float, default=1., help='w0_init for Sign Predictor')
    parser.add_argument('--train_num_point', type=int, default=2048, help='points for distillation')
    parser.add_argument('--eval_num_point', type=int, default=2048, help='points for evaluation')

    parser.add_argument('--channel', type=int, default=3)

    args = parser.parse_args()

    main(args)


