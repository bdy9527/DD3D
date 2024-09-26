import random
import os
import time
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from torchmetrics.functional.classification import multiclass_accuracy

import accelerate
import provider
from CINR import InvariantWrapper
from networks import PointNet, DGCNN, PointCNN, PointNetSeg


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class SegDataset(Dataset):
    def __init__(self, images, segments, labels): # images: n x c x h x w tensor
        self.images = images.detach()
        self.segments = segments.detach()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.segments[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def save_and_print(dirname, msg):
    if not os.path.isfile(dirname):
        f = open(dirname, "w")
        f.write(str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())))
        f.write("\n")
        f.close()
    f = open(dirname, "a")
    f.write(str(msg))
    f.write("\n")
    f.close()
    print(msg)


def get_clouds(clouds_all, indices_class, c, n):
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return clouds_all[idx_shuffle]


def get_segments(clouds_all, segs_all, labels_all, indices_class, c, n):
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return clouds_all[idx_shuffle], segs_all[idx_shuffle], labels_all[idx_shuffle]


''' Initilization '''

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    accelerate.utils.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


''' Gradient Matching '''

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 1e-6))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, dis_metric):
    dis = 0.

    gradient = [0., 0., 0., 0.]
    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 1e-6)

    else:
        exit('unknown distance function: %s'%dis_metric)

    return dis


def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):  # 2874 shapes in total
        if class_choice is None or label[shape_idx] == class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
                iou = 1 if U == 0 else  I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))
        else:
            continue
    return np.asarray(shape_ious)


def epoch_seg(mode, dataloader, net, optimizer, criterion, args, aug=False):
    loss_avg, num_exp = 0., 0

    if mode == 'train':
        net.train()
        device = args.eval_device
    else:
        net.eval()
        device = args.eval_device

    net = net.to(device)
    criterion = criterion.to(device)

    targets = []
    preds = []
    labels = []

    for i_batch, datum in enumerate(dataloader):
        cld = datum[0].float().to(device)
        seg = datum[-2].long().to(device)
        lab = datum[-1].long().to(device)
        n_b = lab.shape[0]

        output = net(cld, lab)
        loss = criterion(output.view(-1, 50), seg.view(-1))

        pred = torch.argmax(output, dim=-1)

        targets.append(seg.data.cpu().numpy())
        preds.append(pred.data.cpu().numpy())
        labels.append(lab.data.cpu().numpy())

        loss_avg += loss.item() * n_b
        num_exp  += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp

    test_pred, test_seg, test_label = np.concatenate(preds), np.concatenate(targets), np.concatenate(labels)
    iou = calculate_shape_IoU(test_pred, test_seg, test_label, class_choice=None)
    miou = np.mean(iou)

    preds = torch.from_numpy(test_pred).view(-1).long()
    targets = torch.from_numpy(test_seg).view(-1).long()
    oa = multiclass_accuracy(preds, targets, num_classes=args.seg_classes, average='micro').item()

    return loss_avg, miou, oa


def epoch(mode, dataloader, net, optimizer, criterion, args, aug=True):
    loss_avg, num_exp = 0., 0

    if mode == 'train':
        net.train()
        device = args.eval_device
    else:
        net.eval()
        device = args.eval_device

    net = net.to(device)
    criterion = criterion.to(device)

    targets = []
    preds = []

    for i_batch, datum in enumerate(dataloader):
        data = datum[0].float()
        label = datum[-1].long()
        n_b = label.shape[0]

        if aug:
            data = data.cpu().numpy()
            data = provider.random_scale_point_cloud(data)
            data = provider.jitter_point_cloud(data)
            data = provider.rotate_perturbation_point_cloud(data) 
            data = torch.from_numpy(data).float()

        data, label = data.to(device), label.to(device)

        output = net(data)
        loss = criterion(output, label)
        pred = torch.argmax(output, dim=-1)

        targets += label.data.cpu().tolist()
        preds += pred.data.cpu().tolist()

        loss_avg += loss.item() * n_b
        num_exp  += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp

    preds = torch.tensor(preds).view(-1).long()
    targets = torch.tensor(targets).view(-1).long()
    oa = multiclass_accuracy(preds, targets, num_classes=args.num_classes, average='micro').item()
    macc = multiclass_accuracy(preds, targets, num_classes=args.num_classes, average='macro').item()

    return loss_avg, macc, oa


def evaluate_segment(it_eval, clouds_train, segment_train, labels_train, testloader, args):

    lr = float(args.lr_net)
    wd = float(args.wd_net)
    Epoch = 1000

    net = PointNetSeg(input_dim=args.channel, output_dim=args.seg_classes)

    if args.pca:
        net = InvariantWrapper(net, hidden_dim=32, w0=10.)
    net = net.to(args.eval_device)

    criterion = nn.CrossEntropyLoss().to(args.eval_device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Epoch, eta_min=1e-3)

    dst_train = SegDataset(clouds_train, segment_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=32, shuffle=True)

    start = time.time()
    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, miou_train, oa_train = epoch_seg('train', trainloader, net, optimizer, criterion, args, aug = False)
        scheduler.step()

    time_train = time.time() - start
    loss_test, miou_test, oa_test = epoch_seg('test', testloader, net, optimizer, criterion, args, aug = False)

    save_and_print(args.log_path, '%s Evaluate_%02d: epoch = %04d, train time = %d s, train loss = %.6f, train miou = %.4f train oa = %.4f, test miou = %.4f test oa = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, miou_train, oa_train, miou_test, oa_test))

    return net, miou_test, oa_test


def evaluate_synset(it_eval, clouds_train, labels_train, testloader, args):

    lr = float(args.lr_net)
    wd = float(args.wd_net)
    Epoch = int(args.epoch_eval_train)

    if args.eval_model == 'pointnet':
        net = PointNet(input_dim=args.channel, output_dim=args.num_classes)
    elif args.eval_model == 'dgcnn':
        net = DGCNN(input_dim=args.channel, output_dim=args.num_classes)
    elif args.dist_model == 'pointcnn':
        net = PointCNN(input_dim=args.channel, output_dim=args.num_classes)
    else:
        print('No existence of evaluation model')
        return

    if args.pca:
        net = InvariantWrapper(net, hidden_dim=32, w0=args.w0_sign)
    net = net.to(args.eval_device)

    criterion = nn.CrossEntropyLoss().to(args.eval_device)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    if args.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=Epoch // 2, gamma=0.1)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=Epoch, eta_min=1.0e-4)

    dst_train = TensorDataset(clouds_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True)

    start = time.time()
    oa_max_test = 0.
    acc_max_test = 0.

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train, oa_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
        scheduler.step()

        loss_test, acc_test, oa_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
        if oa_test > oa_max_test:
            oa_max_test = oa_test
            acc_max_test = acc_test

    time_train = time.time() - start

    save_and_print(args.log_path, '%s Evaluate_%02d: epoch = %04d, train time = %d s, train loss = %.6f, train acc = %.4f train oa = %.4f, test acc = %.4f test oa = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, oa_train, acc_max_test, oa_max_test))

    return net, acc_max_test, oa_max_test



def get_loops(ipc):
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 5
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)

    return outer_loop, inner_loop


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

