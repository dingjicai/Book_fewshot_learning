import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from collections import defaultdict
from itertools import cycle
from torch.nn import functional as F

#超参数
num_classes = 10
sup_batch_size = 32
usp_batch_size = 32
num_labels = 4000
drop_ratio = 0.0
usp_weight = 30.0
mixup_alpha = 1.0
ema_decay = 0.97
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
nesterov = True
epochs = 400
min_lr= 1e-4
print_freq = 100
rampup_length = 80
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mixup_one_target(x, y, alpha=1.0, device='cuda', is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1-lam)

    index = torch.randperm(x.size(0)).to(device)

    mixed_x = lam*x + (1-lam)*x[index, :]
    mixed_y = lam*y + (1-lam)*y[index]
    return mixed_x, mixed_y, lam

def one_hot(targets, nClass):
    logits = torch.zeros(targets.size(0), nClass).to(targets.device)
    return logits.scatter_(1,targets.unsqueeze(1),1)

def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper

class TransformTwice:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def encode_label(label):
    return -1 * (label +1)

def decode_label(label):
    return -1 * label -1

def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper

def split_relabel_data(np_labs, labels, label_per_class,
                        num_classes):
    """ Return the labeled indexes and unlabeled_indexes
    """
    labeled_idxs = []
    unlabed_idxs = []
    for id in range(num_classes):
        indexes = np.where(np_labs==id)[0]
        np.random.shuffle(indexes)
        labeled_idxs.extend(indexes[:label_per_class])
        unlabed_idxs.extend(indexes[label_per_class:])
    np.random.shuffle(labeled_idxs)
    np.random.shuffle(unlabed_idxs)
    ## relabel dataset
    for idx in unlabed_idxs:
        labels[idx] = encode_label(labels[idx])
    return labeled_idxs, unlabed_idxs

def read_cifar10(n_labels, data_root='./'):
    channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                         std = [0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                   transform=eval_transform)

    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(trainset.targets),
                                    trainset.targets,
                                    label_per_class,
                                    num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }

def data_loaders(trainset, evalset, label_idxs, unlab_idxs,
                      num_classes):

    trainset.transform = TransformTwice(trainset.transform)
    ## supervised batch loader
    label_sampler = SubsetRandomSampler(label_idxs)
    label_batch_sampler = BatchSampler(label_sampler, sup_batch_size,
                                       drop_last=True)
    label_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=label_batch_sampler,
                                          pin_memory=True)
    ## unsupervised batch loader
    unlab_idxs += label_idxs
    unlab_sampler = SubsetRandomSampler(unlab_idxs)
    unlab_batch_sampler = BatchSampler(unlab_sampler, usp_batch_size,
                                       drop_last=True)
    unlab_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=unlab_batch_sampler,
                                          pin_memory=True)
    ## test batch loader
    eval_loader = torch.utils.data.DataLoader(evalset,
                                           batch_size=sup_batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           drop_last=False)
    return label_loader, unlab_loader, eval_loader

#卷积块
class CNN_block(nn.Module):

    def __init__(self, in_plane, out_plane, kernel_size, padding, activation):
        super(CNN_block, self).__init__()

        self.act = activation
        self.conv = nn.Conv2d(in_plane,
                              out_plane,
                              kernel_size,
                              padding=padding)

        self.bn = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
#网络结构
class CNN(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, drop_ratio=0.0):
        super(CNN, self).__init__()

        self.in_plane = 3
        self.out_plane = 128
        self.act = nn.LeakyReLU(0.1)
        self.layer1 = self._make_layer(block, num_blocks[0], 128, 3, padding=1)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(drop_ratio)
        self.layer2 = self._make_layer(block, num_blocks[1], 256, 3, padding=1)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(drop_ratio)
        self.layer3 = self._make_layer(block, num_blocks[2],
                                       [512, 256, self.out_plane],
                                       [3, 1, 1],
                                       padding=0)
        self.ap3 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.out_plane, num_classes)

    def _make_layer(self, block, num_blocks, planes, kernel_size, padding=1):
        if isinstance(planes, int):
            planes = [planes] * num_blocks
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_blocks
        layers = []
        for plane, ks in zip(planes, kernel_size):
            layers.append(block(self.in_plane, plane, ks, padding, self.act))
            self.in_plane = plane
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.mp1(out)
        out = self.drop1(out)
        out = self.layer2(out)
        out = self.mp2(out)
        out = self.drop1(out)
        out = self.layer3(out)
        out = self.ap3(out)
        out = out.view(out.size(0), -1)
        return self.fc1(out)

def convLarge(num_classes, drop_ratio=0.0):
    return CNN(CNN_block, [3, 3, 3], num_classes, drop_ratio)
#训练器
class Trainer:

    def __init__(self, model, ema_model, optimizer):
        print("MixMatch")
        self.model      = model
        self.ema_model  = ema_model
        self.optimizer  = optimizer
        self.cons_weight = usp_weight
        self.ema_decay   = ema_decay
        self.rampup      = exp_rampup(rampup_length)
        self.device      = device
        self.global_step = 0
        self.epoch       = 0
        self.alpha       = mixup_alpha
        self.temp        = 0.5  # temperature for sharpening

    def train_iteration(self, label_loader, unlab_loader):
        loop_info = defaultdict(list)
        batch_idx, label_n, unlab_n = 0, 0, 0
        for ((x1,_), label_y), ((u1,u2), unlab_y) in zip(cycle(label_loader), unlab_loader):
            self.global_step += 1; batch_idx+=1;
            label_x, unlab_x1, unlab_x2 = x1.to(device), u1.to(device), u2.to(device)
            label_y, unlab_y = label_y.to(device), unlab_y.to(device)
            ##=== decode targets ===
            self.decode_targets(unlab_y)
            lbs, ubs = x1.size(0), u1.size(0)

            ##=== Training Phase ===
            ## update mean-teacher
            self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)
            ## label guessing
            with torch.no_grad():
                outputs1 = self.model(unlab_x1)
                outputs2 = self.model(unlab_x2)
                avg_outputs = (torch.softmax(outputs1,dim=1) + torch.softmax(outputs2,dim=1))/2
                sharpen_p   = avg_outputs**(1/self.temp)
                sharpen_pslab = sharpen_p / sharpen_p.sum(dim=1, keepdim=True)
                sharpen_pslab = sharpen_pslab.detach()
                labeled_label = one_hot(label_y, outputs1.size(1))
            ## concat data
            input_x = torch.cat([label_x, unlab_x1, unlab_x2], dim=0)
            input_y = torch.cat([labeled_label, sharpen_pslab, sharpen_pslab], dim=0)
            ## forward
            mixed_x, mixed_y, lam = mixup_one_target(input_x, input_y,
                                                     self.alpha,
                                                     self.device,
                                                     is_bias=True)
            mixed_outputs = self.model(mixed_x)
            ## loss for labeled samples
            loss = -torch.mean(torch.sum(mixed_y[:lbs]* F.log_softmax(mixed_outputs[:lbs],dim=1), dim=1))
            loop_info['lSup'].append(loss.item())
            ## loss for unlabeled samples
            unlab_loss = F.mse_loss(torch.softmax(mixed_outputs[lbs:],dim=1), mixed_y[lbs:])
            unlab_loss *= self.rampup(self.epoch)*self.cons_weight
            loss += unlab_loss; loop_info['uMix'].append(unlab_loss.item())

            ##=== backwark ===
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===
            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['uacc'].append(unlab_y.eq(outputs1.max(1)[1]).float().sum().item())
            loop_info['u2acc'].append(unlab_y.eq(sharpen_pslab.max(1)[1]).float().sum().item())
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[train][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        self.update_bn(self.model, self.ema_model)
        print(f">>>[train]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    def test_iteration(self, data_loader):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            lbs, ubs = data.size(0), -1

            ##=== forward ===
            outputs     = self.model(data)
            ema_outputs = self.ema_model(data)

            ##=== log info ===
            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(targets.eq(outputs.max(1)[1]).float().sum().item())
            loop_info['l2acc'].append(targets.eq(ema_outputs.max(1)[1]).float().sum().item())
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[test][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(f">>>[test]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    def train(self, label_loader, unlab_loader):
        self.model.train()
        self.ema_model.train()
        with torch.enable_grad():
            return self.train_iteration(label_loader, unlab_loader)

    def test(self, data_loader):
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            return self.test_iteration(data_loader)

    def loop(self, epochs, label_data, unlab_data, test_data, scheduler=None):
        best_acc, n, best_info = 0., 0., None
        for ep in range(epochs):
            self.epoch = ep
            if scheduler is not None: scheduler.step()
            print("------ Training epochs: {} ------".format(ep))
            self.train(label_data, unlab_data)
            print("------ Testing epochs: {} ------".format(ep))
            info, n = self.test(test_data)
            acc     = sum(info['lacc'])/n
            if acc>best_acc: best_acc, best_info = acc, info

        print(f">>>[best]", self.gen_info(best_info, n, n, False))

    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step +1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)

    def update_bn(self, model, ema_model):
        for m2, m1 in zip(ema_model.named_modules(), model.named_modules()):
            if ('bn' in m2[0]) and ('bn' in m1[0]):
                bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
                bn2['running_mean'].data.copy_(bn1['running_mean'].data)
                bn2['running_var'].data.copy_(bn1['running_var'].data)
                bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)

    def decode_targets(self, targets):
        label_mask = targets.ge(0)
        unlab_mask = targets.le(-1)
        targets[unlab_mask] = decode_label(targets[unlab_mask])
        return label_mask, unlab_mask

    def gen_info(self, info, lbs, ubs, iteration=True):
        ret = []
        nums = {'l': lbs, 'u':ubs, 'a': lbs+ubs}
        for k, val in info.items():
            n = nums[k[0]]
            v = val[-1] if iteration else sum(val)
            s = f'{k}: {v/n:.3%}' if k[-1]=='c' else f'{k}: {v:.5f}'
            ret.append(s)
        return '\t'.join(ret)

data = read_cifar10(num_labels)
loaders = data_loaders(**data)
#模型
net = convLarge(num_classes, drop_ratio)
net = net.to(device)
net2 = convLarge(num_classes, drop_ratio)
net2 = net2.to(device)
#优化器
optimizer = optim.SGD(net.parameters(), lr,
                      momentum=momentum,
                      weight_decay=weight_decay,
                      nesterov=nesterov)
#学习率策略
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                           T_max=epochs,
                                           eta_min=min_lr)
trainer = Trainer(net, net2, optimizer)
trainer.loop(epochs, *loaders, scheduler=scheduler)