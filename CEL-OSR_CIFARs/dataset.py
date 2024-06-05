import os
import torch
import numpy as np
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms

from models.allconv import AllConvNet
from models.densenet import densenet
from models.resnext import resnext29
from models.wideresnet import WideResNet
from common.utils import fix_all_seed, write_log, sgd, entropy_loss


batch_size = 128

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]
class_num = 6
known_list = [
    [0,1, 2, 4, 5, 9],
    [0, 3, 5, 7, 8, 9],
    [0, 1, 5, 6, 7, 8],
    [3, 4, 5, 7, 8, 9],
    [0, 1, 2, 3, 7, 8]
]

known=known_list[0]  ## 0-19
unknown = list(set(list(range(0, 10))) - set(known))
root_folder = '../data'

def Filter(dataset,known):
    targets =np.array(dataset.targets)
    mask, new_targets = [],[]
    for i in range(len(targets)):
        if targets[i] in known:
            mask.append(i)
            new_targets.append((known.index(targets[i])))
    dataset.targets = np.array(new_targets).tolist()
    mask = torch.tensor(mask).long()
    data = torch.tensor(dataset.data)
    dataset.data = torch.index_select(data,0,mask)
    dataset.data = dataset.data.numpy()


def get_cifar10_close_set(batch_size,train=True):
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         preprocess])

    test_transform = preprocess
    if train == True:
        data = datasets.CIFAR10(root_folder, train=train, transform=train_transform, download=True)
        Filter(data, known)
        train_num = int(len(data) * 0.7)
        val_num = int(len(data) * 0.3)
        data, val_data = torch.utils.data.random_split(data,[train_num,val_num])
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        return loader, data, val_loader,  val_data
    else:
        data = datasets.CIFAR10(root_folder, train=train, transform=test_transform, download=True)
        Filter(data, known)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        return loader, data

def get_cifar10_open_set(batch_size,train=False):
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         preprocess])

    test_transform = preprocess
    if train == True:
        data = datasets.CIFAR10(root_folder, train=train, transform=train_transform, download=True)
        Filter(data,unknown)
        train_num = int(len(data) * 0.7)
        val_num = int(len(data) * 0.3)
        data, val_data = torch.utils.data.random_split(data,[train_num,val_num])
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        return loader, data, val_loader,  val_data
    else:
        data = datasets.CIFAR10(root_folder, train=train, transform=test_transform, download=True)
        Filter(data,unknown)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        return loader, data

preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5] * 3, [0.5] * 3)])

def get_cifar10_c_close_set(corruption,batch_size):
    test_data = datasets.CIFAR10(root_folder, train=False, transform=preprocess, download=False)
    base_c_path = os.path.join(root_folder, "CIFAR-10-C")
    test_data.data = np.load(os.path.join(base_c_path, corruption + '.npy'))
    test_data.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
    Filter(test_data,known)
    test_data.data = test_data.data[0:6000]
    test_data.targets = test_data.targets[0:6000]
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    return test_loader, test_data


def get_cifar10_c_open_set(corruption,batch_size):
    test_data2 = datasets.CIFAR10(root_folder, train=False, transform=preprocess, download=False)
    base_c_path = os.path.join(root_folder, "CIFAR-10-C")
    test_data2.data = np.load(os.path.join(base_c_path, corruption + '.npy'))
    test_data2.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
    Filter(test_data2,unknown)
    test_loader = torch.utils.data.DataLoader(
        test_data2,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    return test_loader, test_data2

