import os
from  scipy import io
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST
import bz2
from data import mnistm
import torch
from PIL import Image
import gzip
import pickle
from torchvision.datasets.utils import download_url
import scipy

_image_size = 32


known_list = [
    [0,1, 2, 4, 5, 9],
    [0, 3, 5, 7, 8, 9],
    [0, 1, 5, 6, 7, 8],
    [3, 4, 5, 7, 8, 9],
    [0, 1, 2, 3, 7, 8]
]

known=known_list[0]  ## 0-19
unknown = list(set(list(range(0, 10))) - set(known))
_trans = transforms.Compose([
    transforms.Resize(_image_size),
    transforms.ToTensor()
])
def load_svhn(split='train'):

    # print('Loading SVHN dataset.')
    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join('../data', 'svhn', image_file)
    svhn = io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2])
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels
def load_syn(root_dir, train=True):
    split_list = {
        'train': "synth_train_32x32.mat",
        'test': "synth_test_32x32.mat"
    }

    split = 'train' if train else 'test'
    filename = split_list[split]
    full_path = os.path.join(root_dir, 'SYN', filename)

    raw_data = scipy.io.loadmat(full_path)
    imgs = np.transpose(raw_data['X'], [3, 0, 1, 2])
    images = []
    for img in imgs:
        img = Image.fromarray(img, mode='RGB')
        img = _trans(img)
        images.append(img.numpy())
    targets = raw_data['y'].reshape(-1)
    targets[np.where(targets == 10)] = 0

    return np.stack(images), targets.astype(np.int64)
def load_usps(root_dir, train=True):
    split_list = {
        'train': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2", 'ec16c51db3855ca6c91edd34d0e9b197'
        ],
        'test': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2", '8ea070ee2aca1ac39742fdd1ef5ed118'
        ],
    }

    split = 'train' if train else 'test'
    url, filename, checksum = split_list[split]
    root = os.path.join(root_dir, 'USPS')
    full_path = os.path.join(root, filename)

    if not os.path.exists(full_path):
        download_url(url, root, filename, md5=checksum)

    with bz2.BZ2File(full_path) as fp:
        raw_data = [l.decode().split() for l in fp.readlines()]
        imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
        imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
        imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
        targets = [int(d[0]) - 1 for d in raw_data]

    images, labels = [], []
    for img, target in zip(imgs, targets):
        img = Image.fromarray(img, mode='L')
        img = _trans(img)
        images.append(img.expand(3, -1, -1).numpy())
        labels.append(target)
    return np.stack(images), np.array(labels)
def load_mnist_m():
    with gzip.open('./data/MNIST-M/keras_mnistm.pkl.gz','rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
    return data

class DealDataset(torch.utils.data.Dataset):
    def __init__(self,train=True,images=None,labels=None):
        self.x_data = images
        self.targets = labels
        self.len = images.shape[0]
    def __getitem__(self,index):

        return self.x_data[index],self.targets[index]
    def __len__(self):
        return self.targets.shape[0]
class DealDataset2(torch.utils.data.Dataset):
    def __init__(self,transform=None,train=True,images=None,labels=None):
        self.x_data = images
        self.targets = labels
        self.len = images.shape[0]
        self.transform = transform
    def __getitem__(self,index):
        image = self.x_data[index]
        if self.transform is not None:
            image = self.transform(image)
            return image,self.targets[index]
    def __len__(self):
        return self.targets.shape[0]

def Filter(dataset,known):
    targets = dataset.targets.data.numpy()
    mask, new_targets = [],[]
    for i in range(len(targets)):
        if targets[i] in known:
            mask.append(i)
            new_targets.append((known.index(targets[i])))
    dataset.targets = np.array(new_targets)
    mask = torch.tensor(mask).long()
    dataset.data = torch.index_select(dataset.data,0,mask)
def Filter_svhn(dataset,known):
    targets = dataset.targets.data
    mask, new_targets = [],[]
    for i in range(len(targets)):
        if targets[i] in known:
            mask.append(i)
            new_targets.append((known.index(targets[i])))
    dataset.targets = np.array(new_targets)
    mask = torch.tensor(mask).long()
    data = torch.tensor(dataset.x_data)
    dataset.x_data = torch.index_select(data,0,mask)
    # print(dataset.x_data)

def get_mnist_dataloader_close_set(root, batch_size, train=True, num_workers=0):

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010))
        ])
    if train:
        dataset = MNIST(root, train=train, transform=transform,
                                    download=True)
        # print(len(dataset))
        Filter(dataset,known)
        # print(len(dataset))
        train_num = int(len(dataset) * 0.7)
        val_num = int(len(dataset) -train_num)
        train_data, val_data = torch.utils.data.random_split(dataset, [train_num, val_num])

        dataloader = data.DataLoader(dataset=train_data,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        return dataloader, train_data, val_loader, val_data
    else:
        dataset = MNIST(root, train=train, transform=transform,
                        download=True)
        Filter(dataset, known)
        dataloader = data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
    return dataloader,dataset

def get_mnist_dataloader_open_set(root, batch_size, train=False, num_workers=0):

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010))
        ])
    if train:
        dataset = MNIST(root, train=train, transform=transform,
                                    download=True)
        Filter(dataset,unknown)
        train_num = int(len(dataset) * 0.7)
        val_num = int(len(dataset) -train_num)
        train_data, val_data = torch.utils.data.random_split(dataset, [train_num, val_num])

        dataloader = data.DataLoader(dataset=train_data,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        return dataloader, train_data, val_loader, val_data
    else:
        dataset = MNIST(root, train=train, transform=transform,
                        download=True)
        Filter(dataset, unknown)
        dataloader = data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
    return dataloader,dataset


def get_svhn_dataloader_close_set(batch_size, train=True, num_workers=0):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010))
    ])
    if train:
        images, labels = load_svhn()
    else:
        images, labels = load_svhn(split='test')
    dataset = DealDataset2(train=train, transform=transform, images=images, labels=labels)
    Filter_svhn(dataset,known)
    # print(len(dataset.x_data))
    dataset.x_data = np.transpose(dataset.x_data,(0,3,1,2))
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader, dataset
def get_svhn_dataloader_open_set(batch_size, train=False, num_workers=0):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010))
    ])
    if train:
        images, labels = load_svhn()
    else:
        images, labels = load_svhn(split='test')
    dataset = DealDataset2(train=train, transform=transform, images=images, labels=labels)
    Filter_svhn(dataset,unknown)
    dataset.x_data = np.transpose(dataset.x_data,(0,3,1,2))
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader, dataset



def get_mnist_m_dataloader_close_set(root,batch_size, train_True, num_workers=0):

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        ])
    dataset = mnistm.MNISTM(root, train=train_True, transform=transform, target_transform=None,
                            download=True)
    Filter(dataset,known)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader, dataset
def get_mnist_m_dataloader_open_set(root,batch_size, train_True, num_workers=0):

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        ])
    dataset = mnistm.MNISTM(root, train=train_True, transform=transform, target_transform=None,
                            download=True)
    Filter(dataset,unknown)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader, dataset


def get_syn_dataloader_close_set(batch_size, train=True, num_workers=0):
    if train:
        images, labels = load_syn('../data', train=True)
    else:
        images, labels = load_syn('../data', train=False)
    dataset = DealDataset(train=train,images=images,labels=labels)
    Filter_svhn(dataset,known)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader,dataset
def get_syn_dataloader_open_set(batch_size, train=True, num_workers=0):
    if train:
        images, labels = load_syn('../data', train=True)
    else:
        images, labels = load_syn('../data', train=False)
    dataset = DealDataset(train=train,images=images,labels=labels)
    Filter_svhn(dataset,unknown)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader,dataset


def get_usps_dataloader_close_set(batch_size, train=True, num_workers=0):
    if train:
        images, labels = load_usps('../data', train=True)
    else:
        images, labels = load_usps('../data', train=False)
    dataset = DealDataset(train=train,images=images,labels=labels)
    Filter_svhn(dataset,known)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader,dataset
def get_usps_dataloader_open_set(batch_size, train=True, num_workers=0):
    if train:
        images, labels = load_usps('../data', train=True)
    else:
        images, labels = load_usps('../data', train=False)
    dataset = DealDataset(train=train,images=images,labels=labels)
    Filter_svhn(dataset,unknown)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader,dataset

