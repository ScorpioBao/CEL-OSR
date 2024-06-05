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

def get_mnist_dataloader(root, batch_size, train=True, num_workers=0):

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010))
        ])
    dataset = MNIST(root, train=train, transform=transform,
                                download=True)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader

class Deal_Source_Dataset(torch.utils.data.Dataset):
    def __init__(self,transform=None,train=True,images=None,labels=None):
        self.x_data = images
        self.y_data = labels
        self.len = images.shape[0]
        self.transform = transform
    def __getitem__(self,index):
        if self.transform is not None:
            # x = self.transform(self.x_data[index])
            # y = self.trandform(self.y_data[index])
            return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

def get_all_source_dataloader(batch_size, x,y):
    transform = transforms.Compose([
        transforms.Resize((32, 32))])
    dataset = Deal_Source_Dataset(train=True, transform=transform,images=x,labels=y)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=False)
    return dataloader


# all_source_data = []
# all_source_labels = []
# args = parser.parse_args()
# train_dataloader = get_mnist_dataloader('./data',batch_size=args.batch_size,train=True)
# for i, (x,y) in enumerate(train_dataloader):
#     all_source_data.append(x)
#     all_source_labels.append(y)
#
# print(all_source_data)

def load_svhn(split='train'):

    print('Loading SVHN dataset.')
    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join('', 'data/svhn', image_file)
    svhn = io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2])
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels

def load_mnist_m():
    with gzip.open('./data/MNIST-M/keras_mnistm.pkl.gz','rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
    return data

class DealDataset2(torch.utils.data.Dataset):
    def __init__(self,transform=None,train=True,images=None,labels=None):
        self.x_data = images
        self.y_data = labels
        self.len = images.shape[0]
        self.transform = transform
    def __getitem__(self,index):
        image = self.x_data[index]
        if self.transform is not None:
            image = self.transform(image)
            return image,self.y_data[index]
    def __len__(self):
        return self.len



def get_svhn_dataloader(batch_size, train=True, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010))
    ])
    if train:
        images, labels = load_svhn()
    else:
        images, labels = load_svhn(split='test')
    dataset = DealDataset2(train=train, transform=transform,images=images,labels=labels)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader





def get_mnist_m_dataloader(root,batch_size, train_True, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        ])
    dataset = mnistm.MNISTM(root, train=train_True, transform=transform, target_transform=None,
                            download=True)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader

# data = get_mnist_m_dataloader('./data/MNIST-M',batch_size=32,train_True=False)
# for i,(x,y) in enumerate(data):
#     if i==0:
#         print(x)

_image_size = 32
_trans = transforms.Compose([
    transforms.Resize(_image_size),
    transforms.ToTensor()
])
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
    root = os.path.join(root_dir, 'data/USPS')
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

class DealDataset(torch.utils.data.Dataset):
    def __init__(self,train=True,images=None,labels=None):
        self.x_data = images
        self.y_data = labels
        self.len = images.shape[0]
    def __getitem__(self,index):

        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len


def get_usps_dataloader(batch_size, train=True, num_workers=0):
    if train:
        images, labels = load_usps('', train=True)
    else:
        images, labels = load_usps('', train=False)
    dataset = DealDataset(train=train,images=images,labels=labels)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader

# data = get_usps_dataloader(6,False)
# for i,(x,y) in enumerate(data):
#     print(x.shape)

def load_syn(root_dir, train=True):
    split_list = {
        'train': "synth_train_32x32.mat",
        'test': "synth_test_32x32.mat"
    }

    split = 'train' if train else 'test'
    filename = split_list[split]
    full_path = os.path.join(root_dir, 'data/SYN', filename)

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

def get_syn_dataloader(batch_size, train=True, num_workers=0):
    if train:
        images, labels = load_syn('', train=True)
    else:
        images, labels = load_syn('', train=False)
    dataset = DealDataset(train=train,images=images,labels=labels)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader

