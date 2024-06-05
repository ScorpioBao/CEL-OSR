"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python implementation of Biased-MNIST.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils
from PIL import Image

import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import MNIST



class BiasedMNIST(MNIST):
    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 128, 0], [255, 0, 128], [0, 255, 255], [255, 0, 0]]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.random = True

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class ColourBiasedMNIST(BiasedMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9):
        super(ColourBiasedMNIST, self).__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels)

    def _binary_to_colour(self, data, colour):
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label):
        return self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]


def get_biased_mnist_dataloader(root, batch_size, data_label_correlation,
                                n_confusing_labels=9, train=True, num_workers=0):
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5),
        #                      std=(0.5, 0.5, 0.5))
                              ])

    if train:
        dataset = ColourBiasedMNIST(root, train=train, transform=transform,
                                    download=True, data_label_correlation=data_label_correlation,
                                    n_confusing_labels=n_confusing_labels)
        train_num = int(len(dataset) * 0.7)
        val_num = int(len(dataset) * 0.3)
        tr_data, val_data = torch.utils.data.random_split(dataset,[train_num,val_num])
        tr_loader = data.DataLoader(dataset=tr_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
        val_loader = data.DataLoader(dataset=val_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)

        return tr_loader,tr_data,val_loader,val_data
    else:
        dataset = ColourBiasedMNIST(root, train=train, transform=transform,
                                    download=True, data_label_correlation=data_label_correlation,
                                    n_confusing_labels=n_confusing_labels)
        dataloader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True)
        return dataloader,dataset





tr_loader,tr_data,val_loader,val_data = get_biased_mnist_dataloader(root="../data/color_mnist/biased_mnist", batch_size=1,
                                            data_label_correlation=0.990,
                                            n_confusing_labels=9,
                                            train=True)

#
# test_loaders, test_data = get_biased_mnist_dataloader(root="../data/color_mnist/biased_mnist", batch_size=1,
#                                                     data_label_correlation=0.1,
#                                                     n_confusing_labels=9,
#                                                     train=False)


## 生成训练集和验证集
# class Denormalise(transforms.Normalize):
#     """
#     Undoes the normalization and returns the reconstructed images in the input domain.
#     """
#
#     def __init__(self, mean, std):
#         mean = torch.as_tensor(mean)
#         std = torch.as_tensor(std)
#         std_inv = 1 / (std + 1e-12)
#         mean_inv = -mean * std_inv
#         super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)
#
#     def __call__(self, tensor):
#         return super(Denormalise, self).__call__(tensor.clone())
#
# image_denormalise = Denormalise([0.5] * 3, [0.5] * 3)
# image_transform = transforms.ToPILImage()
#
#
# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=(0.5, 0.5, 0.5),
#     #                      std=(0.5, 0.5, 0.5))
#                       ])
# tr_data.transform = train_transform
# images = []
# labels = []
# for i ,(x,y) in enumerate(tr_loader):
#     if y != 8 and y != 9:
#         images.append(x.detach().cpu())
#         labels.append(y)
# images = np.stack(images,axis=0)
# labels = np.stack(labels,axis=0)
# images = images.reshape([len(labels),3,28,28])
# labels = labels.reshape([len(labels),1])
# images = torch.tensor(images)
# labels = torch.tensor(labels)
# all_images=[]
# for j in range(len(images)):
#     image = image_denormalise(images[j])
#     image = image_transform(images[j].clamp(min=0.0,max=1.0))
#     all_images.append(image)
#
# images = np.stack(all_images)
# images = torch.tensor(images)
# torch.save(images, '../data/color_mnist/color_mnist_images_0.990.pth')
# torch.save(labels, '../data/color_mnist/color_mnist_labels_0.990.pth')
#
#
# images_val = []
# labels_val = []
# for i ,(x,y) in enumerate(val_loader):
#     if y != 8 and y != 9:
#         images_val.append(x.detach().cpu())
#         labels_val.append(y)
# images_val = np.stack(images_val,axis=0)
# labels_val = np.stack(labels_val,axis=0)
# images_val = images_val.reshape([len(labels_val),3,28,28])
# labels_val = labels_val.reshape([len(labels_val),1])
# images_val = torch.tensor(images_val)
# labels_val = torch.tensor(labels_val)
# all_images_val=[]
# for j in range(len(images_val)):
#     image = image_denormalise(images_val[j])
#     image = image_transform(images_val[j].clamp(min=0.0,max=1.0))
#     all_images_val.append(image)
#
# images_val = np.stack(all_images_val)
# images_val = torch.tensor(images_val)
# torch.save(images, '../data/color_mnist/color_mnist_images_0.990_val.pth')
# torch.save(labels, '../data/color_mnist/color_mnist_labels_0.990_val.pth')
#
# images = torch.load('../data/color_mnist/color_mnist_images_0.990.pth')
# labels = torch.load('../data/color_mnist/color_mnist_labels_0.990.pth')
# tr_data.data = np.array(images)
# tr_data.targets = np.array(labels)
#
#




