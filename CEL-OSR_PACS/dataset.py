import torch
from torchvision.datasets import ImageFolder
import os
import sys
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

known_list = [
    [1, 2, 6],
    [0, 3, 4],
    [2, 4, 5],
    [0, 2, 5],
    [0, 3, 6]
]

known=known_list[0]  ## 0-19
unknown = list(set(list(range(0, 7))) - set(known))


class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """

    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class Dataset_My(Dataset):
    def __init__(self, image, target, transform=None):
        self.image = image
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        img = self.image[index]
        target = self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.image)


class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot='../data/PACS', use_gpu=True, num_workers=0, batch_size=128,
                 img_size=227):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 7))) - set(known))

        print('Selected Labels: ', known)

        self.train_transform =transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4, .4, .4, .4),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'PACS', 'photo'), None)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        self.train_data = []
        self.trian_targets = []
        for i in range(len(trainset)):
            self.train_data.append(trainset[i][0])
            self.trian_targets.append(trainset[i][1])
        trainset = Dataset_My(self.train_data, self.trian_targets, transform=self.train_transform)
        trian_num = int(len(trainset) * 0.8)
        val_num = len(trainset) - trian_num
        trainset, valset = torch.utils.data.random_split(trainset, [trian_num, val_num])
        print('Train Data:', len(trainset), 'Val Data:', len(valset))
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.train_set = trainset
        
        self.val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_set = valset
        
        
        
        
        
        
        
        
        
        

        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'PACS', 'art_painting'), None)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        self.test_data = []
        self.test_targets = []
        for i in range(len(testset)):
            self.test_data.append(testset[i][0])
            self.test_targets.append(testset[i][1])
        testset = Dataset_My(self.test_data, self.test_targets, transform=transform)
        trian_num = int(len(trainset) * 0.8)
        val_num = len(trainset) - trian_num
        trainset, valset = torch.utils.data.random_split(trainset, [trian_num, val_num])
        print('Train Data:', len(trainset), 'Val Data:', len(valset))
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.test_set = testset

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'PACS', 'art_painting'), transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        #
        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

# Data = Tiny_ImageNet_OSR(known=known, dataroot='../data', batch_size=32, img_size=227)
# trainloader = Data.train_loader
# valloader = Data.test_loader
# openloader = Data.out_loader


# for i, (data,labels) in enumerate(trainloader):
#     for j in range(len(labels)):
#         if labels[j] == 0:
#             print(labels[j])
#             plt.imshow(data[j].numpy().transpose(1,2,0))
#             plt.show()
#     print(labels)
#
#     break
# #
# for i, (data,labels) in enumerate(valloader):
#     for j in range(len(labels)):
#         if labels[j] == 0:
#             print(labels[j])
#             plt.imshow(data[j].numpy().transpose(1,2,0))
#             plt.show()
#     print(labels)
#     break


# for i, (data,labels) in enumerate(openloader):
#     labels = np.ones_like(labels) * 20
#     for j in range(len(labels)):
#         if labels[j] == 18:
#             print(labels[j])
#             plt.imshow(data[j].numpy().transpose(1,2,0))
#             plt.show()
#     print(labels)
#     break

















