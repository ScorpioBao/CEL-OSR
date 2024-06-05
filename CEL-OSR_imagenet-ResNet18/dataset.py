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
    [2, 3, 13, 30, 44, 45, 64, 66, 76, 101, 111, 121, 128, 130, 136, 158, 167, 170, 187, 193],
    [4, 11, 32, 42, 51, 53, 67, 84, 87, 104, 116, 140, 144, 145, 148, 149, 155, 168, 185, 193],
    [3, 9, 10, 20, 23, 28, 29, 45, 54, 74, 133, 143, 146, 147, 156, 159, 161, 170, 184, 195],
    [1, 15, 17, 31, 36, 44, 66, 69, 84, 89, 102, 137, 154, 160, 170, 177, 182, 185, 195, 197],
    [4, 14, 16, 33, 34, 39, 59, 69, 77, 92, 101, 103, 130, 133, 147, 161, 166, 168, 172, 173]
]



known=known_list[0]  ## 0-19
unknown = list(set(list(range(0, 200))) - set(known))


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
    def __init__(self, known, dataroot='../data/tiny_imagenet-200', use_gpu=True, num_workers=0, batch_size=128,
                 img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))
        self.train_set = None

        print('Selected Labels: ', known)

        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), transform=None)
        # print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        self.train_data = []
        self.trian_targets = []
        for i in range(len(trainset)):
            self.train_data.append(trainset[i][0])
            self.trian_targets.append(trainset[i][1])
        # # self.train_data = np.array(self.train_data)
        # # self.trian_targets = np.array(self.trian_targets)
        # self.train_data = self.train_data
        # self.trian_targets = self.trian_targets
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
        
        val_outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), transform=transform)
        # print('All Train Data:', len(trainset))
        val_outset.__Filter__(known=self.unknown)
        val_outnum = int(len(val_outset) * 0.1)
        _ = len(val_outset) - val_outnum
        val_outset, _ = torch.utils.data.random_split(val_outset, [val_outnum, _])
        print('Val OutData:', len(val_outset))
            
    
        
        self.val_outloader = torch.utils.data.DataLoader(
            val_outset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.val_outset = val_outset
        
        
        


        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        self.test_data = []
        self.test_targets = []
        for i in range(len(testset)):
            self.test_data.append(testset[i][0])
            self.test_targets.append(testset[i][1])
        testset = Dataset_My(self.test_data, self.test_targets, transform=None)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.test_set = testset

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Test: ', len(testset), 'Out: ', len(outset))
        # print('All Test: ', (len(testset) + len(outset)))



# Data = Tiny_ImageNet_OSR(known=known, dataroot='../data', batch_size=2, img_size=64)
#
#
# trainloader = Data.train_loader
# valloader = Data.test_loader
# openloader = Data.out_loader



# for i, (data,labels) in enumerate(trainloader):
#     for j in range(len(labels)):
#         if labels[j] == 19:
#             print(labels[j])
#             plt.imshow(data[j].numpy().transpose(1,2,0))
#             plt.show()
#     print(labels)
#
#     break
#
# for i, (data,labels) in enumerate(valloader):
#     for j in range(len(labels)):
#         if labels[j] == 19:
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

















