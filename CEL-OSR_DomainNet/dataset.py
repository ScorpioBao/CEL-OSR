import torch
from torchvision.datasets import ImageFolder
import os
import sys
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

def fix_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

fix_all_seed(seed=0)
## 从345类中随机选择100个作为known class，其余的作为unknown class
known_list = [
    [6, 7, 12, 15, 17, 20, 21, 22, 26, 37, 49, 52, 54, 55, 56, 59, 60, 63, 64, 65, 66, 68, 74, 76, 78, 81, 89, 90, 92, 100, 101, 102, 103, 106, 113, 122, 124, 126, 132, 133, 135, 136, 137, 141, 142, 144, 146, 156, 164, 166, 168, 170, 173, 189, 190, 196, 198, 204, 210, 212, 214, 215, 217, 218, 219, 230, 231, 234, 235, 236, 237, 240, 246, 248, 250, 252, 254, 260, 263, 266, 268, 272, 276, 278, 279, 282, 283, 286, 287, 291, 293, 299, 310, 311, 315, 326, 334, 336, 339, 344],
    [0, 4, 6, 9, 11, 12, 14, 16, 18, 27, 29, 41, 51, 58, 59, 62, 65, 67, 70, 73, 80, 85, 89, 90, 91, 92, 93, 95, 102, 105, 106, 107, 111, 112, 117, 119, 120, 122, 123, 125, 127, 131, 132, 138, 146, 150, 154, 159, 162, 165, 169, 171, 174, 177, 179, 187, 188, 189, 191, 197, 204, 206, 208, 211, 212, 217, 222, 229, 234, 236, 242, 244, 245, 247, 249, 251, 256, 261, 267, 268, 274, 284, 288, 292, 293, 294, 295, 302, 306, 307, 314, 316, 324, 329, 330, 332, 333, 334, 342, 344],
    [2, 3, 7, 10, 11, 12, 13, 17, 20, 24, 25, 29, 30, 35, 41, 53, 55, 60, 65, 66, 67, 68, 69, 70, 77, 84, 89, 94, 98, 99, 100, 106, 109, 112, 114, 118, 141, 142, 146, 147, 150, 151, 153, 154, 156, 157, 160, 163, 166, 169, 170, 172, 174, 181, 182, 183, 190, 194, 200, 203, 204, 205, 209, 210, 212, 216, 217, 220, 223, 225, 227, 230, 232, 237, 240, 244, 251, 253, 256, 257, 262, 269, 275, 278, 281, 285, 296, 300, 301, 303, 309, 312, 313, 325, 328, 331, 333, 334, 340, 343],
    [5, 10, 14, 15, 16, 24, 25, 30, 31, 37, 38, 46, 47, 50, 55, 56, 59, 61, 65, 66, 67, 68, 73, 74, 75, 79, 80, 82, 83, 84, 98, 101, 102, 105, 112, 114, 115, 121, 123, 127, 128, 134, 140, 142, 146, 153, 155, 157, 162, 163, 166, 177, 180, 186, 187, 191, 193, 194, 195, 196, 203, 204, 205, 207, 211, 213, 217, 218, 221, 222, 224, 229, 233, 236, 238, 239, 240, 241, 247, 252, 261, 264, 266, 273, 275, 278, 280, 281, 290, 297, 300, 311, 312, 327, 328, 329, 333, 339, 342, 343],
    [1, 6, 11, 12, 13, 14, 18, 24, 33, 34, 43, 45, 46, 47, 55, 61, 63, 64, 66, 68, 70, 72, 82, 83, 84, 88, 89, 91, 92, 93, 99, 100, 101, 103, 106, 112, 124, 128, 129, 130, 134, 142, 146, 153, 154, 157, 166, 169, 173, 177, 178, 185, 193, 203, 204, 206, 214, 218, 220, 221, 226, 228, 239, 245, 246, 247, 248, 251, 257, 258, 259, 261, 262, 265, 267, 272, 274, 283, 284, 286, 287, 289, 291, 292, 293, 301, 303, 304, 307, 315, 319, 322, 324, 333, 336, 337, 339, 341, 342, 344]
]

known=known_list[0]  ## 0-19
# known=list([2, 3, 13, 30, 44, 45, 64, 66, 76, 101, 111, 121, 128, 130, 136, 158, 167, 170, 187, 193]) ## 0-19
unknown = list(set(list(range(0, 345))) - set(known))
print(known)

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

class TrainDatasetTransformed(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class Tiny_ImageNet_OSR(object):
    def __init__(self, dataroot='../data/DomainNet', domain='real', use_gpu=True, num_workers=0, batch_size=128,
                 img_size=227):
        self.num_classes = 100
        self.known = known
        self.unknown = unknown

        self.train_transform =transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4, .4, .4, .4),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'DomainNet', domain), transform=None)
        trainset.__Filter__(known=self.known)
        
        train_size = int(0.7 * len(trainset))
        test_size = len(trainset) - train_size
        self.train_set, self.test_set = torch.utils.data.random_split(trainset, [train_size, test_size])

        self.train_set = TrainDatasetTransformed(self.train_set,transform=self.train_transform)
        self.test_set = TrainDatasetTransformed(self.test_set,transform = self.test_transform)
        # 训练取消注释测试打开注释
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size,  sampler=self.train_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )
        # sampler=self.test_sampler,
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, sampler=self.test_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )
        # 训练打开注释测试取消注释
        # self.train_loader = torch.utils.data.DataLoader(
        #     self.train_set, batch_size=batch_size, 
        #     num_workers=num_workers, pin_memory=pin_memory
        # )
        # # sampler=self.test_sampler,
        # self.test_loader = torch.utils.data.DataLoader(
        #     self.test_set, batch_size=batch_size, 
        #     num_workers=num_workers, pin_memory=pin_memory
        # )



        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'DomainNet', domain), transform=None)
        outset.__Filter__(known=self.unknown)
        out_size_train = int(0.7 * len(outset))
        out_size_test = len(outset) - out_size_train
        self.train_outset, self.test_outset = torch.utils.data.random_split(outset, [out_size_train, out_size_test])
        self.train_outset = TrainDatasetTransformed(self.train_outset, transform=self.train_transform)
        self.test_outset = TrainDatasetTransformed(self.test_outset, transform=self.test_transform)
        # self.out_sampler = torch.utils.data.distributed.DistributedSampler(self.test_outset)
        
        self.out_loader = torch.utils.data.DataLoader(
            self.test_outset, batch_size=batch_size, 
            num_workers=num_workers, pin_memory=pin_memory
        )

        print('Train: ', len(self.train_set) )
        print('Val: ', (len(self.test_set)))
        print('Out:',len(self.test_outset))


# Data = Tiny_ImageNet_OSR(dataroot='../data', domain='real', batch_size=1024, img_size=224)
# trainloader = Data.train_loader
# valloader = Data.test_loader
# Data = Tiny_ImageNet_OSR(dataroot='../data', batch_size=32, img_size=227)
# trainloader = Data.train_loader
# valloader = Data.test_loader
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for data in valloader:
#     images, labels = data
#     images, labels = images.to(device), labels.to(device)
#     print(labels)


















