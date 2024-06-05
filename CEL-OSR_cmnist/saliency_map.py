
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
from ops.mnist_config import parser

from models.bias_mnist_model import Model, Model_bias
args = parser.parse_args()
import models
from color_mnist import get_biased_mnist_dataloader
from PIL import Image
from criterions import edl
import torchvision.transforms as transforms

print('Use GPU:{} for testing'.format(args.gpu))
model = torch.load('./checkpoint/best_model_0.990.pt')
# model['f_net']
f_config = {'kernel_size': 7, 'feature_pos': 'post'}
f_net = Model(f_config)
weight_dict={}
for k,v in model.items():
    new_k = k.replace('module.','') if 'module' in k else k
    weight_dict[new_k]= v
f_net.load_state_dict(weight_dict)
f_net.eval()



val_loaders, val_data = get_biased_mnist_dataloader(root="../data/color_mnist/biased_mnist", batch_size=1,
                                                    data_label_correlation=0.1,
                                                    n_confusing_labels=9,
                                                    train=False)
train_transform = transforms.Compose([
    transforms.ToTensor(),
                      ])
val_data.transform = train_transform


images_test = torch.load('../data/color_mnist/color_mnist_images_test.pth')
labels_test = torch.load('../data/color_mnist/color_mnist_labels_test.pth')
labels_test = labels_test.squeeze()
val_data.data = np.array(images_test)
val_data.targets = np.array(labels_test)
val_data.transform = train_transform
tr_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# for i,(x,y) in enumerate(tr_loader):
#     print(x.shape)
#     visual(x,y,count=0)


def saliency(input, model):
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()
    # transoform input PIL image to torch.Tensor and normalize
    # input = transform(img)
    # input.unsqueeze_(0)

    # we want to calculate gradient of higest score w.r.t. input
    # so set requires_grad to True for input
    input.requires_grad = True
    # forward pass to calculate predictions
    preds, _ = model(input)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    # apply inverse transform on image
    with torch.no_grad():
        input_img = inv_normalize(input[0])
    # plot image and its saleincy map
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(input[0].detach().numpy(), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.show()
for i in range(100):
    target = 1
    images = []
    labels = []

    for i ,(x,y) in enumerate(tr_loader):
        if y == target:
            output, cfeatures = f_net(x)
            if torch.argmax(output) == target:
                evidence = edl.softplus_evidence(output)
                alpha = evidence + 1
                uncertain = 8 / torch.sum(alpha, 1, keepdim=True)
                print("uncertain:", uncertain.detach().numpy())
                # visual(x, y, count=0)
                images.append(x.detach().cpu())
                labels.append(y)
                break
            else:
                continue
    images = np.stack(images,axis=0)
    labels = np.stack(labels,axis=0)

    images = images.reshape([len(labels),3,28,28])
    labels = labels.reshape([len(labels),1])
    # showimages = images.squeeze().transpose(1,2,0)
    # plt.imshow(showimages)
    # plt.show()


    images = torch.tensor(images)
    labels = torch.tensor(labels)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    #inverse transform to get normalize image back to original form for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        normalize,
    ])



    saliency(images,f_net)