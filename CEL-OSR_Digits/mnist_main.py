import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torchvision import transforms
from models.mnist_model import Model, Model_bias
from ops.mnist_config import parser
from mnist_train import train
from torch.autograd import Variable
from criterions import get_criterion
import mnist_split
import numpy as np

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main():
    args = parser.parse_args()
    print('Use GPU:{} for training'.format(args.gpu))
    f_model = Model()
    g_model = Model_bias()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        f_model = torch.nn.DataParallel(f_model).cuda(args.gpu)
        g_model = torch.nn.DataParallel(g_model).cuda(args.gpu)
    else:
        f_model = torch.nn.DataParallel(f_model).cuda()
        g_model = torch.nn.DataParallel(g_model).cuda()

    criterion_train_1 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_train_2 = nn.MSELoss().cuda(args.gpu)
    outer_criterion_config = {'sigma_x': 1, 'sigma_y': 1,
                              'algorithm': 'unbiased'}  # {1,1,'unbiased'}无偏的HSIC独立性检测
    outer_criterion_detail = {'sigma_x_type': 1,
                              'sigma_y_type': 1,
                              'sigma_x_scale': 1,
                              'sigma_y_scale': 1}  # {1,1,1,1}
    inner_criterion_config = {'sigma_x': 1, 'sigma_y': 1,
                              'algorithm': 'unbiased'}  # {1,1,'unbiased'}无偏的HSIC独立性检测
    inner_criterion_detail = {'sigma_x_type': 1,
                              'sigma_y_type': 1,
                              'sigma_x_scale': 1,
                              'sigma_y_scale': 1}  # {1,1,1,1}
    sigma_x = inner_criterion_config['sigma_x']
    sigma_y = inner_criterion_config['sigma_y']

    # criterion_train_1 = edl

    cudnn.benchmark = True
    print('Loading MNIST dataset')
    source_train_loader, train_dataset, val_loader, val_dataset = mnist_split.get_mnist_dataloader_close_set(root='../data', batch_size=args.batch_size, train=True)
    print('traing_data_close:',len(train_dataset))
    print('val_data_close:', len(val_dataset))
    source_train_loader_open, train_dataset_open, val_loader_open, val_dataset_open = mnist_split.get_mnist_dataloader_open_set(root='../data', batch_size=args.batch_size, train=True)
    print('traing_data_open:',len(train_dataset_open))
    print('val_data_open:', len(val_dataset_open))

    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset.transform = data_transform

    sigma_x_scale = 1.
    sigma_y_scale = 1.
    outer_criterion_config['sigma_x'] = sigma_x * sigma_x_scale
    outer_criterion_config['sigma_y'] = sigma_y * sigma_y_scale
    inner_criterion_config['sigma_x'] = sigma_x * sigma_x_scale
    inner_criterion_config['sigma_y'] = sigma_y * sigma_y_scale

    outer_criterion = get_criterion('RbfHSIC')(**outer_criterion_config)
    inner_criterion = get_criterion('MinusRbfHSIC')(**inner_criterion_config)


    #保存划分好的原始训练集以及验证集
    all_source_data = []
    all_source_labels = []
    for i,(x,y) in enumerate(source_train_loader):
        all_source_data.append(x)
        all_source_labels.append(y)
    all_source_data = np.concatenate(all_source_data)
    all_source_labels = np.concatenate(all_source_labels)

    torch.save(all_source_data,'../data/split_mnist_source_data.pth')
    torch.save(all_source_labels,'../data/split_mnist_source_labels.pth')

    z_hat = Variable(torch.empty(size=(32, 3, 32, 32), dtype=torch.float32), requires_grad=True)
    max_optimizer = torch.optim.SGD([z_hat, ], args.learning_rate_max)

    f_min_optimizer = torch.optim.Adam(f_model.parameters(), args.learning_rate_min)
    g_min_optimizer = torch.optim.Adam(g_model.parameters(), args.learning_rate_min)

    print('Training.....')
    train(z_hat, val_loader,val_loader_open, f_model, g_model, criterion_train_2, f_min_optimizer, g_min_optimizer,
          max_optimizer, args, outer_criterion, inner_criterion)


if __name__ == '__main__':
    main()
