import argparse

parser = argparse.ArgumentParser(description='PtTorch Training')
parser.add_argument('--source_dataset',default='mnist')
parser.add_argument('--target_dataset',default='svhn')
parser.add_argument('--no_images',default=1000)
parser.add_argument('--train_iters',default=10001)
parser.add_argument('--epochs',default=100)
parser.add_argument('--batch_size',default=128)
parser.add_argument('--lr',default=0.1)
parser.add_argument('--learning_rate_min',default=0.1)
parser.add_argument('--lr_max',default=20.)
parser.add_argument('--momentum',default=0.9)
parser.add_argument('--weight_decay',default=0.0005)
parser.add_argument('--gamma',default=1.0)
parser.add_argument('--loops_adv',default=15)
parser.add_argument('--T_min',default=100)
parser.add_argument('--K',default=6)
parser.add_argument('--epochs_min',default=2)
parser.add_argument('--no_channels',default=3)
parser.add_argument('--img_size',default=32)
parser.add_argument('--gpu',default=0)
parser.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100'])
parser.add_argument('--model',default='wrn',choices=['wrn','allconv','densent','resnext'])
parser.add_argument('--k',default=2)
# WRN Architecture options
parser.add_argument('--layers', default=40, type=int,
                              help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int,
                              help='Widen factor')
parser.add_argument('--droprate', default=0.0, type=float,
                              help='Dropout probability')

