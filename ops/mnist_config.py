import argparse

parser = argparse.ArgumentParser(description='PtTorch Training')
parser.add_argument('--source_dataset',default='mnist')
parser.add_argument('--target_dataset',default='svhn')
parser.add_argument('--no_images',default=1000)
parser.add_argument('--train_iters',default=10001)
parser.add_argument('--batch_size',default=32)
parser.add_argument('--lr',default=0.0001)
parser.add_argument('--learning_rate_min',default=0.0001)
parser.add_argument('--learning_rate_max',default=1.)
parser.add_argument('--gamma',default=1.0)
parser.add_argument('--T_adv',default=15)
parser.add_argument('--T_min',default=100)
parser.add_argument('--K',default=6)
parser.add_argument('--no_classes',default=10)
parser.add_argument('--no_channels',default=3)
parser.add_argument('--img_size',default=32)
parser.add_argument('--gpu',default=0)