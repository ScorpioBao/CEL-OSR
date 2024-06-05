from __future__ import print_function, absolute_import
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# rootpath = str("/home/bqs/CEL-OSR")
# syspath = sys.path
# sys.path = []
# sys.path.append(rootpath)
# sys.path.extend([rootpath + i for i in os.listdir(rootpath) if i[0] != "."])
# sys.path.extend(syspath)
# import sys
# import os
#os.chdir(sys.path[0])  #使用文件所在目录
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse

from train_bias import ModelBaseline, ModelCELOSR
from models.rebias_models import  ReBiasModels


def main():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10'],
                                  help='Choose between CIFAR-10')
    train_arg_parser.add_argument('--algorithm', type=str, default='CEL-OSR', choices=['ERM', 'ADA', 'CEL-OSR'],
                                  help='Choose algorithm.')
    train_arg_parser.add_argument('--model', type=str, default='resnext', choices=['wrn', 'allconv', 'densenet', 'resnext'],
                                  help='Choose architecture.')
    train_arg_parser.add_argument("--epochs", type=int, default=101,
                                  help='Number of epochs to train.')
    train_arg_parser.add_argument("--batch_size", type=int, default=256,
                                  help="")
    train_arg_parser.add_argument("--num_workers", type=int, default=0,
                                  help='Number of pre-fetching threads.')
    train_arg_parser.add_argument("--lr", type=float, default=0.01,
                                  help='')
    train_arg_parser.add_argument("--lr_max", type=float, default=20.0,
                                  help='')
    train_arg_parser.add_argument('--momentum', type=float, default=0.9,
                                  help='Momentum.')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.0005,
                                  help='')
    train_arg_parser.add_argument("--logs", type=str, default='cifar10/logs0/',
                                  help='')
    train_arg_parser.add_argument("--model_path", type=str, default='cifar10/models/',
                                  help='')
    train_arg_parser.add_argument("--deterministic", type=bool, default=True,
                                  help='')
    train_arg_parser.add_argument("--epochs_min", type=int, default=2,
                                  help="")
    train_arg_parser.add_argument("--loops_adv", type=int, default=15,
                                  help="")
    train_arg_parser.add_argument("--k", type=int, default=2,
                                  help="")
    train_arg_parser.add_argument("--gamma", type=float, default=1.0,
                                  help="")
    train_arg_parser.add_argument("--eta", type=float, default=0,
                                  help="")

    # WRN Architecture options
    train_arg_parser.add_argument('--layers', default=40, type=int,
                                  help='total number of layers')
    train_arg_parser.add_argument('--widen-factor', default=2, type=int,
                                  help='Widen factor')
    train_arg_parser.add_argument('--droprate', default=0.0, type=float,
                                  help='Dropout probability')

    args = train_arg_parser.parse_args()

    if args.algorithm == 'ERM':
        model_obj = ModelBaseline(flags=args)
    elif args.algorithm == 'CEL-OSR':
        model_obj = ModelCELOSR(flags=args)
    else:
        raise RuntimeError
    model_obj.train(flags=args)


if __name__ == "__main__":
    main()
