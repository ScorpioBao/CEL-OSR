from __future__ import print_function, absolute_import

import argparse

from train_bias import ModelBaseline, ModelCELOSR
from models.imagenet_models import resnet18, bagnet18
from models.rebias_models import  ReBiasModels


def main():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1,
                                  help="")
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
    train_arg_parser.add_argument("--lr", type=float, default=0.001,
                                  help='')
    train_arg_parser.add_argument("--lr_max", type=float, default=20.0,
                                  help='')
    train_arg_parser.add_argument('--momentum', type=float, default=0.9,
                                  help='Momentum.')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.0005,
                                  help='')
    train_arg_parser.add_argument("--logs", type=str, default='imagenet/logsResNet50/',
                                  help='')
    train_arg_parser.add_argument("--model_path", type=str, default='imagenet/models/',
                                  help='')
    train_arg_parser.add_argument("--deterministic", type=bool, default=True,
                                  help='')
    train_arg_parser.add_argument("--epochs_min", type=int, default=10,
                                  help="")
    train_arg_parser.add_argument("--loops_adv", type=int, default=15,
                                  help="")
    train_arg_parser.add_argument("--k", type=int, default=2,
                                  help="")
    train_arg_parser.add_argument("--gamma", type=float, default=1.0,
                                  help="")
    train_arg_parser.add_argument("--eta", type=float, default=200.0,
                                  help="")



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
