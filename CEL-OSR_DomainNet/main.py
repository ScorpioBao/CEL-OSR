import argparse
import torch
import torchvision.models as models
from train import train_model
from models.imagenet_models import resnet18, bagnet18



def main():
    train_arg_parse = argparse.ArgumentParser(description="parser")
    train_arg_parse.add_argument("--seed", type=int, default=1,
                                  help="")
    train_arg_parse.add_argument("--class_numbers", type=int, default=100,
                                  help="")
    train_arg_parse.add_argument("--model", type=str, default='resnet18', choices=['resnet50', 'resnet18'],
                                  help='Choose architecture.')
    train_arg_parse.add_argument("--epochs", type=int, default=100,
                                    help='Number of epochs to train.')
    train_arg_parse.add_argument("--batch_size", type=int, default=64,
                                    help="")
    train_arg_parse.add_argument("--num_workers", type=int, default=4,
                                    help='Number of pre-fetching threads.')
    train_arg_parse.add_argument("--lr", type=float, default=0.01,
                                    help='')
    train_arg_parse.add_argument("--momentum", type=float, default=0.9,
                                    help='Momentum.')
    train_arg_parse.add_argument("--weight_decay", type=float, default=0.0005,
                                    help='')
    train_arg_parse.add_argument("--logs", type=str, default='logs/',
                                    help='')
    train_arg_parse.add_argument("--model_path", type=str, default='models/',
                                    help='')
    train_arg_parse.add_argument("--epoch_min", type=int, default=10,
                                    help='')
    train_arg_parse.add_argument("--k", type=int, default=3,    
                                    help='')    
    train_arg_parse.add_argument("--lr_max", type=float, default=20.0,    
                                    help='')
    train_arg_parse.add_argument("--gamma", type=float, default=1.0,
                                  help="")
    train_arg_parse.add_argument("--eta", type=float, default=200.0,
                                  help="")

    
    args = train_arg_parse.parse_args()

    if args.model == 'resnet18':
        ## 加载resnet18预训练模型
        # f_net= models.resnet18(pretrained=True)
        # num_ftrs = f_net.fc.in_features
        # f_net.fc = torch.nn.Linear(num_ftrs, args.class_numbers)
        f_net = resnet18(pretrained=True)
        num_ftrs = f_net.fc.in_features
        f_net.fc = torch.nn.Linear(num_ftrs, args.class_numbers)
        # f_net = torch.nn.DataParallel(f_net,device_ids=[0])
        # f_net.load_state_dict(torch.load('models/best_model3.pkl'),strict=False)
        g_net = bagnet18(feature_pos="post",num_classes=args.class_numbers)

    elif args.model == 'resnet50':
        ## 加载resnet50预训练模型
        model_obj = models.resnet50(pretrained=True)
        num_ftrs = model_obj.fc.in_features
        model_obj.fc = torch.nn.Linear(num_ftrs, args.class_numbers)

    train_model(f_net,g_net,args)

if __name__ == "__main__":
    main()     
        