import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import dataset as dataset
from torch.utils.tensorboard import SummaryWriter
from criterions.sigma_utils import median_distance, feature_dimension
from criterions import uncertainty
import edl
from criterions import get_criterion
import torch.utils.data as Data
import numpy as np
import torchvision.transforms
from PIL import Image
from sklearn.metrics import f1_score,roc_auc_score


# parser = argparse.ArgumentParser()
# parser.add_argument("--train_args_file", type=str, default='--', help="")
# parser.add_argument("--deepspeed", type=str, default='--', help="")
# parser.add_argument('--local_rank', type=int, default=-1,
#                 help='local rank passed from distributed launcher')
# args = parser.parse_args()
# train_args_file = args.train_args_file

## 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

rank = torch.distributed.get_rank()
num_classes = 100
writer  = SummaryWriter("logs9")
best_acc = 0

def fix_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

class Denormalise(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-12)
        mean_inv = -mean * std_inv
        super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(Denormalise, self).__call__(tensor.clone())
    
class ImgDataset(Data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def HSIC(f_net,g_net,train_loader):
    outer_criterion = 'RbfHSIC'
    inner_criterion = 'MinusRbfHSIC'
    outer_criterion_config = {'sigma_x': 1.0}
    sigma_update_sampling_rate = 0.25 # 用于计算自适应核半径的采样率
    rbf_sigma_scale_x = 1
    rbf_sigma_scale_y = 1
    rbf_sigma_x = 'median'
    rbf_sigma_y = 'median'
    # rbf_sigma_x = 1
    # rbf_sigma_y = 1
    update_sigma_per_epoch = True
    hsic_alg = 'unbiased'
    outer_criterion_detail={'sigma_x_type': rbf_sigma_x,
                            'sigma_y_type': rbf_sigma_y,
                            'sigma_x_scale': rbf_sigma_scale_x,
                            'sigma_y_scale': rbf_sigma_scale_y}

    inner_criterion_config={'sigma_x': rbf_sigma_x, 'sigma_y': rbf_sigma_y,
                            'algorithm': hsic_alg}

    inner_criterion_detail={'sigma_x_type': rbf_sigma_x,
                            'sigma_y_type': rbf_sigma_y,
                            'sigma_x_scale': rbf_sigma_scale_x,
                            'sigma_y_scale': rbf_sigma_scale_y}
    
    if outer_criterion_detail.get('sigma_x_type') == 'median':
                print('computing sigma from data median')  # 从数据中位数计算sigma
                sigma_x, sigma_y = median_distance(f_net, g_net,train_loader, 0.25,
                                                device=device)
    elif outer_criterion_detail.get('sigma_x_type') == 'dimension':
        sigma_x, sigma_y = feature_dimension(f_net, g_net,train_loader, device=device)
    else:
        return
    sigma_x_scale = outer_criterion_detail.get('sigma_x_scale', 1)
    sigma_y_scale = outer_criterion_detail.get('sigma_y_scale', 1)
    outer_criterion_config['sigma_x'] = sigma_x * sigma_x_scale
    outer_criterion_config['sigma_y'] = sigma_y * sigma_y_scale

    inner_criterion_config['sigma_x'] = sigma_x * sigma_x_scale
    inner_criterion_config['sigma_y'] = sigma_y * sigma_y_scale
    print('current sigma: ({}) * {} ({}) * {}'.format(sigma_x,
                                                    sigma_x_scale,
                                                    sigma_y,
                                                    sigma_y_scale,
                                                    ))
    return inner_criterion_config,outer_criterion_config


def update_f(f_net,g_net, optimizer,x,y,epoch,outer_criterion):
        g_net.eval()
        _, g_feature = g_net(x)## _没有参与梯度计算,所以不能输出
        f_net.train()
        f_loss=0
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        f_logits, f_feature = f_net(x)
        y_onehat = edl.one_hot_embedding(y.long(), num_classes)
        f_loss_cls = edl.edl_digamma_loss(f_logits,y_onehat,epoch,num_classes,10)
        # f_loss_cls = edl.edl_mse_loss(f_logits,y_onehat,epoch,3,10)
        f_loss += f_loss_cls
        f_loss_indep = outer_criterion(g_feature,f_feature,labels=y,f_pred=f_logits,g_pred=_)
        util_adv_loss = torch.nn.functional.softplus(-_).mean() * 0 + torch.nn.functional.softplus(-g_feature).mean() * 0
        f_loss +=  f_loss_indep
        f_loss += util_adv_loss
        optimizer.zero_grad()
        f_loss.backward()
        for name, param in f_net.named_parameters():
            if param.grad is None:
                print(name)
        for name, param in g_net.named_parameters():
            if param.grad is None:
                print("g name",name)
        # print(prof)
        optimizer.step()
        return f_loss_indep.item(),f_loss.item(),f_logits

def update_g(f_net,g_net, optimizer, x,y,epoch,inner_criterion):
        f_net.eval()
        _, f_feature = f_net(x)
        g_net.train()
        g_loss=0
        g_logits, g_feature = g_net(x)
        y_onehat = edl.one_hot_embedding(y.long(), num_classes)
        g_loss_cls = edl.edl_digamma_loss(g_logits,y_onehat,epoch,num_classes,10)
        # g_loss_cls = edl.edl_mse_loss(g_logits, y_onehat, epoch, 3, 10)
        g_loss += g_loss_cls

        g_loss_inner = inner_criterion(g_feature,f_feature,labels=y,f_pred=_,g_pred=g_logits)
        util_adv_loss = torch.nn.functional.softplus(-_).mean() * 0 + torch.nn.functional.softplus(-f_feature).mean() * 0
        g_loss +=  g_loss_inner
        g_loss+=util_adv_loss
        optimizer.zero_grad()
        g_loss.backward()
        for name, param in f_net.named_parameters():
            if param.grad is None:
                print("name",name)
        for name, param in g_net.named_parameters():
            if param.grad is None:
                print(name)
        optimizer.step()
        return g_loss_inner.item(),g_loss.item(),g_logits

def maximize(f_net,train_loader,args):
     ## 生成对抗样本
    dist_fn = torch.nn.MSELoss()
    image_denormalise = Denormalise([0.5] * 3, [0.5] * 3)
    image_transform = transforms.ToPILImage()
    f_net.eval()
    images, labels = [],[]
    for i,(images_train,label_train) in enumerate(train_loader):
        inputs, targets = images_train.to(device),label_train.to(device)
        _, inputs_embedding = f_net(x=inputs)
        inputs_embedding = inputs_embedding.detach().clone() 
        inputs_embedding.require_grad = True
        inputs_max = inputs.detach().clone()
        inputs_max = inputs_max.to(device)
        inputs_max.requires_grad_(True)
        optimizer = torch.optim.SGD([inputs_max], args.lr_max)
        for ite_max in range(15):
            f_logits,f_features = f_net(x=inputs_max)
            y_onehat = edl.one_hot_embedding(targets.long(), num_classes)
            uncertainty_loss = uncertainty.uncertainty_loss(f_logits,num_classes)
            loss = edl.edl_digamma_loss(f_logits, y_onehat, 0, num_classes, 10) + args.eta * uncertainty_loss - args.gamma * dist_fn(
                f_features, inputs_embedding)
            f_net.zero_grad()
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
        inputs_max = inputs_max.detach().clone().cpu()
        for j in range(len(inputs_max)):
            input_max = image_denormalise(inputs_max[j])
            input_max = image_transform(input_max.clamp(0, 1))
            images.append(input_max)
            labels.append(label_train[j].item())
            if local_rank ==0:
                if (len(images)%500==0):
                    print("Generated {} images".format(len(images)))
                    # 将inputmax由Image格式转为tensor
                    # transform = transforms.ToTensor()
                    # input_max_w = transform(input_max)
                    # writer.add_image('Generated_images',input_max_w,i)
        if local_rank ==0:
            print("Add generated images to original dataset,len(dataset)", len(labels))
    return images, labels

            
def train_model(f_net,g_net,args):
    ## 训练模型
    ## 多卡并行训练
    fix_all_seed(seed=1+rank)
    if torch.cuda.device_count()>1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        f_net.to(device)
        f_net = torch.nn.parallel.DistributedDataParallel(f_net,device_ids=[local_rank],output_device=local_rank)
        g_net.to(device)
        g_net = torch.nn.parallel.DistributedDataParallel(g_net,device_ids=[local_rank],output_device=local_rank)
    f_net.to(device)  
    g_net.to(device)
    Data = dataset.Tiny_ImageNet_OSR(dataroot='../data', domain='real', batch_size=args.batch_size, img_size=224)
    trainloader = Data.train_loader
    valloader = Data.test_loader
    outloader = Data.out_loader
    inner_criterion_config,outer_criterion_config = HSIC(f_net,g_net,trainloader)
    outer_criterion = get_criterion('RbfHSIC')(**outer_criterion_config)
    inner_criterion = get_criterion('MinusRbfHSIC')(**inner_criterion_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(f_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    counter_k = 0
    counter_ite = 0
    write = 0
    for epoch in range(args.epochs):
        if((epoch+1)% args.epoch_min==0) and (counter_k < args.k):
            if local_rank==0:
                print('Generating adversarial images [iter {}]'.format(counter_k))
            trainloader = Data.train_loader
            images, labels = maximize(f_net,trainloader,args)

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(.4, .4, .4, .4),
                transforms.RandomGrayscale(0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_data = ImgDataset(data=images, targets=labels, transform=train_transform)
            ## 重新定义train_loader,数据为原来的数据加上生成的数据
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data+Data.train_set)
            trainloader = torch.utils.data.DataLoader(
                train_data+Data.train_set,
                batch_size=args.batch_size,
                sampler = train_sampler,
                num_workers=args.num_workers,
                pin_memory=True)
            counter_k += 1
        
        f_net.train()
        g_net.train()
        # scheduler.T_max = counter_ite + len(trainloader) * (args.epochs - epoch)
        # running_loss = 0.0
        # running_acc = 0.0
        # total= 0
        # if local_rank ==0:
        #     print(len(trainloader.dataset))
        trainloader.sampler.set_epoch(epoch)
        for i, (inputs,labels) in enumerate(trainloader):
            write = write + 1
            # inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            f_loss_inner,f_loss,f_logits = update_f(f_net,g_net, optimizer, x=inputs,y=labels,epoch=epoch,outer_criterion=outer_criterion)
            g_loss_indep, g_loss, g_logits = update_g(f_net,g_net, optimizer, x=inputs,y=labels,epoch=epoch,inner_criterion=inner_criterion)
            if local_rank==0:
                writer.add_scalar("f_loss",f_loss,write)
                if (i+1)%1==0:
                    print('epoch:', epoch+1, 'ite', i+1, 'f_total loss:', f_loss, 'lr:',
                            scheduler.get_last_lr()[0])
            if (i+1)%10 ==0:
               eval_model(f_net, args,inputs, labels,device,writer, write)
        test_model(f_net, args,valloader,outloader,device,writer,epoch)
        scheduler.step()
    writer.close()


def eval_model(f_net, args,inputs,labels,device,writer,write):
    ## 测试模型
    fix_all_seed(seed=1+rank)
    f_net.eval()
    running_acc = 0.0
    total = 0
    with torch.no_grad():
        # inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        logits, _ = f_net(inputs)
        _, preds = torch.max(logits, 1)
        running_acc += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        if local_rank ==0:
            print('Accuracy on Train: %.5f' % (
                running_acc / total))
            writer.add_scalar("Train_Accuracy",running_acc/total,write)
    # f_net.train()

       
def test_model(f_net, args,valloader,outloader,device,writer,write):
    ## 在验证集上验证，如果准确率高于之前的最高准确率，则保存模型
    fix_all_seed(seed=1+rank)
    correct = 0
    total = 0
    global best_acc
    with torch.no_grad():
        f_net.eval()
        # for data in valloader: ## 验证集上的准确率
        #     images, labels = data
        #     images, labels = images.to(device), labels.to(device)
        #     logits, _ = f_net(images)
        #     _, preds = torch.max(logits, 1)
        #     total += labels.size(0)
        #     correct += (preds == labels).sum().item()
        acc, auc = evaluation(f_net, valloader, outloader)
        if local_rank ==0:
            writer.add_scalar("Val_Accuracy",acc,write)
            writer.add_scalar("Val_AUC",auc,write)
            print('Accuracy on Val: %.5f, AUC: %.5f' % (acc, auc))
            print('Best ACC: %.5f'% (best_acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(f_net.state_dict(), 'models/best_model.pkl')
                print('model saved! best accuracy: %.5f'%(best_acc))
            # f_net.train()

def evaluation(net2, testloader, outloader):
    net2.eval()
    correct, total, n = 0, 0, 0
    torch.cuda.empty_cache()
    pred_close, pred_open, labels_close, labels_open,u_close,u_open = [], [], [], [], [],[]
    open_labels = torch.zeros(500000)
    probs = torch.zeros(500000)
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)
            with torch.set_grad_enabled(False):
                logits, _ = net2(data)
                e = edl.relu_evidence(logits)
                alpha = e + 1
                S = torch.sum(alpha, dim=1)
                S = S.view(len(labels), 1)
                U = num_classes / S
                logits = alpha / S
                # logits = torch.softmax(logits / 1, dim=1)
                confidence = logits.data.max(1)[0]
                for b in range(bsz):
                    probs[n] = confidence[b]
                    open_labels[n] = 1
                    n += 1
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                pred_close.append(logits.data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())
                u_close.append(U.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)
            oodlabel = torch.zeros_like(labels) - 1
            with torch.set_grad_enabled(False):
                logits, _ = net2(data)
                e = edl.relu_evidence(logits)
                alpha = e + 1
                S = torch.sum(alpha, dim=1)
                S = S.view(len(labels), 1)
                U = num_classes / S
                logits = alpha / S
                # logits = torch.softmax(logits / 1, dim=1)
                confidence = logits.data.max(1)[0]
                for b in range(bsz):
                    probs[n] = confidence[b]
                    open_labels[n] = 0
                    n += 1
                pred_open.append(logits.data.cpu().numpy())
                labels_open.append(oodlabel.data.cpu().numpy())
                u_open.append(U.data.cpu().numpy())
    # Accuracy
    acc = float(correct)  / float(total)
    # print('Acc: {:.5f}'.format(acc))

    pred_close = np.concatenate(pred_close, 0)#[n_close,6]概率
    pred_open = np.concatenate(pred_open, 0)#[n_open,6]概率
    labels_close = np.concatenate(labels_close, 0)#[n_close,]
    labels_open = np.concatenate(labels_open, 0)#[n_close,]   -1
    u_close = np.concatenate(u_close,0)#[n_close,]
    u_open = np.concatenate(u_open,0)#[n_open,]
    # F1 score Evaluation
    x1, x2 = np.max(pred_close, axis=1), np.max(pred_open, axis=1)
    pred1, pred2 = np.argmax(pred_close, axis=1), np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_pred = np.concatenate([x1, x2], axis=0)
    total_u = np.squeeze(np.concatenate([u_close,u_open],axis=0))
    open_pred = (total_u < 0.25).astype(np.float32)
    # open_pred = 1-open_pred
    total_pred_label= ((total_pred_label + 1) * open_pred) - 1
    f1 = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')

    # AUROC score Evaluation
    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    u = total_u[:n].reshape(-1,1)
    u = 1- u
    auc = roc_auc_score(open_labels, u)

    return acc, auc

            
    














    