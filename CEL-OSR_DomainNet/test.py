## 测试模型性能
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torchvision
import torch.nn as nn
from dataset import Tiny_ImageNet_OSR
from models.imagenet_models import resnet18, bagnet18
import random
import edl
from sklearn.metrics import f1_score,roc_auc_score

# rank = torch.distributed.get_rank()

def fix_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_num = 100
def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    # x1, x2 = -x1, -x2
    x1,x2 = 1-x1,1-x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    # correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    predict = np.squeeze(predict,1)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort() ## 从小到大排序

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    # roc_auc = auroc(FPR, CCR)  ###计算auc的值
    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    print(f'OSCR: {OSCR}')



    # Plot ROC Curve
    plt.figure()
    lw = 2
    plt.figure(figsize=(6, 5))
    plt.plot(FPR[:n], CCR[:n], color='darkorange',
             lw=lw, label='OSCR curve (area = %0.3f)' % OSCR)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 0], [1, 1], color='navy', lw=2, linestyle='--',alpha=0.5)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Correct Classification Rate')
    plt.legend(loc="lower right")
    plt.show()

    return OSCR



def evaluation(net2, testloader, outloader):

    net2.eval()
    correct, total, n = 0, 0, 0
    torch.cuda.empty_cache()
    pred_close, pred_open, labels_close, labels_open,u_close,u_open, inds_accurate, prediction_close = [], [], [], [], [],[],[],[]
    open_labels = torch.zeros(500000)
    probs = torch.zeros(500000)
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)
            with torch.set_grad_enabled(False):
                logits, _ = net2(data)
                e = edl.softplus_evidence(logits)
                alpha = e + 1
                S = torch.sum(alpha, dim=1)
                S = S.view(len(labels), 1)
                U = class_num / S
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
                predictions = predictions.cpu().numpy()
                labels_numpy = labels.data.cpu().numpy()
                inds_acc = (predictions==labels_numpy)

                inds_accurate.append(inds_acc)
                pred_close.append(logits.data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())
                u_close.append(U.data.cpu().numpy())
                prediction_close.append(predictions)
        prediction_close = np.concatenate(prediction_close, 0)

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
                U = class_num / S
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
    acc = float(correct) * 100. / float(total)
    # print('Acc: {:.5f}'.format(acc))

    pred_close = np.concatenate(pred_close, 0)#[n_close,6]概率
    pred_open = np.concatenate(pred_open, 0)#[n_open,6]概率
    labels_close = np.concatenate(labels_close, 0)#[n_close,]
    labels_open = np.concatenate(labels_open, 0)#[n_close,]   -1
    u_close = np.concatenate(u_close,0)#[n_close,]
    u_open = np.concatenate(u_open,0)#[n_open,]
    inds_accurate = np.concatenate(inds_accurate,0)
    # np.save(file='acc_cel.npy',arr=inds_accurate)
    # np.save(file='u1.npy',arr=u_close)
    # np.save(file='u2.npy',arr=u_open)

    # OSRC Evaluation
    oscr = compute_oscr(u_close,u_open,prediction_close,labels_close)

    total_u = np.squeeze(np.concatenate([u_close,u_open],axis=0))


    # AUROC score Evaluation
    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    u = total_u[:n].reshape(-1,1)
    u = 1- u
    auc = roc_auc_score(open_labels, u)

    return acc, auc, open_labels, u






f_net = resnet18(pretrained=False)
num_ftrs = f_net.fc.in_features
f_net.fc = torch.nn.Linear(num_ftrs, 100)
f_net = torch.nn.DataParallel(f_net, device_ids=[0])
f_net.load_state_dict(torch.load('models/best_model.pkl'))


domains = ['real', 'sketch', 'quickdraw', 'clipart', 'painting','infograph']
total_acc = []
total_auc=[]
for domain in domains:
    print("Domain", domain)
    fix_all_seed(seed=1)
    Data = Tiny_ImageNet_OSR(dataroot='../data', domain=domain, batch_size=1024, img_size=224)
    testloader = Data.test_loader
    outloader = Data.out_loader
    ## 测试模型
    fix_all_seed(seed=1)
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        f_net.eval()
        # for data in testloader: ## 验证集上的准确率
        #     images, labels = data
        #     images, labels = images.to(device), labels.to(device)
        #     # print(labels)
        #     # print(images)
        #     logits, _ = f_net(images)
        #     _, preds = torch.max(logits, 1)
        #     total += labels.size(0)
        #     correct += (preds == labels).sum().item()
        #     all_preds.extend(preds.cpu().numpy())
        #     all_labels.extend(labels.cpu().numpy())
        acc,auc,_,_ = evaluation(f_net, testloader, outloader)
        print('Accuracy of the network on the %s images: %.5f' % (domain,
        acc))
        total_acc.append(acc)
        print('AUC of the network on the %s images: %.5f' % (domain,
            auc))
        total_auc.append(auc)
print("Mean Acc", np.mean(total_acc))
print("Mean Auc", np.mean(total_auc))
    
    



# print("Confusion Matrix")
# print(confusion_matrix(all_labels, all_preds))
# plt.imshow(confusion_matrix(all_labels, all_preds))
# plt.show()



