import sys
import os
from PIL import Image as Image
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse
from sklearn.metrics import f1_score,roc_auc_score
from criterions import edl
from dataset import *
from models.imagenet_models import resnet18, resnet101, resnet50
import torch.nn as nn

class_num = 20



def set_seed(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed(1)

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
    open_labels = torch.zeros(50000)
    probs = torch.zeros(50000)
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


known_list = [
    [2, 3, 13, 30, 44, 45, 64, 66, 76, 101, 111, 121, 128, 130, 136, 158, 167, 170, 187, 193],
    [4, 11, 32, 42, 51, 53, 67, 84, 87, 104, 116, 140, 144, 145, 148, 149, 155, 168, 185, 193],
    [3, 9, 10, 20, 23, 28, 29, 45, 54, 74, 133, 143, 146, 147, 156, 159, 161, 170, 184, 195],
    [1, 15, 17, 31, 36, 44, 66, 69, 84, 89, 102, 137, 154, 160, 170, 177, 182, 185, 195, 197],
    [4, 14, 16, 33, 34, 39, 59, 69, 77, 92, 101, 103, 130, 133, 147, 161, 166, 168, 172, 173]
]


Data = Tiny_ImageNet_OSR(known=known_list[0], dataroot='../data', batch_size=32, img_size=64)
train_loader = Data.train_loader
valloader = Data.test_loader
openloader = Data.out_loader
train_data = Data
model = resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 20)
model.load_state_dict(torch.load('./imagenet/models-1/best_model.tar')['state'])
model.eval()
model = model.cuda()


val_set = Data.test_set

## 为每一张图像添加高斯噪声
val_set_noise_data = []
for i in val_set.image:
    j = 0
    image = np.array(i)
    noise = np.random.randn(image.shape[0], image.shape[1], image.shape[2]) * 0.1 ## 0.1是噪声系数 噪声的均值为0 方差为1 也就是说噪声的范围在[-0.1,0.1]之间 randn()
    noise_image = image + noise
    # val_set.image[j]=noise_image
    noise_image = noise_image.clip(min=0, max=1)
    noise_image = noise_image.astype(np.float32)
    val_set_noise_data.append(noise_image)
    j+=1

val_set_noise = Dataset_My(val_set_noise_data, val_set.target, transform=None)


val_loader_noise = torch.utils.data.DataLoader(
    val_set_noise, batch_size=32, shuffle=False,
    num_workers=0, pin_memory=True,
)



val_loader = train_data.test_loader
open_loader = train_data.out_loader
print('testing_data_close:', len(val_loader_noise.dataset.target))
print('testing_data_open:', len(open_loader.dataset.targets))
acc, auc, open_labels_mnist, u_mnist = evaluation(model, val_loader_noise, open_loader)
# acc, auc, f1 = evaluation2(model, test_loader, test_loader_open)
print("ACC AUC: [%.3f], [%.3f]" % (acc, auc))




model.load_state_dict(torch.load('./imagenet/models-1/best_model.tar')['state'])
model.eval()
model = model.cuda()

# val_loader = train_data.test_loader
# open_loader = train_data.out_loader
# print('testing_data_close:', len(val_loader.dataset.targets))
# print('testing_data_open:', len(open_loader.dataset.targets))
# acc, auc, f1, open_labels_mnist, u_mnist = evaluation(model, val_loader, open_loader)
# # acc, auc, f1 = evaluation2(model, test_loader, test_loader_open)
# print("ACC AUC F1: [%.3f], [%.3f], [%.3f]" % (acc, auc, f1))

val_set = Data.test_set

## 为每一张图像添加高斯噪声
val_set_noise_data = []
for i in val_set.image:
    j = 0
    image = np.array(i)
    noise = np.random.randn(image.shape[0], image.shape[1], image.shape[2]) * 0.2 ## 0.1是噪声系数 噪声的均值为0 方差为1 也就是说噪声的范围在[-0.1,0.1]之间 randn()
    noise_image = image + noise
    # val_set.image[j]=noise_image
    noise_image = noise_image.clip(min=0, max=1)
    noise_image = noise_image.astype(np.float32)
    val_set_noise_data.append(noise_image)
    j+=1

val_set_noise = Dataset_My(val_set_noise_data, val_set.target, transform=None)





val_loader_noise = torch.utils.data.DataLoader(
    val_set_noise, batch_size=32, shuffle=False,
    num_workers=0, pin_memory=True,
)



val_loader = train_data.test_loader
open_loader = train_data.out_loader
print('testing_data_close:', len(val_loader_noise.dataset.target))
print('testing_data_open:', len(open_loader.dataset.targets))
acc, auc, open_labels_mnist, u_mnist = evaluation(model, val_loader_noise, open_loader)
# acc, auc, f1 = evaluation2(model, test_loader, test_loader_open)
print("ACC AUC: [%.3f], [%.3f]" % (acc, auc))