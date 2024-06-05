import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auroc

from sklearn.metrics import f1_score

import torch

import numpy as np

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
from ops.mnist_config import parser
from color_mnist import get_biased_mnist_dataloader

from models.bias_mnist_model import Model, Model_bias
args = parser.parse_args()

from criterions import edl
import torchvision.transforms as transforms

print('Use GPU:{} for testing'.format(args.gpu))
model = torch.load('./checkpoint/best_model_0.990.pt')
# model['f_net']
f_config = {'kernel_size': 7, 'feature_pos': 'post'}
f_net = Model(f_config).cuda()
weight_dict={}
for k,v in model.items():
    new_k = k.replace('module.','') if 'module' in k else k
    weight_dict[new_k]= v
f_net.load_state_dict(weight_dict)
f_net.eval()

def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auroc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def evaluation(net2, testloader, outloader):

    net2.eval()
    correct, total, n = 0, 0, 0
    torch.cuda.empty_cache()
    pred_close, pred_open, labels_close, labels_open,u_close,u_open = [], [], [], [], [],[]
    open_labels = torch.zeros(50000)
    probs = torch.zeros(50000)
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
                U = 8 / S
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
                U = 8 / S
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

    # F1 score Evaluation
    x1, x2 = np.max(pred_close, axis=1), np.max(pred_open, axis=1)
    pred1, pred2 = np.argmax(pred_close, axis=1), np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_pred = np.concatenate([x1, x2], axis=0)
    total_u = np.squeeze(np.concatenate([u_close,u_open],axis=0))
    open_pred = (total_u < 0.55).astype(np.float32)
    # open_pred = 1-open_pred
    total_pred_label= ((total_pred_label + 1) * open_pred) - 1
    f1 = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')

    # AUROC score Evaluation
    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    u = total_u[:n].reshape(-1,1)
    u = 1-u
    auc = roc_auc_score(open_labels, u)

    return acc, auc, f1, open_labels, u

batch_size = 128 #svhn

val_loaders, val_data = get_biased_mnist_dataloader(root="../data/color_mnist/biased_mnist", batch_size=128,
                                                    data_label_correlation=0.1,
                                                    n_confusing_labels=9,
                                                    train=False)
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
                      ])
val_data.transform = train_transform

images_test = torch.load('../data/color_mnist/color_mnist_images_test.pth')
labels_test = torch.load('../data/color_mnist/color_mnist_labels_test.pth')
labels_test = labels_test.squeeze()
val_data.data = np.array(images_test)
val_data.targets = np.array(labels_test)
# val_data.transform = train_transform
tr_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
val_loaders, val_data_open = get_biased_mnist_dataloader(root="../data/color_mnist/biased_mnist", batch_size=128,
                                                    data_label_correlation=0.1,
                                                    n_confusing_labels=9,
                                                    train=False)
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
                      ])
val_data.transform = train_transform
images_test_open = torch.load('../data/color_mnist/color_mnist_images_test_new.pth')
labels_test_open = torch.load('../data/color_mnist/color_mnist_labels_test_new.pth')
labels_test_open = torch.zeros_like(labels_test_open) - 1
val_data_open.data = np.array(images_test_open)
val_data_open.targets = np.array(labels_test_open)
val_data_open.transform = train_transform
tr_loader_open = torch.utils.data.DataLoader(
    val_data_open,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
# for i,(x,y) in enumerate(tr_loader):
#     print(x.shape)
# for i,(x,y) in enumerate(tr_loader_open):
#     print(x.shape)


print('testing_data_close:', len(val_data))
print('testing_data_open:', len(val_data_open))
acc, auc, f1, open_labels_mnist,u_mnist = evaluation(f_net, tr_loader, tr_loader_open)
print("ACC AUC F1: [%.3f], [%.3f], [%.3f]" % (acc, auc, f1))