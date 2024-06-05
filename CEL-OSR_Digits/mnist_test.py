import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch
from models.mnist_model import Model
from ops.mnist_config import parser
from sklearn.metrics import roc_auc_score
import numpy as np
import mnist_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auroc
import matplotlib.pyplot as plt
from criterions import edl
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


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
                logits,_ = net2(data)
                e = edl.softplus_evidence(logits)
                alpha = e + 1
                S = torch.sum(alpha, dim=1)
                S = S.view(len(labels), 1)
                U = 6 / S
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
                e = edl.softplus_evidence(logits)
                alpha = e + 1
                S = torch.sum(alpha, dim=1)
                S = S.view(len(labels), 1)
                U = 6 / S
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
    
        # OSRC Evaluation
    oscr = compute_oscr(u_close,u_open,prediction_close,labels_close)
    
    
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

    return acc, auc, f1, open_labels, u


args = parser.parse_args()
print('Use GPU:{} for testing'.format(args.gpu))
model = Model()
if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = torch.nn.DataParallel(model).cuda(args.gpu)
else:
    model = torch.nn.DataParallel(model).cuda()


model.load_state_dict(torch.load('checkpoints/best_model.pt'))
batch_size = 128 #svhn
#
target_test_loader, target_test_dataset = mnist_split.get_mnist_dataloader_close_set('../data', args.batch_size, train=False)
print('testing_data_close:', len(target_test_dataset))
target_test_loader_open, target_test_dataset_open = mnist_split.get_mnist_dataloader_open_set('../data', args.batch_size, train=False)
print('testing_data_open:', len(target_test_dataset_open))
acc, auc, f1, open_labels_mnist,u_mnist = evaluation(model, target_test_loader, target_test_loader_open)
print("ACC AUC: [%.3f], [%.3f]" % (acc, auc))
# acu_curve(open_labels_mnist,u_mnist)


target_test_loader, target_test_dataset = mnist_split.get_svhn_dataloader_close_set(args.batch_size, train=False)
print('testing_data_close:', len(target_test_dataset))
target_test_loader_open, target_test_dataset_open = mnist_split.get_svhn_dataloader_open_set(args.batch_size, train=False)
print('testing_data_open:', len(target_test_dataset_open))
acc, auc, f1, open_labels,u = evaluation(model, target_test_loader, target_test_loader_open)
print("ACC AUC: [%.3f], [%.3f]" % (acc, auc))
# # acu_curve(open_labels,u)
#
target_test_loader, target_test_dataset = mnist_split.get_mnist_m_dataloader_close_set('../data/MNIST-M', args.batch_size, train_True=False)
print('testing_data_close:', len(target_test_dataset))
target_test_loader_open, target_test_dataset_open = mnist_split.get_mnist_m_dataloader_open_set('../data/MNIST-M', args.batch_size, train_True=False)
print('testing_data_open:', len(target_test_dataset_open))
acc, auc, f1, open_labels,u = evaluation(model, target_test_loader, target_test_loader_open)
print("ACC AUC: [%.3f], [%.3f]" % (acc, auc))
# acu_curve(open_labels,u)
#
target_test_loader, target_test_dataset = mnist_split.get_syn_dataloader_close_set(args.batch_size, train=False)
print('testing_data_close:', len(target_test_dataset))
target_test_loader_open, target_test_dataset_open = mnist_split.get_syn_dataloader_open_set(args.batch_size, train=False)
print('testing_data_open:', len(target_test_dataset_open))
acc, auc, f1, open_labels,u = evaluation(model, target_test_loader, target_test_loader_open)
print("ACC AUC: [%.3f], [%.3f]" % (acc, auc))
# # acu_curve(open_labels,u)
# #
target_test_loader, target_test_dataset = mnist_split.get_usps_dataloader_close_set(args.batch_size, train=False)
print('testing_data_close:', len(target_test_dataset))
target_test_loader_open, target_test_dataset_open = mnist_split.get_usps_dataloader_open_set(args.batch_size, train=False)
print('testing_data_open:', len(target_test_dataset_open))
acc, auc, f1, open_labels,u = evaluation(model, target_test_loader, target_test_loader_open)
print("ACC AUC: [%.3f], [%.3f]" % (acc, auc))
# acu_curve(open_labels,u)

