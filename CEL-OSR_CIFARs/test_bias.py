import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# rootpath = str("/home/bqs/CEL-OSR")
# syspath = sys.path
# sys.path = []
# sys.path.append(rootpath)
# sys.path.extend([rootpath + i for i in os.listdir(rootpath) if i[0] != "."])
# sys.path.extend(syspath)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from sklearn.metrics import f1_score,roc_auc_score
import edl
from dataset import *
from models.imagenet_models import resnet18
from thop import profile
from thop import clever_format
import time

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]
class_num = 6
known_list = [
    [0,1, 2, 4, 5, 9],
    [0, 3, 5, 7, 8, 9],
    [0, 1, 5, 6, 7, 8],
    [3, 4, 5, 7, 8, 9],
    [0, 1, 2, 3, 7, 8]
]

known=known_list[0]  ## 0-19
unknown = list(set(list(range(0, 200))) - set(known))

train_arg_parser = argparse.ArgumentParser(description="parser")
train_arg_parser.add_argument("--seed", type=int, default=1,
                              help="")
train_arg_parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                              help='Choose between CIFAR-10, CIFAR-100.')
train_arg_parser.add_argument('--algorithm', type=str, default='CEL-OSR', choices=['ERM', 'ADA', 'CEL-OSR'],
                              help='Choose algorithm.')
train_arg_parser.add_argument('--model', type=str, default='resnext', choices=['wrn', 'allconv', 'densenet', 'resnext'],
                              help='Choose architecture.')
train_arg_parser.add_argument("--epochs", type=int, default=100,
                              help='Number of epochs to train.')
train_arg_parser.add_argument("--batch_size", type=int, default=256,
                              help="")
train_arg_parser.add_argument("--num_workers", type=int, default=4,
                              help='Number of pre-fetching threads.')
train_arg_parser.add_argument("--lr", type=float, default=0.1,
                              help='')
train_arg_parser.add_argument("--lr_max", type=float, default=20.0,
                              help='')
train_arg_parser.add_argument('--momentum', type=float, default=0.9,
                              help='Momentum.')
train_arg_parser.add_argument("--weight_decay", type=float, default=0.0005,
                              help='')
train_arg_parser.add_argument("--logs", type=str, default='cifar10/logs_0/',
                              help='')
train_arg_parser.add_argument("--model_path", type=str, default='cifar10/models_0/',
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

# WRN Architecture options
train_arg_parser.add_argument('--layers', default=40, type=int,
                              help='total number of layers')
train_arg_parser.add_argument('--widen-factor', default=2, type=int,
                              help='Widen factor')
train_arg_parser.add_argument('--droprate', default=0.0, type=float,
                              help='Dropout probability')

args = train_arg_parser.parse_args()

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

# test_transform = preprocess
# test_data = datasets.CIFAR10( '../data', train=False, transform=test_transform, download=True)
# base_c_path = os.path.join('../data', "CIFAR-10-C")


model = resnet18(num_classes=class_num)
model.load_state_dict(torch.load('./cifar10/models/best_model_0_un.tar')['state'])


model.eval()
model = model.cuda()


all_acc=[]
all_auc=[]
test_loader, test_dataset = get_cifar10_close_set(batch_size=256,train=False)
# test_loader, test_dataset = get_cifar10_c_close_set(CORRUPTIONS[0],batch_size=128)
test_loader_open, test_dataset_open = get_cifar10_open_set(batch_size=256,train=False)
print('testing_data_close:', len(test_loader.dataset.targets))
print('testing_data_open:', len(test_loader_open.dataset.targets))
acc, auc, _, open_labels_mnist, u_mnist = evaluation(model, test_loader, test_loader_open)
all_acc.append(acc)
all_auc.append(auc)
# acc, auc, f1 = evaluation2(model, test_loader, test_loader_open)
print("ACC AUC: [%.3f], [%.3f]" % (acc, auc))
# for count, corruption in enumerate(CORRUPTIONS[0:15]):
#         # Reference to original data is mutated
#         test_data.data = np.load(os.path.join(base_c_path, corruption + '.npy'))
#         test_data.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
#         Filter(test_data, known)
#         test_loader = torch.utils.data.DataLoader(
#             test_data,
#             batch_size=30000,
#             shuffle=True,
#             num_workers=0,
#             pin_memory=True)


#         test_loader, test_dataset = get_cifar10_c_close_set(corruption, batch_size=128)
#         test_loader_open, test_dataset_open = get_cifar10_open_set(batch_size=128, train=False)
#         print('testing_data_close:', len(test_loader.dataset.targets))
#         print('testing_data_open:', len(test_loader_open.dataset.targets))
#         acc, auc, f1, open_labels_mnist,u_mnist = evaluation(model, test_loader, test_loader_open)
#         print("count ACC AUC: [%.d], [%.3f], [%.3f]" % (count,acc, auc))
#         all_acc.append(acc)
#         all_auc.append(auc)
# print("mean_acc",np.mean(all_acc))
# print("mean_auc",np.mean(all_auc))