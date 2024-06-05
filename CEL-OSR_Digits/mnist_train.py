import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.matrix import accuracy
from torch.autograd import Variable
from download_and_process import get_all_source_dataloader
import time
from criterions import  uncertainty
import edl
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

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
                pred_close.append(logits.data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())
                u_close.append(U.data.cpu().numpy())

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

    pred_close = np.concatenate(pred_close, 0)
    pred_open = np.concatenate(pred_open, 0)
    labels_close = np.concatenate(labels_close, 0)
    labels_open = np.concatenate(labels_open, 0)
    u_close = np.concatenate(u_close,0)
    u_open = np.concatenate(u_open,0)
    # F1 score Evaluation
    x1, x2 = np.max(pred_close, axis=1), np.max(pred_open, axis=1)
    pred1, pred2 = np.argmax(pred_close, axis=1), np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_pred = np.concatenate([x1, x2], axis=0)
    total_u = np.squeeze(np.concatenate([u_close,u_open],axis=0))
    # thr = 0.5 / 6 + (1 - 0.5)
    # open_pred = (total_pred > thr - 0.05).astype(np.float32)
    open_pred = (total_u > 0.5).astype(np.float32)
    open_pred = 1-open_pred
    f1 = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')

    # AUROC score Evaluation
    open_labels = open_labels[:n].cpu().numpy()
    # prob = probs[:n].reshape(-1, 1)
    u = total_u[:n].reshape(-1,1)
    u = 1-u
    auc = roc_auc_score(open_labels, u)


    return acc, auc, f1


def train(z_hat,target_test_loader,target_test_loader_open,f_model,g_model,criterion_train_2,
          f_min_optimizer,g_min_optimizer,max_optimizer,args,outer_criterion,inner_criterion):
    all_source_data_or = torch.tensor(torch.load('../data/split_mnist_source_data.pth')).cuda()
    all_source_data_or = all_source_data_or[0:25056] #9984
    all_source_labels_or = torch.tensor(torch.load('../data/split_mnist_source_labels.pth')).cuda()
    all_source_labels_or = all_source_labels_or[0:25056] #9984
    G_data = torch.empty(size=(32, 3, 32, 32), dtype=torch.float32).cuda()
    G_labels = torch.empty(size=(32,), dtype=torch.float32).cuda()
    counter_k = 0

    def update_g(x,y,t):
        g_model.train()
        f_model.eval()
        g_loss = 0
        g_logits, g_feature = g_model(x)
        y_onehat = edl.one_hot_embedding(y.long(), 6)
        g_loss_cls = edl.edl_digamma_loss(g_logits,y_onehat,t,6,10)
        g_loss += g_loss_cls
        _, f_feature = f_model(x)
        g_loss_inner = inner_criterion(g_feature,f_feature,labels=y)
        g_loss +=  g_loss_inner
        g_min_optimizer.zero_grad()
        g_loss.backward()
        g_min_optimizer.step()
        return g_loss_inner.item(),g_loss.item(),g_logits

    def update_f(x,y,t):
        f_model.train()
        g_model.eval()
        f_loss = 0
        f_logits, f_feature = f_model(x)
        y_onehat = edl.one_hot_embedding(y.long(), 6)
        f_loss_cls = edl.edl_digamma_loss(f_logits,y_onehat,t,6,10)
        f_loss += f_loss_cls
        _, g_feature = g_model(x)
        f_loss_indep = outer_criterion(g_feature,f_feature,labels=y,f_pred=f_logits,g_pred=_)
        f_loss +=  f_loss_indep
        f_min_optimizer.zero_grad()
        f_loss.backward()
        f_min_optimizer.step()
        return f_loss_indep.item(),f_loss.item(),f_logits

    for epoch in range(3):
        if epoch ==0:
            all_source_data = all_source_data_or
            all_source_labels = all_source_labels_or
        all_source_dataloader = get_all_source_dataloader(args.batch_size, all_source_data, all_source_labels)
        for i, (x,y) in enumerate(all_source_dataloader):
            x = Variable(x,requires_grad=False)
            y = Variable(y,requires_grad=False)
            g_loss_inner,g_loss,g_logits = update_g(x,y,epoch)
            f_loss_indep, f_loss,f_logits = update_f(x,y,epoch)


            if i % 50 ==0:
                train_acc,_ = accuracy(f_logits.cuda(),y.cuda(),topk=(1, 5))
                train_acc_g,_ = accuracy(g_logits.cuda(),y.cuda(),topk=(1,5))
                # print("epoch:%d,step:%d,f_train_acc:%.4f,f_loss:%.4f" % ( epoch,i, train_acc.item() * 0.01,min_loss))
                print("epoch:%d,step:%d,f_train_acc:%.4f,g_train_acc:%.4f,f_loss:%.10f,g_loss:%.10f"
                       % ( epoch,i, train_acc.item() * 0.01,train_acc_g.item() * 0.01,f_loss,g_loss))
                acc,auc,f1 = evaluation(f_model,target_test_loader,target_test_loader_open)
                print("ACC AUC F1: [%.3f], [%.3f], [%.3f]" % (acc, auc, f1))

            if (((i+1) % 100==0) and counter_k < 3):
                print('Generating adversarial images [iter %d]' % (counter_k))
                ger_source_dataloader = get_all_source_dataloader(args.batch_size, all_source_data, all_source_labels)
                G_data_tmp = torch.empty(size=(32, 3, 32, 32), dtype=torch.float32).cuda()
                G_labels_tmp = torch.empty(size=(32,), dtype=torch.float32).cuda()
                for i, (x, y) in enumerate(ger_source_dataloader):
                    # if i < 1:
                    #     visual(x,y,count)

                    x = Variable(x,requires_grad=False).cuda()
                    y = Variable(y,requires_grad=False).cuda()
                    logits, feature = f_model(x)
                    feature = feature.detach()
                    z_hat.data = x.detach().clone()
                    z_hat.requires_grad_(True)
                    y_onehat = edl.one_hot_embedding(y.long(),6)
                    f_model.eval()
                    for n in range(args.T_adv):
                        logits_hat, feature_hat = f_model(z_hat)
                        #max_loss1 = criterion_train_1(logits_hat, y.long())
                        max_loss1 = edl.edl_digamma_loss(logits_hat,y_onehat,counter_k,6,10)
                        max_loss2 = criterion_train_2(feature, feature_hat)
                        # entropy_loss = entropy.entropy_loss(logits_hat)
                        uncertainty_loss = uncertainty.uncertainty_loss(logits_hat,num_classes=6)
                        max_loss = max_loss1 - args.gamma * max_loss2 + 150 * uncertainty_loss
                        max_loss = - max_loss
                        f_model.zero_grad()
                        max_optimizer.zero_grad()
                        max_loss.backward()
                        max_optimizer.step()
                        # if i < 1:
                        #     visual(z_hat.detach().cpu(), y,count)

                    if i % 100==0:
                        print("Generated {0} images".format((i+1)*args.batch_size))
                    learn_imgs_tmp = z_hat.cuda()
                    learn_label_tmp = y.cuda()
                    G_data_tmp = torch.cat([G_data_tmp.cuda(),learn_imgs_tmp])
                    G_labels_tmp = torch.cat([G_labels_tmp.cuda(),learn_label_tmp])
                    G_data = G_data_tmp[32:]
                    G_labels = G_labels_tmp[32:]


                all_source_data = all_source_data.detach().cpu()
                G_data = G_data.detach().cpu()
                all_source_labels = all_source_labels.detach().cpu()
                G_labels = G_labels.detach().cpu()

                all_source_data = torch.cat([all_source_data, G_data], dim=0)
                all_source_labels = torch.cat([all_source_labels, G_labels], dim=0)
                print('Add learned images to original data,len_data:%d'%(len(all_source_data)))
                counter_k += 1
                break;




    all_source_dataloader = get_all_source_dataloader(args.batch_size, all_source_data.cuda(), all_source_labels.cuda())
    best_correct = 0
    best_auc = 0
    scheduler = lr_scheduler.StepLR(optimizer=f_min_optimizer,step_size=7,gamma=0.1)

    for t in range(3,10):
        print(scheduler.get_last_lr())

        for i, (x, y) in enumerate(all_source_dataloader):

            f_loss_indep, f_loss,f_logits = update_f(x,y,t=t)
            g_loss_inner,g_loss,g_logits = update_g(x,y,t=t)

            if i % 100 == 0:
                f_train_acc, _ = accuracy(f_logits, y, topk=(1, 5))
                g_train_acc,_ = accuracy(g_logits,y,topk=(1,5))
                print("epoch:%d,step:%d,f_train_acc:%.4f,g_train_acc:%.4f,f_loss:%.4f,g_loss:%.4f" % (t, i, f_train_acc.item() * 0.01, g_train_acc.item() * 0.01,f_loss, g_loss))

                f_model.eval()
                acc, auc,f1 = evaluation(f_model,target_test_loader,target_test_loader_open)
                print("ACC AUC F1: [%.3f], [%.3f], [%.3f]" % (acc, auc, f1))
                if auc > best_auc:
                    best_auc = auc

                    torch.save(f_model.state_dict(), os.path.join("checkpoints", 'best_model.pt'))
                print('best_AUC:', best_auc)
        scheduler.step()
    now_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    torch.save(f_model.state_dict(),os.path.join("checkpoints",'model'+now_time+'.pt'))





























