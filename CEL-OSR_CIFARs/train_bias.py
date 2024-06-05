from __future__ import print_function, absolute_import, division
from copy import deepcopy
import time
from criterions import get_criterion
from criterions.sigma_utils import median_distance, feature_dimension
from criterions import uncertainty
import edl
from models.imagenet_models import resnet18, bagnet18
from dataset import *
from sklearn.metrics import f1_score,roc_auc_score
import torch.utils.data as Data
from torch.utils.data import Subset




class CustomSubset(Subset): 
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        ## 按照indices重新定义dataset的data和target
        self.data = self.dataset.data[indices]
        self.targets = [self.dataset.targets[i] for i in indices]


    def __getitem__(self, idx): #同时支持索引访问操作
        x, y = self.dataset[self.indices[idx]]
        return x, y

    def __len__(self): # 同时支持取长度操作
        return len(self.indices)


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


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]
class_num = 6
# known=[2,8,7,3,5,1]
# unknown = list(set(list(range(0, 10))) - set(known))

outer_criterion = 'RbfHSIC'
inner_criterion = 'MinusRbfHSIC'
outer_criterion_config = {'sigma_x': 1.0}
sigma_update_sampling_rate = 0.25 # 用于计算自适应核半径的采样率
rbf_sigma_scale_x = 1
rbf_sigma_scale_y = 1
rbf_sigma_x = 'median'
rbf_sigma_y = 'median'
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


class ModelBaseline(object):
    def __init__(self, flags):

        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        # print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        if flags.model == 'resnext':
            self.f_net = resnet18(num_classes=6,pretrained=True)
            self.g_net = bagnet18(feature_pos='post',num_classes=6)
            # self.f_net = resnext29(num_classes=num_classes)
        else:
            raise Exception('Unknown model.')
        self.f_net = self.f_net.to(device='cuda')
        self.g_net = self.g_net.to(device='cuda')

        # print(self.f_net)
        # print(self.g_net)
        # print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)
        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

    def setup_path(self, flags):

        root_folder = '../data'
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        self.preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)])
        self.train_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             self.preprocess])
        self.test_transform = transforms.Compose(
            [transforms.ToPILImage(),
                self.preprocess])

        if flags.dataset == 'cifar10':
            self.train_data = datasets.CIFAR10(root_folder, train=True, transform=self.train_transform, download=False) ## self.train_data的类型是torchvision.datasets.cifar.CIFAR10
            Filter(self.train_data, known)
            # origin_data = deepcopy(self.train_data)
            train_num = int(len(self.train_data) * 0.7)
            val_num = int(len(self.train_data) * 0.3)
            self.train_data_s, self.val_data = torch.utils.data.random_split(self.train_data, [train_num, val_num])
            print("cifar_close_train:", len(self.train_data_s))
            print("cifar_close_val:", len(self.val_data))
            ## 将train_data_s按照indices重新定义dataset的data和target
            self.train_data_s = CustomSubset(self.train_data, np.arange(train_num))
            self.val_data = CustomSubset(self.train_data, np.arange(train_num, len(self.train_data))- train_num)
            ## 将train_data_s的转为dataset类型
            self.train_data_s = ImgDataset(self.train_data_s.data, self.train_data_s.targets, transform=self.train_transform)
            self.val_data = ImgDataset(self.val_data.data, self.val_data.targets, transform=self.test_transform)
            # ## 对train_data_s以及val_data进行transform
            # self.test_data = datasets.CIFAR10(root_folder, train=False, transform=self.train_transform, download=False)
            # Filter(self.test_data, known)
            # print("cifar_close_test:", len(self.test_data))
            # self.base_c_path = os.path.join(root_folder, "CIFAR-10-C")
        else:
            print("no this dataset")

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data_s,
            batch_size=flags.batch_size,
            shuffle=True,
            num_workers=flags.num_workers,
            pin_memory=True)
        # print(outer_criterion_detail[0].get('sigma_x_type'))
        if outer_criterion_detail.get('sigma_x_type') == 'median':
            print('computing sigma from data median')  # 从数据中位数计算sigma
            sigma_x, sigma_y = median_distance(self.f_net, self.g_net,self.train_loader, 0.25,
                                               device='cuda')
        elif outer_criterion_detail.get('sigma_x_type') == 'dimension':
            sigma_x, sigma_y = feature_dimension(self.f_net, self.g_net,self.train_loader, device=self.device)
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

    def configure(self, flags):

        # for name, param in self.f_net.named_parameters():
        #     print(name, param.size())

        self.optimizer = torch.optim.SGD(
            self.f_net.parameters(),
            flags.lr,
            momentum=flags.momentum,
            weight_decay=flags.weight_decay,
            nesterov=True)

        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, flags.epochs)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def update_f(self,x,y,epoch,outer_criterion):
        self.f_net.train()
        self.g_net.eval()
        f_loss=0
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        f_logits, f_feature = self.f_net(x)
        y_onehat = edl.one_hot_embedding(y.long(), class_num)
        f_loss_cls = edl.edl_digamma_loss(f_logits,y_onehat,epoch,class_num,10)
        # f_loss_cls = edl.edl_mse_loss(f_logits,y_onehat,epoch,3,10)
        f_loss += f_loss_cls
        _, g_feature = self.g_net(x)
        f_loss_indep = outer_criterion(g_feature,f_feature,labels=y,f_pred=f_logits,g_pred=_)
        f_loss +=  f_loss_indep
        self.optimizer.zero_grad()
        f_loss.backward()
        self.optimizer.step()
        return f_loss_indep.item(),f_loss.item(),f_logits

    def update_g(self,x,y,epoch,inner_criterion):
        self.f_net.eval()
        self.g_net.train()
        g_loss=0
        # start = time.time()
        g_logits, g_feature = self.g_net(x)
        y_onehat = edl.one_hot_embedding(y.long(), class_num)
        g_loss_cls = edl.edl_digamma_loss(g_logits,y_onehat,epoch,class_num,10)
        # g_loss_cls = edl.edl_mse_loss(g_logits, y_onehat, epoch, 3, 10)
        g_loss += g_loss_cls
        _, f_feature = self.f_net(x)
        g_loss_inner = inner_criterion(g_feature,f_feature,labels=y,f_pred=_,g_pred=g_logits)
        g_loss +=  g_loss_inner
        self.optimizer.zero_grad()
        g_loss.backward()
        self.optimizer.step()
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time:",end-start)
        return g_loss_inner.item(),g_loss.item(),g_logits

    def train(self, flags):
        outer_criterion = get_criterion('RbfHSIC')(**outer_criterion_config)
        inner_criterion = get_criterion('MinusRbfHSIC')(**inner_criterion_config)
        counter_k = 0
        counter_ite = 0
        self.best_accuracy_test = -1

        for epoch in range(0, flags.epochs):
            if ((epoch + 1) % flags.epochs_min == 0) and (counter_k < flags.k):  # if T_min iterations are passed
                print('Generating adversarial images [iter {}]'.format(counter_k))
                images, labels = self.maximize(flags)
                self.train_data_s.data = np.concatenate([self.train_data_s.data, images])
                self.train_data_s.targets.extend(labels)
                counter_k += 1


            self.f_net.train()
            self.g_net.train()
            self.train_data.transform = self.train_transform
            
            self.train_loader = torch.utils.data.DataLoader(
                self.train_data_s,
                batch_size=flags.batch_size,
                shuffle=True,
                num_workers=flags.num_workers,
                pin_memory=True)
            self.scheduler.T_max = counter_ite + len(self.train_loader) * (flags.epochs - epoch)

            for i, (images_train, labels_train) in enumerate(self.train_loader):
                counter_ite += 1
                # wrap the inputs and labels in Variable
                inputs, labels = images_train.cuda(), labels_train.cuda()

                f_loss_inner,f_loss,f_logits = self.update_f(x=inputs,y=labels_train,epoch=epoch,outer_criterion=outer_criterion)
                g_loss_indep, g_loss, g_logits = self.update_g(x=inputs,y=labels_train,epoch=epoch,inner_criterion=inner_criterion)
                self.scheduler.step()

                if i % 50 == 0:
                    print(
                        'epoch:', epoch, 'ite', i, 'f_total loss:', f_loss, 'lr:',
                        self.scheduler.get_last_lr()[0])

                flags_log = os.path.join(flags.logs, 'loss_log.txt')
                write_log(str(f_loss), flags_log)
                write_log(str(g_loss), flags_log)
            if epoch % 2  == 0:
                self.test_workflow(epoch, flags)

    def test_workflow(self, epoch, flags):

        """Evaluate f_net on given corrupted dataset."""
        accuracies = []
        _, _, val_loader, val_data = get_cifar10_close_set(batch_size=flags.batch_size, train=True)
        _, _, val_loader_open, val_dataset_open = get_cifar10_open_set(batch_size=flags.batch_size, train=True)
        # val_loader, val_data = get_cifar10_close_set(batch_size=flags.batch_size, train=False)
        # val_loader_open, val_dataset_open = get_cifar10_open_set(batch_size=flags.batch_size, train=False)
        print('val_data_close:', len(val_loader.dataset))
        print('val_data_open:', len(val_loader_open.dataset))
        acc, auc, _, _, _ = evaluation(self.f_net, val_loader, val_loader_open) 
        # acc, auc, f1 = evaluation2(model, test_loader, test_loader_open)
        print("ACC AUC : [%.3f], [%.3f]" % (acc, auc))
        accuracies.append(auc)

        mean_acc = np.mean(accuracies)
        if epoch > 20 :
            if mean_acc > self.best_accuracy_test:
                self.best_accuracy_test = mean_acc

                f = open(os.path.join(flags.logs, 'best_test.txt'), mode='a')
                f.write('epoch:{}, best test accuracy:{}\n'.format(epoch, self.best_accuracy_test))
                f.close()

                if not os.path.exists(flags.model_path): 
                    os.makedirs(flags.model_path)

                outfile = os.path.join(flags.model_path, 'best_model_0_un.tar')
                torch.save({'epoch': epoch, 'state': self.f_net.state_dict()}, outfile)
        # if epoch % 10 == 0:
        #     outfile = os.path.join(flags.model_path, 'best_model_end_auc_'+str(epoch)+'_soft.tar')
        #     torch.save({'epoch': epoch, 'state': self.f_net.state_dict()}, outfile)
        # if epoch == 148:
        #     outfile = os.path.join(flags.model_path, 'best_model_end_auc_150_soft.tar')
        #     torch.save({'epoch': epoch, 'state': self.f_net.state_dict()}, outfile)







class ModelCELOSR(ModelBaseline):
    def __init__(self, flags):
        super(ModelCELOSR, self).__init__(flags)
        self.dist_fn = torch.nn.MSELoss()
        self.image_denormalise = Denormalise([0.5] * 3, [0.5] * 3)
        self.image_transform = transforms.ToPILImage()

    def maximize(self, flags):
        self.f_net.eval()
        # self.train_data.data = self.train_data.data.transpose(0, 3, 1, 2).astype(np.float32)
    
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data_s,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=True)
        images, labels = [], []

        for i, (images_train, labels_train) in enumerate(self.train_loader):
            # if i < 2:
            #     visual(images_train, labels_train)
            # wrap the inputs and labels in Variable
            inputs, targets = images_train.cuda(), labels_train.cuda()
            _, inputs_embedding = self.f_net(x=inputs)
            inputs_embedding = inputs_embedding.detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = sgd(parameters=[inputs_max], lr=flags.lr_max)
            
            # start = time.time()
            for ite_max in range(flags.loops_adv):
                f_logits,f_features = self.f_net(x=inputs_max)
                # loss
                # loss = self.loss_fn(tuples[0], targets) + flags.eta * entropy_loss(tuples[0]) - \
                #        flags.gamma * self.dist_fn(tuples[-1]['Embedding'], inputs_embedding)
                y_onehat = edl.one_hot_embedding(targets.long(), class_num)
                uncertainty_loss = uncertainty.uncertainty_loss(f_logits,num_classes=6)
                loss = edl.edl_digamma_loss(f_logits, y_onehat, 0, class_num, 10) + flags.eta * uncertainty_loss - flags.gamma * self.dist_fn(
                    f_features, inputs_embedding)
                self.f_net.zero_grad()
                optimizer.zero_grad()

                # backward your f_net
                (-loss).backward()

                # optimize the parameters
                optimizer.step()

                # if i < 2:
                #     visual(inputs_max.detach().cpu(), labels_train)
                flags_log = os.path.join(flags.logs, 'max_loss_log.txt')
                write_log('ite_adv:{}, {}'.format(ite_max, loss.item()), flags_log)
            # torch.cuda.synchronize()
            # end = time.time()
            # print('time',end-start)

            inputs_max = inputs_max.detach().clone().cpu()
            for j in range(len(inputs_max)):
                input_max = self.image_denormalise(inputs_max[j])
                input_max = self.image_transform(input_max.clamp(min=0.0, max=1.0))
                images.append(input_max)
                labels.append(labels_train[j].item())
                if (len(images) % 2000 == 0):
                    print("Generated {} images".format(len(images)))
            print("Add generated images to original dataset,len(dataset)", len(labels))


        return np.stack(images), labels
