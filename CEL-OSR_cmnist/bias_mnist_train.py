import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import warnings

import munch

warnings.filterwarnings("ignore", category=UserWarning)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from criterions.sigma_utils import median_distance, feature_dimension
from criterions import get_criterion
from optims import get_optim, get_scheduler
from utilis.matrix import accuracy
from criterions import  edl, uncertainty
from torchvision import transforms
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

class Trainer(object):
    def __init__(self,
                 outer_criterion='RbfHSIC',
                 inner_criterion='MinusRbfHSIC',
                 outer_criterion_config={'sigma': 1.0},
                 outer_criterion_detail={},
                 inner_criterion_config={},
                 inner_criterion_detail={},
                 # network settings
                 f_config={},
                 g_config={},
                 # optimiser settings
                 f_lambda_outer=1,
                 g_lambda_inner=1,
                 n_g_update=1,
                 update_g_cls=True,
                 n_g_nets=1,
                 optimizer='Adam',
                 f_optim_config=None,
                 g_optim_config=None,
                 scheduler='StepLR',
                 f_scheduler_config={'step_size': 20},
                 g_scheduler_config={'step_size': 20},
                 n_g_pretrain_epochs=0,
                 n_f_pretrain_epochs=0,
                 n_epochs=80,
                 log_step=100,
                 # adaptive sigma settings
                 train_loader=None,
                 sigma_update_sampling_rate=0.25,
                 # others
                 device='cuda:0',
                 logger=None):
        self.device = device
        self.sigma_update_sampling_rate = sigma_update_sampling_rate
        self.num_class = f_config['num_classes']
        options = {
            'outer_criterion': outer_criterion,
            'inner_criterion': inner_criterion,
            'outer_criterion_config': outer_criterion_config,
            'outer_criterion_detail': outer_criterion_detail,
            'inner_criterion_config': inner_criterion_config,
            'inner_criterion_detail': inner_criterion_detail,
            'f_config': f_config,
            'g_config': g_config,
            'f_lambda_outer': f_lambda_outer,
            'g_lambda_inner': g_lambda_inner,
            'n_g_update': n_g_update,
            'update_g_cls': update_g_cls,
            'n_g_nets': n_g_nets,
            'optimizer': optimizer,
            'f_optim_config': f_optim_config,
            'g_optim_config': g_optim_config,
            'scheduler': scheduler,
            'f_scheduler_config': f_scheduler_config,
            'g_scheduler_config': g_scheduler_config,
            'n_g_pretrain_epochs': n_g_pretrain_epochs,
            'n_f_pretrain_epochs': n_f_pretrain_epochs,
            'n_epochs': n_epochs,
        }
        self.options = munch.munchify(options)
        self.evaluator = None
        self._set_models()
        self._to_parallel()
        self._set_criterion(train_loader)
        self._set_optimizer()

    def _set_models(self):
        raise NotImplementedError

    def _to_device(self):
        self.model.f_net = self.model.f_net.to(self.device)
        self.model.g_nets = self.model.g_net.to(self.device)

    def _to_parallel(self):
        self.model.f_net = torch.nn.DataParallel(self.model.f_net).to(self.device)
        self.model.g_nets = torch.nn.DataParallel(self.model.g_net).to(self.device)

    def _set_adaptive_sigma(self, train_loader):
        if self.options.outer_criterion_detail.get('sigma_x_type') == 'median':
            self.logger.log('computing sigma from data median')
            sigma_x, sigma_y = median_distance(self.model, train_loader, self.sigma_update_sampling_rate,
                                               device=self.device)
        elif self.options.outer_criterion_detail.get('sigma_x_type') == 'dimension':
            sigma_x, sigma_y = feature_dimension(self.model, train_loader, device=self.device)
        else:
            return
        sigma_x_scale = self.options.outer_criterion_detail.get('sigma_x_scale', 1)
        sigma_y_scale = self.options.outer_criterion_detail.get('sigma_y_scale', 1)
        self.options.outer_criterion_config['sigma_x'] = sigma_x * sigma_x_scale
        self.options.outer_criterion_config['sigma_y'] = sigma_y * sigma_y_scale

        self.options.inner_criterion_config['sigma_x'] = sigma_x * sigma_x_scale
        self.options.inner_criterion_config['sigma_y'] = sigma_y * sigma_y_scale
        self.logger.log('current sigma: ({}) * {} ({}) * {}'.format(sigma_x,
                                                                    sigma_x_scale,
                                                                    sigma_y,
                                                                    sigma_y_scale,
                                                                    ))

    def _set_criterion(self, train_loader):
        self._set_adaptive_sigma(train_loader)
        self.outer_criterion = get_criterion(self.options.outer_criterion)(**self.options.outer_criterion_config)
        self.inner_criterion = get_criterion(self.options.inner_criterion)(**self.options.inner_criterion_config)
        self.classification_criterion = nn.CrossEntropyLoss()

    def _set_optimizer(self):
        f_net_parameters = self.model.f_net.parameters()
        g_net_parameters = self.model.g_net.parameters()

        if 'fc' in self.outer_criterion.__dict__:
            """[NOTE] for comparison methods (LearnedMixin, RUBi)
            """
            f_net_parameters += list(self.outer_criterion.fc.parameters())

        self.f_optimizer = get_optim(f_net_parameters,
                                     self.options.optimizer,
                                     self.options.f_optim_config)
        self.g_optimizer = get_optim(g_net_parameters,
                                     self.options.optimizer,
                                     self.options.g_optim_config)

        self.f_lr_scheduler = get_scheduler(self.f_optimizer,
                                            self.options.scheduler,
                                            self.options.f_scheduler_config)
        self.g_lr_scheduler = get_scheduler(self.g_optimizer,
                                            self.options.scheduler,
                                            self.options.g_scheduler_config)

    def _update_g(self, x, labels, cur_epoch,update_inner_loop=True):

        self.model.train()

        g_loss = 0
        preds, g_feats = self.model.g_net(x)

        _g_loss = 0
        if self.options.update_g_cls:
            #_g_loss_cls = self.classification_criterion(preds, labels)
            y_onehat = edl.one_hot_embedding(labels, 8)
            _g_loss_cls = edl.edl_log_loss(preds,y_onehat,cur_epoch,8,10)
            _g_loss += _g_loss_cls

        if update_inner_loop and self.options.g_lambda_inner:
            _, f_feats = self.model.f_net(x)
            _g_loss_inner = self.inner_criterion(g_feats, f_feats, labels=labels)
            _g_loss += self.options.g_lambda_inner * _g_loss_inner
            g_loss += _g_loss

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss,_g_loss_inner,preds
    def _update_f(self, x, labels, cur_epoch,update_outer_loop=True):


        self.model.train()

        f_loss = 0
        preds, f_feats = self.model.f_net(x)

        if self.options.outer_criterion not in ('LearnedMixin', 'RUBi'):
            """[NOTE] Comparison methods (LearnedMixin, RUBi) do not compute f_loss_cls
            """
            #f_loss_cls = self.classification_criterion(preds, labels)
            y_onehat = edl.one_hot_embedding(labels, 8)
            f_loss_cls = edl.edl_log_loss(preds,y_onehat,cur_epoch,8,10)
            f_loss += f_loss_cls


        if update_outer_loop and self.options.f_lambda_outer:
            f_loss_indep = 0

            _g_preds, _g_feats = self.model.g_net(x)

            _f_loss_indep = self.outer_criterion(f_feats, _g_feats, labels=labels, f_pred=preds, g_pred=_g_preds)
            f_loss_indep += _f_loss_indep


            f_loss += self.options.f_lambda_outer * f_loss_indep

        self.f_optimizer.zero_grad()
        f_loss.backward()
        self.f_optimizer.step()
        return f_loss,f_loss_indep,preds

    def train(self,batch_size,tr_loader,tr_data,val_loaders=None,val_data=None,val_epoch_step=1,update_sigma_per_epoch=False,save_dir='./checkpoints'):
        best_acc = 0
        counter_k = 0
        image_denormalise = Denormalise([0.5] * 3, [0.5] * 3)
        image_transform = transforms.ToPILImage()
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))])
        def sgd(parameters, lr, weight_decay=0.0, momentum=0.0):
            opt = torch.optim.SGD(params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
            return opt
        mse_loss = torch.nn.MSELoss()

        def maximize(network,  tr_loader):
            count=0
            network.eval()
            images, labels = [],[]
            for i,(images_train,labels_train) in enumerate(tr_loader):
                inputs,targets = images_train.cuda(),labels_train.cuda()
                _,features = network(inputs)
                inputs_embedding = features.detach().clone()
                inputs_embedding.requires_grad_(False)
                inputs_max = inputs.detach().clone()
                inputs_max.requires_grad_(True)
                optimizer = sgd(parameters=[inputs_max],lr=1.0)
                for ite_max in range(15):
                    preds, features = network(x=inputs_max)
                    y_onehat = edl.one_hot_embedding(targets.long(), 8)
                    uncertainty_loss = uncertainty.uncertainty_loss(preds)
                    loss = edl.edl_digamma_loss(preds,y_onehat,0,8,10) - mse_loss(features,inputs_embedding) + 200 * uncertainty_loss
                    network.zero_grad()
                    optimizer.zero_grad()
                    (-loss).backward()
                    optimizer.step()

                inputs_max = inputs_max.detach().clone().cpu()
                for j in range(len(inputs_max)):
                    input_max = image_denormalise(inputs_max[j])
                    input_max = image_transform(input_max.clamp(min=0.0, max=1.0))
                    images.append(input_max)
                    labels.append(labels_train[j].item())
                    if (len(images) % 1000== 0):
                        print("Generated {} images".format(len(images)))

            print("Add generated images to original dataset,len(dataset)", len(labels))
            return np.stack(images), labels


        print('strat training')

        for cur_epoch in range(self.options.n_epochs):
            print('F learning rate: {}, G learning rate: {}'.format(
                self.f_lr_scheduler.get_last_lr(),
                self.g_lr_scheduler.get_last_lr()))

            if((cur_epoch +1 ) % 10 ==0) and (counter_k < 1):
                images, labels = maximize(self.model.f_net, tr_loader)
                tr_data.dataset.data = np.concatenate([tr_data.dataset.data, images])
                labels = torch.tensor(labels)
                tr_data.dataset.targets=np.concatenate([tr_data.dataset.targets,labels])
                counter_k += 1
            self.model.f_net.train()

            print("dataset len(dataset)",len(tr_data.dataset.data))
            # print(len(tr_loader))
            for idx,(x,labels) in enumerate(tr_loader):
                x = x.to(self.device)
                labels = labels.to(self.device)
                g_loss,g_loss_inner,g_preds = self._update_g(x,labels,cur_epoch)
                f_loss,f_loss_indep,f_preds = self._update_f(x,labels,cur_epoch)

                if idx % 50==0:
                    train_acc, _ = accuracy(f_preds, labels, topk=(1, 5))
                    train_acc_g, _ = accuracy(g_preds, labels, topk=(1, 5))
                    print(
                        "epoch:%d,step:%d,f_train_acc:%.4f,g_train_acc:%.4f,f_loss:%.10f,g_loss:%.10f,f_loss_indep:%.10f,"
                        "g_loss_inner:%.10f" % (
                        cur_epoch, idx, train_acc.item() * 0.01, train_acc_g.item() * 0.01, f_loss, g_loss, f_loss_indep,
                        g_loss_inner))
                    correct = 0
                    self.model.f_net.eval()

                    # val_data.transform = train_transform
                    # val_loaders = torch.utils.data.DataLoader(
                    #     val_data,
                    #     batch_size=batch_size,
                    #     shuffle=True,
                    #     num_workers=0,
                    #     pin_memory=True
                    # )
                    print("dataset_val len(dataset)", len(val_loaders.dataset))
                    for idx,(x,y) in enumerate(val_loaders):
                        x, y = x.cuda(), y.cuda()
                        x = x.type(torch.FloatTensor)
                        logits, _ = self.model.f_net(x)
                        pred = logits.data.max(1)[1]
                        correct += pred.eq(y.data).sum()
                    print("target_acc:", (correct / len(val_loaders.dataset)).item())
                    print("best_acc:",best_acc)
                    correct_acc = (correct / len(val_loaders.dataset)).item()
                    if correct_acc > best_acc:
                        best_acc = correct_acc
                        torch.save(self.model.f_net.state_dict(), os.path.join("./checkpoint", 'best_model_7.pt'))

                    self.model.f_net.train()
            self.f_lr_scheduler.step()
            self.g_lr_scheduler.step()


































