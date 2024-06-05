import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from color_mnist import get_biased_mnist_dataloader
from models.bias_mnist_model import Model, Model_bias
import torch
import torch.optim
from bias_mnist_train import Trainer
from models.rebias_models import ReBiasModels
from evaluator import MNISTEvaluator
from torch.utils import data
from torchvision import transforms
import numpy as np


class MNISTTrainer(Trainer):
    def _set_models(self):
        if not self.options.f_config:
            self.options.f_config = {'kernel_size': 7, 'feature_pos': 'post'}
            self.options.g_config = {'kernel_size': 1, 'feature_pos': 'post'}

        f_net = Model(**self.options.f_config)
        g_nets = Model_bias(**self.options.g_config)

        self.model = ReBiasModels(f_net, g_nets)
        self.evaluator = MNISTEvaluator(device=self.device)


def main(root="../data/color_mnist/biased_mnist",
         batch_size=256,
         train_correlation=0.990,
         n_confusing_labels=9,
         # optimizer config
         lr=0.001,
         optim='AdamP',
         n_epochs=100,
         lr_step_size=90,
         n_f_pretrain_epochs=0,
         n_g_pretrain_epochs=0,
         f_lambda_outer=1,
         g_lambda_inner=1,
         n_g_update=1,
         update_g_cls=True,
         # criterion config
         outer_criterion='RbfHSIC',
         inner_criterion='MinusRbfHSIC',
         rbf_sigma_scale_x=1,
         rbf_sigma_scale_y=1,
         rbf_sigma_x=1,
         rbf_sigma_y=1,
         update_sigma_per_epoch=False,
         hsic_alg='unbiased',
         feature_pos='post',
         # model configs
         n_g_nets=1,
         f_kernel_size=7,
         g_kernel_size=1,
         # others
         save_dir='../checkpoint_cmnist'):


    tr_loader ,tr_data,val_loader, val_data = get_biased_mnist_dataloader(root, batch_size=batch_size,
                                            data_label_correlation=train_correlation,
                                            n_confusing_labels=n_confusing_labels,
                                            train=True)


    images = torch.load('../data/color_mnist/color_mnist_images_0.990.pth')
    labels = torch.load('../data/color_mnist/color_mnist_labels_0.990.pth')
    tr_data.dataset.data = np.array(images)
    labels = labels.squeeze()
    tr_data.dataset.targets = np.array(labels)

    tr_loader = data.DataLoader(dataset=tr_data.dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)






    images_val = torch.load('../data/color_mnist/color_mnist_images_0.990_val.pth')
    labels_val = torch.load('../data/color_mnist/color_mnist_labels_0.990_val.pth')
    val_data.dataset.data = np.array(images_val)
    labels_val = labels_val.squeeze()
    tr_data.dataset.targets = np.array(labels_val)
    val_loader = data.DataLoader(dataset=val_data.dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)


    engine = MNISTTrainer(
        outer_criterion=outer_criterion,
        inner_criterion=inner_criterion,
        outer_criterion_config={'sigma_x': rbf_sigma_x, 'sigma_y': rbf_sigma_y,
                                'algorithm': hsic_alg},
        outer_criterion_detail={'sigma_x_type': rbf_sigma_x,
                                'sigma_y_type': rbf_sigma_y,
                                'sigma_x_scale': rbf_sigma_scale_x,
                                'sigma_y_scale': rbf_sigma_scale_y},
        inner_criterion_config={'sigma_x': rbf_sigma_x, 'sigma_y': rbf_sigma_y,
                                'algorithm': hsic_alg},
        inner_criterion_detail={'sigma_x_type': rbf_sigma_x,
                                'sigma_y_type': rbf_sigma_y,
                                'sigma_x_scale': rbf_sigma_scale_x,
                                'sigma_y_scale': rbf_sigma_scale_y},
        n_epochs=n_epochs,
        n_f_pretrain_epochs=n_f_pretrain_epochs,
        n_g_pretrain_epochs=n_g_pretrain_epochs,
        f_config={'num_classes': 8, 'kernel_size': f_kernel_size, 'feature_pos': feature_pos},
        g_config={'num_classes': 8, 'kernel_size': g_kernel_size, 'feature_pos': feature_pos},
        f_lambda_outer=f_lambda_outer,
        g_lambda_inner=g_lambda_inner,
        n_g_update=n_g_update,
        update_g_cls=update_g_cls,
        n_g_nets=n_g_nets,
        optimizer=optim,
        f_optim_config={'lr': lr, 'weight_decay': 0.0001},
        g_optim_config={'lr': lr, 'weight_decay': 0.0001},
        scheduler='StepLR',
        f_scheduler_config={'step_size': lr_step_size},
        g_scheduler_config={'step_size': lr_step_size},
        train_loader=tr_loader)


    engine.train(batch_size,tr_loader,tr_data,val_loaders=val_loader,val_data=val_data,
                 val_epoch_step=1,
                 update_sigma_per_epoch=update_sigma_per_epoch,
                 save_dir=save_dir)

if __name__ =='__main__':
    main()
