from criterions.hsic import RbfHSIC, MinusRbfHSIC
from criterions.dist import L1Loss, MSELoss
from criterions.comparison_methods import RUBi, LearnedMixin
from criterions.edl import edl_mse_loss, edl_log_loss

__all__ = ['RbfHSIC', 'MinusRbfHSIC',
           'L1Loss', 'MSELoss',
           'RUBi', 'LearnedMixin','edl_mse_loss','edl_log_loss']


def get_criterion(criterion_name):
    return globals()[criterion_name]