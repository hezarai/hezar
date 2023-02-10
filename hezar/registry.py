from copy import deepcopy

from torch import nn, optim

models_registry = {}

criterions_registry = {
    'bce': nn.BCELoss,
    'nll': nn.NLLLoss,
    'cross_entropy': nn.CrossEntropyLoss,
    'mse': nn.MSELoss,
    'ctc': nn.CTCLoss
}

optimizers_registry = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD
}

lr_schedulers_registry = {
    'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
    'cosine_lr': optim.lr_scheduler.CosineAnnealingLR
}


def build_optimizer(name, params, config=None):
    optimizer = optimizers_registry[name](params, **config)
    return optimizer


def build_scheduler(name, optimizer, config=None):
    scheduler = lr_schedulers_registry[name](optimizer, **config)
    return scheduler


def build_criterion(name, config=None):
    criterion = criterions_registry[name](**config)
    return criterion
