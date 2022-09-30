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
    'adamW': optim.AdamW,
    'sgd': optim.SGD
}
