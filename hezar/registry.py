from torch import nn, optim


__all__ = [
    "models_registry",
    "preprocessors_registry",
    "datasets_registry",
    "criterions_registry",
    "optimizers_registry",
    "lr_schedulers_registry",
]

models_registry = {}
preprocessors_registry = {}
datasets_registry = {}
criterions_registry = {
    "bce": nn.BCELoss,
    "nll": nn.NLLLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "ctc": nn.CTCLoss,
}
optimizers_registry = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
lr_schedulers_registry = {
    "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau,
    "cosine_lr": optim.lr_scheduler.CosineAnnealingLR,
}



