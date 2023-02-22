from torch import nn, optim


__all__ = [
    "models_registry",
    "preprocessors_registry",
    "datasets_registry",
    "criterions_registry",
    "optimizers_registry",
    "lr_schedulers_registry",
    "register_model",
    "register_preprocessor",
    "register_dataset",
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


def register_model(model_name: str, config_class):
    def register_model_class(cls):
        if model_name in models_registry:
            raise ValueError(f"Requested model `{model_name}` already exists in the registry!")

        config_class.name = model_name
        models_registry[model_name] = dict(model_class=cls, config_class=config_class)

        return cls

    return register_model_class


def register_dataset(dataset_name: str, config_class):
    def register_dataset_class(cls):
        if dataset_name in datasets_registry:
            raise ValueError(f"Requested dataset `{dataset_name}` already exists in the registry!")

        config_class.name = dataset_name
        datasets_registry[dataset_name] = dict(dataset_class=cls, config_class=config_class)

        return cls

    return register_dataset_class


def register_preprocessor(preprocessor_name: str, config_class):
    def register_preprocessor_class(cls):
        if preprocessor_name in preprocessors_registry:
            raise ValueError(f"Requested preprocessor `{preprocessor_name}` already exists in the registry!")

        config_class.name = preprocessor_name
        preprocessors_registry[preprocessor_name] = dict(preprocessor_class=cls, config_class=config_class)

        return cls

    return register_preprocessor_class
