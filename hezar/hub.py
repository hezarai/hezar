import os
from typing import *

import torch
from huggingface_hub import Repository, HfApi, create_repo
from omegaconf import OmegaConf, DictConfig

from hezar.utils.logging import get_logger

HEZAR_HUB_ID = 'hezar-ai'
HEZAR_CACHE_DIR = os.getenv('HEZAR_CACHE_DIR', f'{os.path.expanduser("~")}/.hezar')
HEZAR_TMP_DIR = os.getenv('HEZAR_TMP_DIR', f'{HEZAR_CACHE_DIR}/tmp')
HEZAR_SNAPSHOTS_DIR = os.getenv('HEZAR_SNAPSHOTS_DIR', f'{HEZAR_CACHE_DIR}/snapshots')
HEZAR_MODELS_CACHE_DIR = os.getenv('HEZAR_MODELS_CACHE_DIR', f'{HEZAR_CACHE_DIR}/models')
HEZAR_DATASETS_CACHE_DIR = os.getenv('HEZAR_DATASETS_CACHE_DIR', f'{HEZAR_CACHE_DIR}/datasets')
REPO_TYPE_TO_DIR_MAPPING = dict(
    model=HEZAR_MODELS_CACHE_DIR,
    dataset=HEZAR_DATASETS_CACHE_DIR
)

logger = get_logger('hezar.hub_interface')


def resolve_hub_path(hub_path):
    """
    If hub_path contains the namespace leave it as is, otherwise change to hezar-ai/{hub_path}

    Args:
        hub_path: repo name or id

    Returns:
        A proper repo id on the hub
    """
    repo_id = f'{HEZAR_HUB_ID}/{hub_path}' if '/' not in hub_path else hub_path
    return repo_id


def get_local_cache_path(hub_path, repo_type):
    """
    Given the hub path and repo type, configure the local path to save everything e.g, ~/.hezar/models/<repo_name>

    Args:
        hub_path: repo name or id
        repo_type: repo type e.g, model, dataset, etc

    Returns:
        path to local cache directory
    """
    repo_id = resolve_hub_path(hub_path)
    repo_name = repo_id.split('/')[1]
    cache_path = f'{REPO_TYPE_TO_DIR_MAPPING[repo_type]}/{repo_name}'
    return cache_path


def exists_on_hub(hub_path: str, type='model'):
    """
    Determine whether the repo exists on the hub or not

    Args:
        hub_path: repo name or id
        type: repo type like model, dataset, etc.

    Returns:
        True or False
    """
    author, repo_name = hub_path.split('/')
    api = HfApi()
    if type == 'model':
        paths = list(iter(api.list_models(author=author)))
    elif type == 'dataset':
        paths = list(iter(api.list_datasets(author=author)))
    elif type == 'space':
        paths = list(iter(api.list_spaces(author=author)))
    else:
        raise ValueError(f'Unknown type: {type}! Use `model`, `dataset`, `space`, etc.')

    return hub_path in [path.id for path in paths]


def clone_repo(hub_path: str, save_path: str, **kwargs):
    """
    Clone a repo on the hub to local directory

    Args:
        hub_path: repo name or id
        save_path: path to clone the repo to

    Returns:
        the local path to the repo
    """
    repo_id = resolve_hub_path(hub_path)
    repo = Repository(local_dir=save_path, clone_from=repo_id, **kwargs)
    return repo.local_dir


class HubInterface:
    def __init__(self, hub_path: str, repo_type: str, init_repo: bool = False, **kwargs):
        self.repo_id, self.repo_name = self._get_repo_name_and_id(hub_path)
        self.repo_type = repo_type
        self.api = HfApi()
        if init_repo:
            self.repo = self._setup_repo(**kwargs)
        self.repo_dir = self.repo.local_dir

    @staticmethod
    def _get_repo_name_and_id(hub_path):
        repo_id = f'{HEZAR_HUB_ID}/{hub_path}' if '/' not in hub_path else hub_path
        repo_name = hub_path if '/' not in hub_path else os.path.basename(hub_path)
        return repo_id, repo_name

    def _setup_repo(self, **kwargs):
        if not self.exists_on_hub(self.repo_name):
            create_repo(self.repo_id)
            logger.info(f'Created `{self.repo_id}` on the Hub!')
        else:
            logger.info(f'Repo `{self.repo_id}` already exists_on_hub on the Hub, skipping repo creation...')

        local_dir = f'{REPO_TYPE_TO_DIR_MAPPING[self.repo_type]}/{self.repo_name}'
        repo = Repository(local_dir=local_dir, clone_from=self.repo_id, **kwargs)
        return repo

    def list_models(self):
        models = self.api.list_models(author=HEZAR_HUB_ID)
        model_names = [model.modelId.split('/')[-1] for model in models]
        return model_names

    def list_datasets(self):
        datasets = self.api.list_datasets(author=HEZAR_HUB_ID)
        dataset_names = [dataset.datasetId.split('/')[-1] for dataset in datasets]
        return dataset_names

    def exists_on_hub(self, repo_name) -> bool:
        if self.repo_type == 'model':
            repos = self.list_models()
        elif self.repo_type == 'dataset':
            repos = self.list_datasets()
        else:
            raise ValueError(f'Unknown repo type : `{self.repo_type}`')
        return repo_name in repos

    def push_to_hub(self, commit_message: str = 'Commit from Hezar'):
        self.repo.push_to_hub(commit_message=commit_message)

    def get_config(self, config_file='config.yaml', model_config_class=None):
        config_file_path = os.path.join(self.repo_dir, config_file)
        dict_config = OmegaConf.load(config_file_path)
        if model_config_class:
            config = model_config_class.from_dict(dict_config)
            config.pretrained_path = f'{HEZAR_HUB_ID}/{self.repo_name}'
            return config
        return dict_config

    def get_model_registry_name(self):
        config = self.get_config()
        return config.model.name

    def get_model(self, model_file='pytorch_model.bin', return_state_dict=True):
        model_file_path = os.path.join(self.repo_dir, model_file)
        if return_state_dict:
            state_dict = torch.load(model_file_path)
            return state_dict
        else:
            return model_file_path
