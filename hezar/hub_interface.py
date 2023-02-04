import os
from typing import *

import torch
from huggingface_hub import Repository, HfApi, create_repo
from omegaconf import OmegaConf, DictConfig

from hezar.utils.logging import get_logger

logger = get_logger('hezar.hub_interface')

HEZAR_HUB_ID = 'hezar-ai'
HEZAR_CACHE_DIR = f'{os.path.expanduser("~")}/.hezar'
HEZAR_SNAPSHOTS_DIR = f'{HEZAR_CACHE_DIR}/snapshots'
HEZAR_MODELS_CACHE_DIR = f'{HEZAR_CACHE_DIR}/models'
HEZAR_DATASETS_CACHE_DIR = f'{HEZAR_CACHE_DIR}/datasets'
REPO_TYPE_TO_DIR_MAPPING = dict(
    model=HEZAR_MODELS_CACHE_DIR,
    dataset=HEZAR_DATASETS_CACHE_DIR
)


class HubInterface:
    def __init__(self, repo_name_or_id: str, repo_type: str, **kwargs):
        self.repo_id, self.repo_name = self._get_repo_name_and_id(repo_name_or_id)
        self.repo_type = repo_type
        self.api = HfApi()
        self.repo = self._setup_repo(**kwargs)
        self.repo_dir = self.repo.local_dir

    @staticmethod
    def _get_repo_name_and_id(repo_name_or_id):
        repo_id = f'{HEZAR_HUB_ID}/{repo_name_or_id}' if '/' not in repo_name_or_id else repo_name_or_id
        repo_name = repo_name_or_id if '/' not in repo_name_or_id else os.path.basename(repo_name_or_id)
        return repo_id, repo_name

    def _setup_repo(self, **kwargs):
        if not self.exists(self.repo_name):
            create_repo(self.repo_id)
            logger.info(f'Created `{self.repo_id}` on the Hub!')
        else:
            logger.info(f'Repo `{self.repo_id}` already exists on the Hub, skipping repo creation...')

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

    def exists(self, repo_name) -> bool:
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
