import os
import shutil
from typing import *
import logging

import torch
from huggingface_hub import snapshot_download, Repository, HfApi, create_repo
from omegaconf import OmegaConf, DictConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.Logger(__name__)
logger.setLevel(20)

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

    def _get_repo_name_and_id(self, repo_name_or_id):
        repo_id = f'{HEZAR_HUB_ID}/{repo_name_or_id}' if '/' not in repo_name_or_id else repo_name_or_id
        repo_name = repo_name_or_id if '/' not in repo_name_or_id else os.path.basename(repo_name_or_id)
        return repo_id, repo_name

    def _setup_repo(self, **kwargs):
        if self.model_exists(self.repo_name):
            create_repo(self.repo_id)
            logging.info(f'Created `{self.repo_id}` on the Hub!')
        else:
            logging.info(f'Repo `{self.repo_id}` already exists on the Hub, skipping repo creation...')

        local_dir = f'{REPO_TYPE_TO_DIR_MAPPING[self.repo_type]}/{self.repo_name}'
        repo = Repository(local_dir=local_dir, clone_from=self.repo_id, **kwargs)
        return repo

    def list_models(self):
        models = self.api.list_models(author=HEZAR_HUB_ID)
        model_names = [model.modelId.split('/')[-1] for model in models]
        return model_names

    def model_exists(self, model_name) -> bool:
        models = self.list_models()
        return model_name in models

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
