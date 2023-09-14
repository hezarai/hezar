import os
from collections import OrderedDict
from typing import List, Union

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from ..constants import DEFAULT_PREPROCESSOR_SUBFOLDER, RegistryType, RepoType, Backends
from ..utils import get_module_class, list_repo_files, verify_dependencies


class Preprocessor:
    """
    Base class for all data preprocessors.

    Args:
        config: Preprocessor properties
    """
    required_backends: List[Union[str, Backends]] = []

    preprocessor_subfolder = DEFAULT_PREPROCESSOR_SUBFOLDER

    def __init__(self, config, **kwargs):
        verify_dependencies(self, self.required_backends)  # Check if all the required dependencies are installed

        self.config = config.update(kwargs)

    def __call__(self, inputs, **kwargs):
        """
        An abstract call method for a preprocessor. All preprocessors must implement this.

        Args:
            inputs: Raw inputs to process. Usually a list or a dict
            **kwargs: Extra keyword arguments depending on the preprocessor
        """
        raise NotImplementedError

    def save(self, path, **kwargs):
        raise NotImplementedError

    def push_to_hub(
        self,
        repo_id,
        subfolder=None,
        commit_message=None,
        private=None,
        **kwargs,
    ):
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        subfolder: str = None,
        force_return_dict: bool = False,
        **kwargs
    ):
        """
        Load a preprocessor or a pipeline of preprocessors from a local or Hub path. This method automatically detects
        any preprocessor in the path. If there's only one preprocessor, returns it and if there are more, returns a
        dictionary of preprocessors.

        This method must also be overriden by subclasses as it internally calls this method for every possible
        preprocessor found in the repo.

        Args:
            hub_or_local_path: Path to hub or local repo
            subfolder: Subfolder for the preprocessor.
            force_return_dict: Whether to return a dict even if there's only one preprocessor available on the repo
            **kwargs: Extra kwargs

        Returns:
            A Preprocessor subclass or a dict of Preprocessor subclass instances
        """
        subfolder = subfolder or cls.preprocessor_subfolder
        preprocessor_files = list_repo_files(hub_or_local_path, subfolder=subfolder)
        preprocessors = PreprocessorsContainer()
        for f in preprocessor_files:
            if f.endswith(".yaml"):
                if os.path.isdir(hub_or_local_path):
                    config_file = os.path.join(hub_or_local_path, subfolder, f)
                else:
                    config_file = hf_hub_download(
                        hub_or_local_path,
                        filename=f,
                        subfolder=subfolder,
                        repo_type=RepoType.MODEL
                    )
                config = OmegaConf.load(config_file)
                name = config.get("name", None)
                if name:
                    preprocessor_cls = get_module_class(name, registry_type=RegistryType.PREPROCESSOR)
                    preprocessor = preprocessor_cls.load(hub_or_local_path, subfolder=subfolder)
                    preprocessors[name] = preprocessor
                else:
                    raise ValueError(f"The config file `{config_file}` does not have the property `name`!")
        if len(preprocessors) == 1 and not force_return_dict:
            return list(preprocessors.values())[0]

        return preprocessors


class PreprocessorsContainer(OrderedDict):
    """
    A class to hold the preprocessors by their name
    """
    def __getattr__(self, item):
        """
        Override this method to be able to access preprocessors as attributes

        Examples:
            >>> preprocessor["text_normalizer"] is preprocessor.text_normalizer  # noqa
            ... True
        """
        if item in self:
            return self[item]
        else:
            super().__getattribute__(item)

    def save(self, path):
        """
        Save every preprocessor item in the container
        """
        for name, preprocessor in self.items():
            preprocessor.save(path)

    def push_to_hub(
        self,
        repo_id,
        subfolder=None,
        commit_message=None,
        private=None,
    ):
        """
        Push every preprocessor item in the container
        """
        for name, preprocessor in self.items():
            preprocessor.push_to_hub(repo_id, subfolder=subfolder, commit_message=commit_message, private=private)
