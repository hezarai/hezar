import os

from huggingface_hub import HfApi, Repository

from ..constants import HEZAR_CACHE_DIR, RepoType
from ..utils.logging import Logger


__all__ = [
    "get_local_cache_path",
    "exists_in_cache",
    "exists_on_hub",
    "clone_repo",
    "list_repo_files",
    "get_state_dict_from_hub",
    "clean_cache",
]

logger = Logger(__name__)


def get_local_cache_path(repo_id, repo_type):
    """
    Given the hub path and repo type, configure the local path to save everything e.g, ~/.hezar/models/<repo_name>

    Args:
        repo_id: Repo name or id
        repo_type: Repo type e.g, model, dataset, etc

    Returns:
        Path to local cache directory
    """
    repo_owner, repo_name = repo_id.split("/")
    cache_path = f"{HEZAR_CACHE_DIR}/{repo_type}s--{repo_owner}--{repo_name}"
    return cache_path


def exists_in_cache(hub_path, repo_type="model"):
    cache_path = get_local_cache_path(hub_path, repo_type)
    return os.path.exists(cache_path)


def exists_on_hub(hub_path: str, repo_type="model"):
    """
    Determine whether the repo exists on the hub or not

    Args:
        hub_path: Repo name or id
        repo_type: Repo type like model, dataset, etc.

    Returns:
        True or False
    """
    author, repo_name = hub_path.split("/")
    api = HfApi()
    if repo_type == "model":
        paths = list(iter(api.list_models(author=author)))
    elif repo_type == "dataset":
        paths = list(iter(api.list_datasets(author=author)))
    elif repo_type == "space":
        paths = list(iter(api.list_spaces(author=author)))
    else:
        raise ValueError(f"Unknown type: {repo_type}! Use `model`, `dataset`, `space`, etc.")

    return hub_path in [path.id for path in paths]


def clone_repo(repo_id: str, save_path: str, **kwargs):
    """
    Clone a repo on the hub to local directory

    Args:
        repo_id: Repo name or id
        save_path: Path to clone the repo to

    Returns:
        the local path to the repo
    """
    repo = Repository(local_dir=save_path, clone_from=repo_id, **kwargs)
    return repo.local_dir


def list_repo_files(hub_or_local_path: str, subfolder: str = None):
    """
    List all files in a Hub or local model repo

    Args:
        hub_or_local_path: Path to hub or local repo
        subfolder: Optional subfolder path

    Returns:
        A list of all file names
    """
    if os.path.isdir(hub_or_local_path):
        files_itr = os.walk(hub_or_local_path)
        files = []
        for root, dirs, files_ in files_itr:
            for file in files_:
                files.append(os.path.relpath(os.path.join(root, file), hub_or_local_path))
    else:
        files = HfApi().list_repo_files(hub_or_local_path, repo_type=str(RepoType.MODEL))

    if subfolder:
        files = [os.path.relpath(f, subfolder) for f in files if subfolder in f]
    return files


def get_state_dict_from_hub(hub_id, filename, subfolder=None):
    """
    Load a state dict from a repo on the HF Hub. Works on any repo no matter the library.

    Args:
        hub_id: Path to repo id
        filename: Weights file name
        subfolder: Optional subfolder in the repo

    Returns:
        A PyTorch state dict obj
    """
    import torch

    api = HfApi()

    subfolder = subfolder or ""

    # Download or load the cached file
    weights_file = api.hf_hub_download(
        repo_id=hub_id,
        filename=filename,
        subfolder=subfolder,
        cache_dir=HEZAR_CACHE_DIR,
    )

    state_dict = torch.load(weights_file)

    return state_dict


def clean_cache(cache_dir: str = None, delay: int = 10):
    """
    Clean the whole cache directory of Hezar

    Args:
        cache_dir: Optionally provide the cache dir path or the default cache dir will be used otherwise.
        delay: How many seconds to wait before performing the deletion action
    """
    import shutil
    import time

    cache_dir = cache_dir or HEZAR_CACHE_DIR

    logger.warning(f"Attempting to delete the files in `{cache_dir}` in {delay} seconds...")
    time.sleep(delay)

    shutil.rmtree(cache_dir)
    logger.info(f"Successfully deleted `{cache_dir}`!")
