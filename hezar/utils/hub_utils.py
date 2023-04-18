import os.path

from huggingface_hub import HfApi, Repository

from ..constants import HEZAR_CACHE_DIR, HEZAR_HUB_ID
from ..utils.logging import get_logger


__all__ = [
    "resolve_pretrained_path",
    "get_local_cache_path",
    "exists_in_cache",
    "exists_on_hub",
    "clone_repo",
]

logger = get_logger(__name__)


def resolve_pretrained_path(hub_or_local_path):
    """
    Resolve a local or Hub path. If path exists locally it just returns the input, otherwise tries to resolve
    hub_or_local_path. If it contains the namespace (author/org) leave it as is, otherwise change to hezarai/{hub_path}

    Args:
        hub_or_local_path: repo name or id

    Returns:
        A proper pretrained path
    """
    if os.path.isdir(hub_or_local_path):
        return hub_or_local_path
    repo_id = f"{HEZAR_HUB_ID}/{hub_or_local_path}" if "/" not in hub_or_local_path else hub_or_local_path
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
    repo_id = resolve_pretrained_path(hub_path)
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
        hub_path: repo name or id
        repo_type: repo type like model, dataset, etc.

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


def clone_repo(hub_path: str, save_path: str, **kwargs):
    """
    Clone a repo on the hub to local directory

    Args:
        hub_path: repo name or id
        save_path: path to clone the repo to

    Returns:
        the local path to the repo
    """
    repo_id = resolve_pretrained_path(hub_path)
    repo = Repository(local_dir=save_path, clone_from=repo_id, **kwargs)
    return repo.local_dir
