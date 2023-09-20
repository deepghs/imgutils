import os
import re
import zipfile
from contextlib import contextmanager

from github import Github
from hbutils.system import TemporaryDirectory
from hfmirror.resource import LocalDirectoryResource
from hfmirror.storage import HuggingfaceStorage
from hfmirror.sync import SyncTask
from hfmirror.utils import download_file
from huggingface_hub import HfApi
from tqdm.auto import tqdm

MODEL_ASSET_PATTERN = re.compile(r'^waifu2x_onnx_models_(?P<version>[\s\S]*)\.zip$')


@contextmanager
def load_model_project():
    github_client = Github(os.environ['GH_ACCESS_TOKEN'])
    repo = github_client.get_repo('nagadomi/nunif')
    release = repo.get_release('0.0.0')
    with TemporaryDirectory() as ztd, TemporaryDirectory() as ptd:
        for asset in tqdm(release.get_assets()):
            matching = MODEL_ASSET_PATTERN.fullmatch(asset.name)
            if not matching:
                continue

            version = matching.group('version')
            url = asset.browser_download_url
            zip_file = os.path.join(ztd, asset.name)
            download_file(url, zip_file)

            version_dir = os.path.join(ptd, version)
            os.makedirs(version_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(version_dir)

        yield ptd


def sync_to_huggingface(repository: str = 'deepghs/waifu2x_onnx'):
    hf_client = HfApi(token=os.environ['HF_TOKEN'])
    hf_client.create_repo(repository, repo_type='model', exist_ok=True)
    storage = HuggingfaceStorage(repository, repo_type='model', hf_client=hf_client)

    with load_model_project() as ptd:
        resource = LocalDirectoryResource(ptd)
        task = SyncTask(resource, storage)
        task.sync()
