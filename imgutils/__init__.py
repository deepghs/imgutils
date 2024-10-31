"""
Fixed an issue where HTTP requests were rejected in offline mode for some libraries
Add here to make sure all calls fire this code
"""

import requests
from huggingface_hub import configure_http_backend
from huggingface_hub.utils import OfflineModeIsEnabled

from requests.adapters import HTTPAdapter


class CustomOfflineAdapter(HTTPAdapter):
    def send(self, request, *args, **kwargs):
        blocked_domains = ["huggingface.co", "hf.co"]
        if any(domain in request.url for domain in blocked_domains):
            raise OfflineModeIsEnabled(f"Cannot reach {request.url}: offline mode is enabled.")
        return super().send(request, *args, **kwargs)


def backend_factory() -> requests.Session:
    """
    Any HTTP calls made by `huggingface_hub` will use a
    Session object instantiated by this factory
    """
    session = requests.Session()
    session.mount("http://", CustomOfflineAdapter())
    session.mount("https://", CustomOfflineAdapter())
    return session


configure_http_backend(backend_factory=backend_factory)
