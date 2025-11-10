from dataclasses import dataclass
from typing import final


@dataclass
@final
class _AIEndpointConfig(object):
    MODEL_NAME = "gpt-oss-120b"
    ENDPOINT_URL = "http://127.0.0.1:12687/v1"
    API_KEY = "-"


@dataclass
@final
class _Config(object):
    AI_ENDPOINT = _AIEndpointConfig()


CONFIG = _Config()
