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
class _HumanizerConfig(object):
    API_KEY = "0c4e3755-f255-441b-8b31-936340837e1d"


@dataclass
@final
class _AIDetectorConfig(object):
    API_KEY = "5468ddc7-c691-42db-8d28-53e9b856d87d"


@dataclass
@final
class _Config(object):
    AI_ENDPOINT = _AIEndpointConfig()
    HUMANIZER = _HumanizerConfig()
    AI_DETECTOR = _AIDetectorConfig()


CONFIG = _Config()
