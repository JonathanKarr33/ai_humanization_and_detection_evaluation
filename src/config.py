from dataclasses import dataclass
from typing import final


@dataclass
@final
class _AIEndpointConfig(object):
    MODEL_NAME = "google/gemini-3-flash-preview"
    ENDPOINT_URL = "https://openrouter.ai/api/v1"
    API_KEY = "-"


@dataclass
@final
class _HumanizerConfig(object):
    API_KEY = "-"


@dataclass
@final
class _AIDetectorConfig(object):
    API_KEY = "-"


@dataclass
@final
class _Config(object):
    AI_ENDPOINT = _AIEndpointConfig()
    HUMANIZER = _HumanizerConfig()
    AI_DETECTOR = _AIDetectorConfig()


CONFIG = _Config()
