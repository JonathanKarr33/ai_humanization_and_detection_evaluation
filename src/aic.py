from openai import OpenAI

from config import CONFIG

CLIENT = OpenAI(
    base_url=CONFIG.AI_ENDPOINT.ENDPOINT_URL, api_key=CONFIG.AI_ENDPOINT.API_KEY
)
