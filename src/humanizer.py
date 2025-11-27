import time
from typing import cast

import requests
from typing_extensions import Literal

from config import CONFIG

SUBMIT_ENDPOINT_URL = "https://humanize.undetectable.ai/submit"
RETRIEVE_ENDPOINT_URL = "https://humanize.undetectable.ai/document"
API_KEY = CONFIG.HUMANIZER.API_KEY


# https://help.undetectable.ai/en/article/humanization-api-v2-p28b2n/
def humanize_text(
    text: str,
    readability: Literal[
        "High School", "University", "Doctorate", "Journalist", "Marketing"
    ] = "University",
    purpose: Literal[
        "General Writing",
        "Essay",
        "Article",
        "Marketing Material",
        "Story",
        "Cover Letter",
        "Report",
        "Business Material",
        "Legal Material",
    ] = "General Writing",
    strength: Literal["Quality", "Balanced", "More Human"] = "More Human",
    model: Literal["v2", "v11", "v11sr"] = "v11sr",
    retry_time: int = 8,
    n_retries: int = 100,
):
    headers = {"apikey": API_KEY, "Content-Type": "application/json"}

    s_payload = {
        "content": text,
        "readability": readability,
        "purpose": purpose,
        "strength": strength,
        "model": model,
    }

    s_response = requests.post(SUBMIT_ENDPOINT_URL, headers=headers, json=s_payload)
    s_response.raise_for_status()

    s_resp_json: dict[str, str] = cast(dict[str, str], s_response.json())
    assert s_resp_json["status"].strip() == "Document submitted successfully", (
        f"Unexpected status: {s_resp_json['status']}"
    )
    doc_id = s_resp_json["id"]

    r_payload = {"id": doc_id}
    for _ in range(n_retries):
        try:
            r_resp = requests.post(
                RETRIEVE_ENDPOINT_URL, headers=headers, json=r_payload
            )
            r_resp.raise_for_status()
            r_resp_json: dict[str, str] = cast(dict[str, str], r_resp.json())
            out = r_resp_json["output"]
            return out
        except requests.exceptions.HTTPError:
            time.sleep(retry_time)
            continue
    raise Exception("Retry limit reached!")
