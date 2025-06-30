# deps/deepinfra_client.py
from typing import Final
from openai import OpenAI

# put your real token in an env-var; fall back to the old hard-code
import os
DEEPINFRA_API_TOKEN: Final[str] = os.getenv(
    "DEEPINFRA_TOKEN",
    "7zQEIIApTNJ37jYJHI8Yl8fWyOPe9Drn"  # fallback
)

client = OpenAI(
    api_key = DEEPINFRA_API_TOKEN,
    base_url="https://api.deepinfra.com/v1/openai",
)
