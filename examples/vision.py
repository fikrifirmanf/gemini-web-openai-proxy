# Send a picture with a text question (any OpenAI vision-style client works).
# Requires: pip install openai
import base64
import pathlib

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8011/v1", api_key="anything")

data_url = (
    "data:image/png;base64,"
    + base64.b64encode(pathlib.Path("seal.jpg").read_bytes()).decode()
)

resp = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What animal is this? One word."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ],
)
print(resp.choices[0].message.content)
