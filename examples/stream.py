# SSE streaming: partial text arrives as the web model generates it.
# Requires: pip install openai
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8011/v1", api_key="anything")

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Count from 1 to 20, comma separated."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
