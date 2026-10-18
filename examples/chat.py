# Requires: pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8011/v1",
    api_key="anything",  # the proxy doesn't check unless GWOP_API_KEY is set
)

resp = client.chat.completions.create(
    model="default",  # or a model id from /v1/models, e.g. "gemini-pro"
    messages=[
        {"role": "system", "content": "You are a terse pirate."},
        {"role": "user", "content": "Say hi in one sentence."},
    ],
)
print(resp.choices[0].message.content)
message = resp.model_dump()["choices"][0]["message"]
print(message.get("reasoning_content"))  # model thoughts, when the model chose to think
