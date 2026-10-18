# Gemini image generation through /v1/images/generations (b64 output).
# Requires: pip install openai
import base64
import pathlib

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8011/v1", api_key="anything")

img = client.images.generate(
    model="default",  # the web picks its image model when the prompt asks to generate
    prompt="a cozy watercolor of a cat reading a book by the fireplace, warm tones",
    n=1,
    response_format="b64_json",
)
path = pathlib.Path("out.png")
path.write_bytes(base64.b64decode(img.data[0].b64_json))
print(f"saved {path} ({path.stat().st_size} bytes)")
