import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from .base import VLMBackend

load_dotenv()


class OpenAIVLM(VLMBackend):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    def predict(self, image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        )
        return response.choices[0].message.content.strip()
