import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from .base import VLMBackend

load_dotenv()


class GeminiVLM(VLMBackend):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = genai.Client(api_key=os.environ["GEMINI_KEY"])

    def predict(self, image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        return response.text.strip()
