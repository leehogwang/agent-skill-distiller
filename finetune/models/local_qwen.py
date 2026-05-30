from .base import VLMBackend


class LocalQwenVLM(VLMBackend):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError(
                "로컬 Qwen 모델 실행에 필요한 패키지가 없습니다.\n"
                "pip install torch transformers accelerate qwen-vl-utils"
            )

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    def predict(self, image_path: str, prompt: str) -> str:
        import torch

        self._load()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            from PIL import Image
            img = Image.open(image_path)
            inputs = self._processor(
                text=[text],
                images=[img],
                padding=True,
                return_tensors="pt",
            )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=128)

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        return self._processor.decode(generated[0], skip_special_tokens=True).strip()
