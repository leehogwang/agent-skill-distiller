from abc import ABC, abstractmethod


class VLMBackend(ABC):
    @abstractmethod
    def predict(self, image_path: str, prompt: str) -> str:
        """이미지 경로와 프롬프트를 받아 모델 응답 문자열을 반환."""
        ...
