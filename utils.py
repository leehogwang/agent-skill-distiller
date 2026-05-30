from pathlib import Path

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def parse_food_list(path: str) -> list[str]:
    foods = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            name = parts[1].strip() if len(parts) == 2 else parts[0].strip()
            if name:
                foods.append(name)
    return foods


def find_q3_image(food_name: str, image_root: str) -> str | None:
    q3_dir = Path(image_root) / food_name / "Q3"
    if not q3_dir.exists():
        return None
    jpgs = sorted(p for p in q3_dir.iterdir() if p.suffix.upper() == ".JPG")
    return str(jpgs[0]) if jpgs else None


def embed_all(texts: list[str], client: OpenAI) -> dict[str, list[float]]:
    unique = list(dict.fromkeys(t.strip() for t in texts if t.strip()))
    if not unique:
        return {}
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=unique)
    return {text: item.embedding for text, item in zip(unique, response.data)}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
