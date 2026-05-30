import json
import re
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
            if " = " in name:
                name = name.split(" = ", maxsplit=1)[0].strip()
            if name:
                foods.append(name)
    return foods


def parse_food_aliases(path: str) -> dict[str, list[str]]:
    """Returns {main_name: [main_name, alias, ...]} for each food in the list."""
    aliases: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            name = parts[1].strip() if len(parts) == 2 else parts[0].strip()
            if " = " in name:
                main, alias = name.split(" = ", maxsplit=1)
                main, alias = main.strip(), alias.strip()
                aliases[main] = [main] + ([alias] if alias else [])
            else:
                name = name.strip()
                if name:
                    aliases[name] = [name]
    return aliases


def find_q3_image(food_name: str, image_root: str) -> str | None:
    q3_dir = Path(image_root) / food_name / "Q3"
    if not q3_dir.exists():
        return None
    jpgs = sorted(p for p in q3_dir.iterdir() if p.suffix.upper() == ".JPG")
    return str(jpgs[0]) if jpgs else None


def parse_trajectory(raw: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return {
                "think":  str(obj.get("think", "")),
                "action": str(obj.get("action", "")),
                "obs":    str(obj.get("obs", "")).strip(),
            }
        except json.JSONDecodeError:
            pass
    return {"think": "", "action": "", "obs": raw.strip()}


def embed_all(texts: list[str], client: OpenAI) -> dict[str, list[float]]:
    unique = list(dict.fromkeys(t.strip() for t in texts if t.strip()))
    if not unique:
        return {}
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=unique)
    return {text: item.embedding for text, item in zip(unique, response.data)}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_outcomes(
    results: list[dict],
    openai_key: str,
    aliases: dict[str, list[str]] | None = None,
) -> list[dict]:
    client = OpenAI(api_key=openai_key)

    all_texts = []
    for item in results:
        food = item["food"]
        all_texts.append(food)
        if aliases:
            all_texts.extend(aliases.get(food, [food]))
        for traj in item["trajectory"].values():
            all_texts.append(traj["obs"])

    print(f"\n임베딩 요청: {len(set(t.strip() for t in all_texts if t.strip()))}개 고유 텍스트")
    emb_cache = embed_all(all_texts, client)

    for item in results:
        food = item["food"]
        food_names = aliases.get(food, [food]) if aliases else [food]
        food_embs = [emb_cache[n.strip()] for n in food_names if n.strip() in emb_cache]
        outcome = {}

        for model_key, traj in item["trajectory"].items():
            obs = traj["obs"]
            es = obs == food
            if food_embs and obs.strip() in emb_cache:
                obs_emb = emb_cache[obs.strip()]
                ewc = max(cosine_similarity(fe, obs_emb) for fe in food_embs)
            else:
                ewc = None
            outcome[model_key] = {"text": obs, "ES": es, "EWC": ewc}

        item["outcome"] = outcome

    return results
