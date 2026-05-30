import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


# ── trajectory parsing ────────────────────────────────────────────────────────

def parse_trajectory(raw: str) -> dict:
    """Extract think/action/obs from model response. Falls back gracefully."""
    # strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()
    # find first {...} block
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


# ── embedding & cosine ────────────────────────────────────────────────────────

def embed_all(texts: list[str], client: OpenAI) -> dict[str, list[float]]:
    unique = list(dict.fromkeys(t.strip() for t in texts if t.strip()))
    if not unique:
        return {}
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=unique)
    return {text: item.embedding for text, item in zip(unique, response.data)}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── outcome computation ───────────────────────────────────────────────────────

def compute_outcomes(results: list[dict]) -> list[dict]:
    """Add outcome (ES + EWC) to each result item. Uses OpenAI embeddings."""
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    # collect all texts for batch embedding
    all_texts = []
    for item in results:
        all_texts.append(item["food"])
        for traj in item["trajectory"].values():
            all_texts.append(traj["obs"])

    print(f"\n임베딩 요청: {len(set(t.strip() for t in all_texts if t.strip()))}개 고유 텍스트")
    emb_cache = embed_all(all_texts, client)

    for item in results:
        food = item["food"]
        food_emb = emb_cache.get(food.strip())
        outcome = {}

        for model_key, traj in item["trajectory"].items():
            obs = traj["obs"]
            es = obs == food

            if food_emb and obs.strip() in emb_cache:
                ewc = cosine_similarity(food_emb, emb_cache[obs.strip()])
            else:
                ewc = None

            outcome[model_key] = {"text": obs, "ES": es, "EWC": ewc}

        item["outcome"] = outcome

    return results


# ── food list & image ─────────────────────────────────────────────────────────

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


def find_q3_image(food_name: str) -> str | None:
    q3_dir = Path(config.IMAGE_ROOT) / food_name / "Q3"
    if not q3_dir.exists():
        return None
    jpgs = sorted(p for p in q3_dir.iterdir() if p.suffix.upper() == ".JPG")
    return str(jpgs[0]) if jpgs else None


# ── main run ──────────────────────────────────────────────────────────────────

def run(
    model_keys: list[str],
    task: str,
    trajectory_prompt: str,
    limit: int | None,
    output_path: str,
):
    foods = parse_food_list(config.FOOD_LIST_PATH)
    if limit:
        foods = foods[:limit]

    backends = {}
    for key in model_keys:
        if key not in config.MODEL_REGISTRY:
            print(f"[경고] 알 수 없는 모델: {key} — 건너뜀", file=sys.stderr)
            continue
        print(f"[로드] {key} ...")
        backends[key] = config.MODEL_REGISTRY[key]()

    results = []
    for food in foods:
        image_path = find_q3_image(food)
        if image_path is None:
            print(f"[건너뜀] {food}: Q3 이미지 없음", file=sys.stderr)
            continue

        print(f"\n[{food}] {os.path.basename(image_path)}")
        trajectory = {}

        for key, backend in backends.items():
            try:
                raw = backend.predict(image_path, trajectory_prompt)
                traj = parse_trajectory(raw)
                trajectory[key] = traj
                print(f"  {key}: obs={traj['obs']!r}")
            except Exception as e:
                trajectory[key] = {"think": "", "action": "", "obs": f"ERROR: {e}"}
                print(f"  {key}: 오류 — {e}", file=sys.stderr)

        results.append({
            "food": food,
            "image": image_path,
            "trajectory": trajectory,
        })

    # outcome은 예측 모델과 무관하게 OpenAI embedding으로 계산
    results = compute_outcomes(results)

    output = {
        "task": task,
        "trajectory_prompt": trajectory_prompt,
        "models": model_keys,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_path}")
    return output


def main():
    available = list(config.MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser(description="VLM 음식 인식 trajectory 파이프라인")
    parser.add_argument(
        "--models", nargs="+", default=available,
        help=f"실행할 모델 키 목록 (기본: 전체). 가능한 값: {available}",
    )
    parser.add_argument(
        "--task", default=config.TASK,
        help="태스크 설명 (task 필드에 저장됨)",
    )
    parser.add_argument(
        "--trajectory-prompt", default=config.TRAJECTORY_PROMPT,
        help="VLM에 전달할 trajectory 프롬프트",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="테스트용: food_list에서 앞 N개만 처리",
    )
    parser.add_argument(
        "--output", default="result/results.json",
        help="결과 JSON 저장 경로",
    )
    args = parser.parse_args()

    run(
        model_keys=args.models,
        task=args.task,
        trajectory_prompt=args.trajectory_prompt,
        limit=args.limit,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
