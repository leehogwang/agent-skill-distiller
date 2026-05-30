import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


# ── trajectory parsing ────────────────────────────────────────────────────────

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
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])

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


# ── main run ──────────────────────────────────────────────────────────────────

def run(input_path: str, skill_path: str, output_path: str):
    # 스킬 로드
    with open(skill_path, encoding="utf-8") as f:
        skills = json.load(f)
    if not skills:
        print("[오류] skill.json이 비어 있습니다.", file=sys.stderr)
        sys.exit(1)
    skill = skills[0]
    skill_prefix = skill.get("body", "")
    skill_name = skill.get("name", "unknown")

    # 기존 results.json에서 food+image 목록 재사용
    with open(input_path, encoding="utf-8") as f:
        prev = json.load(f)

    model_keys = prev.get("models", list(config.MODEL_REGISTRY.keys()))
    food_items = [
        {"food": r["food"], "image": r["image"]}
        for r in prev.get("results", [])
    ]

    # 프롬프트 = 스킬 body + 기존 TRAJECTORY_PROMPT
    trajectory_prompt = skill_prefix + "\n\n" + config.TRAJECTORY_PROMPT

    # 모델 로드
    backends = {}
    for key in model_keys:
        if key not in config.MODEL_REGISTRY:
            print(f"[경고] 알 수 없는 모델: {key} — 건너뜀", file=sys.stderr)
            continue
        print(f"[로드] {key} ...")
        backends[key] = config.MODEL_REGISTRY[key]()

    # 추론
    results = []
    for item in food_items:
        food = item["food"]
        image_path = item["image"]

        if not os.path.exists(image_path):
            print(f"[건너뜀] {food}: 이미지 없음 ({image_path})", file=sys.stderr)
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

    # outcome 계산 (OpenAI embedding, 예측 모델과 무관)
    results = compute_outcomes(results)

    output = {
        "task": prev.get("task", config.TASK),
        "trajectory_prompt": trajectory_prompt,
        "skill_applied": True,
        "skill_name": skill_name,
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
    parser = argparse.ArgumentParser(description="스킬 적용 후 VLM 재평가")
    parser.add_argument("--input",  default="result/results.json",          help="기존 results.json 경로 (food/image 목록 재사용)")
    parser.add_argument("--skill",  default="result/skill.json",            help="skill.json 경로")
    parser.add_argument("--output", default="result/skill_after_result.json", help="결과 저장 경로")
    args = parser.parse_args()

    run(args.input, args.skill, args.output)


if __name__ == "__main__":
    main()
