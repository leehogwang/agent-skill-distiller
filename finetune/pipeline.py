import argparse
import json
import os
import sys
from datetime import datetime

import config
from utils import parse_food_list, find_q3_image, parse_trajectory, compute_outcomes


def run(
    model_keys: list[str],
    task: str,
    trajectory_prompt: str,
    limit: int | None,
    output_path: str,
):
    foods = parse_food_list(config.FOOD_LIST_TRAIN_PATH)
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
        image_path = find_q3_image(food, config.IMAGE_ROOT)
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

    results = compute_outcomes(results, os.environ["OPENAI_KEY"])

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

    parser = argparse.ArgumentParser(description="VLM 음식 인식 trajectory 파이프라인 (train set)")
    parser.add_argument(
        "--models", nargs="+", default=available,
        help=f"실행할 모델 키 목록 (기본: 전체). 가능한 값: {available}",
    )
    parser.add_argument("--task",              default=config.TASK)
    parser.add_argument("--trajectory-prompt", default=config.TRAJECTORY_PROMPT)
    parser.add_argument("--limit",             type=int, default=None)
    parser.add_argument("--output",            default="result/results.json")
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
