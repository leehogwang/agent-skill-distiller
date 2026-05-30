import argparse
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

import config
from utils import parse_food_list, find_q3_image, embed_all, cosine_similarity

load_dotenv()


def run(model_keys: list[str], prompt: str, output_path: str):
    foods = parse_food_list(config.FOOD_LIST_TEST_PATH)

    backends = {}
    for key in model_keys:
        if key not in config.MODEL_REGISTRY:
            print(f"[경고] 알 수 없는 모델: {key} — 건너뜀", file=sys.stderr)
            continue
        print(f"[로드] {key} ...")
        backends[key] = config.MODEL_REGISTRY[key]()

    raw_results = []
    for food in foods:
        image_path = find_q3_image(food, config.IMAGE_ROOT)
        if image_path is None:
            print(f"[건너뜀] {food}: Q3 이미지 없음", file=sys.stderr)
            continue

        print(f"\n[{food}] {os.path.basename(image_path)}")
        predictions = {}
        for key, backend in backends.items():
            try:
                answer = backend.predict(image_path, prompt)
                predictions[key] = answer
                print(f"  {key}: {answer}")
            except Exception as e:
                predictions[key] = f"ERROR: {e}"
                print(f"  {key}: 오류 — {e}", file=sys.stderr)

        raw_results.append({"food": food, "image": image_path, "predictions": predictions})

    # ES + EWC 계산
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])
    all_texts = [item["food"] for item in raw_results]
    for item in raw_results:
        all_texts.extend(item["predictions"].values())

    print(f"\n임베딩 요청: {len(set(t.strip() for t in all_texts if t.strip()))}개 고유 텍스트")
    emb_cache = embed_all(all_texts, client)

    model_stats = {k: {"es_hits": 0, "ewc_sum": 0.0, "count": 0} for k in model_keys}
    details = []

    for item in raw_results:
        food = item["food"]
        food_emb = emb_cache.get(food.strip())
        item_preds = {}

        for model_key, pred_text in item["predictions"].items():
            pred_clean = pred_text.strip()
            es = pred_clean == food
            ewc = cosine_similarity(food_emb, emb_cache[pred_clean]) if food_emb and pred_clean in emb_cache else None
            item_preds[model_key] = {"text": pred_text, "ES": es, "EWC": ewc}

            if model_key in model_stats:
                model_stats[model_key]["es_hits"] += int(es)
                model_stats[model_key]["ewc_sum"] += ewc if ewc is not None else 0.0
                model_stats[model_key]["count"] += 1

        details.append({"food": food, "image": item["image"], "predictions": item_preds})

    summary = {}
    for k, stat in model_stats.items():
        n = stat["count"]
        summary[k] = {
            "ES":  round(stat["es_hits"] / n, 4) if n else 0.0,
            "EWC": round(stat["ewc_sum"] / n, 4) if n else 0.0,
        }

    weakness = {}
    for detail in details:
        food = detail["food"]
        preds = detail["predictions"]
        es_vals  = [int(v["ES"]) for v in preds.values()]
        ewc_vals = [v["EWC"] for v in preds.values() if v["EWC"] is not None]
        ranked = sorted(
            preds.items(),
            key=lambda x: (x[1]["EWC"] if x[1]["EWC"] is not None else -1, int(x[1]["ES"])),
            reverse=True,
        )
        weakness[food] = {
            "avg_ES":  round(sum(es_vals)  / len(es_vals),  4) if es_vals  else 0.0,
            "avg_EWC": round(sum(ewc_vals) / len(ewc_vals), 4) if ewc_vals else 0.0,
            "ranking": {model: rank + 1 for rank, (model, _) in enumerate(ranked)},
        }

    if not config.EVAL_SHOW_ES:
        for detail in details:
            for pred in detail["predictions"].values():
                pred.pop("ES", None)

    output = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "prompt":          prompt,
        "food_list":       config.FOOD_LIST_TEST_PATH,
        "embedding_model": "text-embedding-3-small",
        "summary":         summary,
        "weakness":        weakness,
        "details":         details,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== 평가 결과 (test set) ===")
    for model, scores in summary.items():
        print(f"  {model}: ES={scores['ES']:.2%}  EWC={scores['EWC']:.4f}")
    print(f"\n저장: {output_path}")


def main():
    available = list(config.MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser(description="VLM 음식 인식 평가 (test set 추론 + ES/EWC)")
    parser.add_argument("--models", nargs="+", default=available)
    parser.add_argument("--prompt", default=config.PROMPT)
    parser.add_argument("--output", default="result/metric.json")
    args = parser.parse_args()

    run(model_keys=args.models, prompt=args.prompt, output_path=args.output)


if __name__ == "__main__":
    main()
