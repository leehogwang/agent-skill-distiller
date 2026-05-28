import argparse
import json
import os
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def batch_embed(texts: list[str], client: OpenAI) -> dict[str, list[float]]:
    unique = list(dict.fromkeys(t.strip() for t in texts if t.strip()))
    if not unique:
        return {}
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=unique)
    return {item.object: vec.embedding for item, vec in zip(unique, response.data)}


def embed_all(texts: list[str], client: OpenAI) -> dict[str, list[float]]:
    unique = list(dict.fromkeys(t.strip() for t in texts if t.strip()))
    if not unique:
        return {}
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=unique)
    return {text: item.embedding for text, item in zip(unique, response.data)}


def evaluate(input_path: str, output_path: str):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    model_keys = data.get("models", [])

    client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    # 임베딩이 필요한 모든 텍스트 수집 (정답 + 예측값)
    all_texts = []
    for item in results:
        all_texts.append(item["food"])
        for pred in item["predictions"].values():
            all_texts.append(pred)

    print(f"임베딩 요청: {len(set(t.strip() for t in all_texts if t.strip()))}개 고유 텍스트")
    emb_cache = embed_all(all_texts, client)

    # 모델별 집계
    model_stats: dict[str, dict] = {k: {"es_hits": 0, "ewc_sum": 0.0, "count": 0} for k in model_keys}
    details = []

    for item in results:
        food = item["food"]
        food_emb = emb_cache.get(food.strip())
        item_preds = {}

        for model_key, pred_text in item["predictions"].items():
            pred_clean = pred_text.strip()
            es = pred_clean == food

            if food_emb and pred_clean in emb_cache:
                ewc = cosine_similarity(food_emb, emb_cache[pred_clean])
            else:
                ewc = None

            item_preds[model_key] = {"text": pred_text, "ES": es, "EWC": ewc}

            if model_key in model_stats:
                model_stats[model_key]["es_hits"] += int(es)
                model_stats[model_key]["ewc_sum"] += ewc if ewc is not None else 0.0
                model_stats[model_key]["count"] += 1

        details.append({
            "food": food,
            "image": item.get("image", ""),
            "predictions": item_preds,
        })

    summary = {}
    for k, stat in model_stats.items():
        n = stat["count"]
        summary[k] = {
            "ES": round(stat["es_hits"] / n, 4) if n else 0.0,
            "EWC": round(stat["ewc_sum"] / n, 4) if n else 0.0,
        }

    # weakness: 음식별 전체 모델 평균 ES/EWC + 모델 순위
    weakness = {}
    for detail in details:
        food = detail["food"]
        preds = detail["predictions"]
        es_vals = [int(v["ES"]) for v in preds.values()]
        ewc_vals = [v["EWC"] for v in preds.values() if v["EWC"] is not None]

        # EWC 기준 내림차순 정렬 → 순위 부여 (동점이면 ES 기준)
        ranked = sorted(
            preds.items(),
            key=lambda x: (x[1]["EWC"] if x[1]["EWC"] is not None else -1, int(x[1]["ES"])),
            reverse=True,
        )
        ranking = {model: rank + 1 for rank, (model, _) in enumerate(ranked)}

        weakness[food] = {
            "avg_ES": round(sum(es_vals) / len(es_vals), 4) if es_vals else 0.0,
            "avg_EWC": round(sum(ewc_vals) / len(ewc_vals), 4) if ewc_vals else 0.0,
            "ranking": ranking,
        }

    output = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results_file": input_path,
        "prompt": data.get("prompt", ""),
        "embedding_model": EMBEDDING_MODEL,
        "summary": summary,
        "weakness": weakness,
        "details": details,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== 평가 결과 ===")
    for model, scores in summary.items():
        print(f"  {model}: ES={scores['ES']:.2%}  EWC={scores['EWC']:.4f}")
    print(f"\n저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VLM 음식 인식 결과 평가")
    parser.add_argument("--input", default="result/results.json", help="results.json 경로")
    parser.add_argument("--output", default="result/metric.json", help="metric.json 저장 경로")
    args = parser.parse_args()
    evaluate(args.input, args.output)


if __name__ == "__main__":
    main()
