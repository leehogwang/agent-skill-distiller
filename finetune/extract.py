import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv
from google import genai

import config

load_dotenv()


# ── Gemini helper ─────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """매 호출마다 새 Client 생성 — 세션/컨텍스트 공유 없음."""
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    response = client.models.generate_content(
        model=config.EXTRACT_MODEL,
        contents=[prompt],
    )
    return response.text.strip()


def parse_json_list(raw: str) -> list:
    """모델 응답에서 JSON 리스트 추출. 마크다운 코드펜스 대응."""
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


# ── Step 1: pattern extraction ────────────────────────────────────────────────

def build_extract_prompt(task: str, food: str, model_key: str, traj: dict, outcome: dict) -> str:
    ewc = outcome.get("EWC")
    ewc_str = f"{ewc:.4f}" if ewc is not None else "N/A"
    context = (
        f"Task: {task}\n"
        f"Food (ground truth): {food}\n"
        f"Model: {model_key}\n"
        f"Trajectory:\n"
        f"  think:  {traj.get('think', '')}\n"
        f"  action: {traj.get('action', '')}\n"
        f"  obs:    {traj.get('obs', '')}\n"
        f"Outcome:\n"
        f"  ES: {outcome.get('ES', False)}  (True = exact match)\n"
        f"  EWC: {ewc_str}  (cosine similarity, 1.0 = perfect)\n\n"
    )
    instruction = config.EXTRACT_PROMPT.format(K=config.EXTRACT_K)
    return context + instruction


def extract_all_patterns(data: dict) -> list[dict]:
    task = data.get("task", "")
    model_keys = data.get("models", [])
    results = data.get("results", [])

    all_entries = []
    total = len(results) * len(model_keys)
    done = 0

    for item in results:
        food = item["food"]
        for model_key in model_keys:
            done += 1
            traj = item.get("trajectory", {}).get(model_key, {})
            outcome = item.get("outcome", {}).get(model_key, {})

            print(f"[{done}/{total}] 추출: {food} / {model_key}", end=" ... ", flush=True)
            try:
                prompt = build_extract_prompt(task, food, model_key, traj, outcome)
                patterns = parse_json_list(call_gemini(prompt))
            except Exception as e:
                patterns = []
                print(f"오류({e})", end="")
            print(f"{len(patterns)}개 패턴")

            all_entries.append({
                "food": food,
                "model": model_key,
                "outcome": {"ES": outcome.get("ES"), "EWC": outcome.get("EWC")},
                "patterns": patterns,
            })

    return all_entries


# ── Step 2: hierarchical merge ────────────────────────────────────────────────

def merge_batch(pattern_sets: list[list]) -> list:
    combined = "\n\n".join(
        f"--- Set {i + 1} ---\n{json.dumps(ps, ensure_ascii=False)}"
        for i, ps in enumerate(pattern_sets)
    )
    prompt = config.MERGE_PROMPT + f"\n\nPattern sets:\n{combined}"
    return parse_json_list(call_gemini(prompt))


def hierarchical_merge(all_sets: list[list], H: int) -> list:
    current = all_sets
    round_num = 1
    while len(current) > 1:
        next_round = []
        batch_count = 0
        for i in range(0, len(current), H):
            batch = current[i: i + H]
            batch_count += 1
            print(f"  라운드 {round_num} 배치 {batch_count}: {len(batch)}세트 병합 ...", flush=True)
            try:
                merged = merge_batch(batch)
            except Exception as e:
                merged = [p for ps in batch for p in ps]
                print(f"    병합 오류({e}), 단순 합치기 fallback")
            next_round.append(merged)

        print(f"  라운드 {round_num} 완료: {len(current)} → {len(next_round)} 세트")
        current = next_round
        round_num += 1

    return current[0] if current else []


# ── Step 3: skill synthesis ───────────────────────────────────────────────────

def synthesize_skills(final_patterns: list) -> list:
    instruction = config.SYNTHESIS_PROMPT.format(
        max_skills=config.SYNTHESIS_MAX_SKILLS,
        max_skill_chars=config.SYNTHESIS_MAX_SKILL_CHARS,
    )
    prompt = instruction + f"\n\nPatterns:\n{json.dumps(final_patterns, ensure_ascii=False)}"
    print("스킬 합성 중 ...", flush=True)
    try:
        return parse_json_list(call_gemini(prompt))
    except Exception as e:
        print(f"합성 오류: {e}")
        return []


# ── main ──────────────────────────────────────────────────────────────────────

def run(input_path: str, pattern_output: str, skill_output: str):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Step 1
    print("\n=== Step 1: 패턴 추출 ===")
    entries = extract_all_patterns(data)

    os.makedirs(os.path.dirname(pattern_output) or ".", exist_ok=True)
    with open(pattern_output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"저장: {pattern_output}  ({len(entries)}개 엔트리)")

    # Step 2
    print("\n=== Step 2: 계층적 병합 ===")
    pattern_sets = [e["patterns"] for e in entries if e["patterns"]]
    if not pattern_sets:
        print("추출된 패턴이 없어 병합 생략")
        final_patterns = []
    elif len(pattern_sets) == 1:
        final_patterns = pattern_sets[0]
    else:
        final_patterns = hierarchical_merge(pattern_sets, config.MERGE_H)
    print(f"최종 패턴 수: {len(final_patterns)}")

    # Step 3
    print("\n=== Step 3: 스킬 합성 ===")
    skills = synthesize_skills(final_patterns)

    os.makedirs(os.path.dirname(skill_output) or ".", exist_ok=True)
    with open(skill_output, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)
    print(f"저장: {skill_output}  ({len(skills)}개 스킬)")
    for s in skills:
        print(f"  - {s.get('name', '(unnamed)')}: {s.get('description', '')[:60]}")


def main():
    parser = argparse.ArgumentParser(description="trajectory → 패턴 추출 → 스킬 합성")
    parser.add_argument("--input",          default="result/results.json")
    parser.add_argument("--pattern-output", default="result/success_failure_pattern.json")
    parser.add_argument("--skill-output",   default="result/skill.json")
    args = parser.parse_args()

    run(args.input, args.pattern_output, args.skill_output)


if __name__ == "__main__":
    main()
