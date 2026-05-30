"""
Iterative skill improvement loop.

Usage:
  python3 loop.py [--epochs N] [--models MODEL1 MODEL2 ...]
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime

from dotenv import load_dotenv
from google import genai

import config
from utils import (
    parse_food_list,
    find_q3_image,
    parse_trajectory,
    compute_outcomes,
)

load_dotenv()


# ── Gemini helper ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    response = client.models.generate_content(
        model=config.EXTRACT_MODEL,
        contents=[prompt],
    )
    return response.text.strip()


def _parse_json_list(raw: str) -> list:
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


# ── A. TRAIN 추론 (pipeline) ──────────────────────────────────────────────────

def _run_pipeline(
    model_keys: list[str],
    trajectory_prompt: str,
    epoch_dir: str,
) -> dict:
    foods = parse_food_list(config.FOOD_LIST_TRAIN_PATH)

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

        print(f"\n[TRAIN/{food}] {os.path.basename(image_path)}")
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
        "task": config.TASK,
        "trajectory_prompt": trajectory_prompt,
        "models": model_keys,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    os.makedirs(epoch_dir, exist_ok=True)
    out_path = os.path.join(epoch_dir, "train_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")
    return output


# ── B. TEST 추론 (스킬 없음, 로그용) ─────────────────────────────────────────

def _run_evaluate(
    model_keys: list[str],
    epoch_dir: str,
) -> dict:
    foods = parse_food_list(config.FOOD_LIST_TEST_PATH)

    backends = {}
    for key in model_keys:
        if key not in config.MODEL_REGISTRY:
            continue
        backends[key] = config.MODEL_REGISTRY[key]()

    results = []
    for food in foods:
        image_path = find_q3_image(food, config.IMAGE_ROOT)
        if image_path is None:
            print(f"[건너뜀] {food}: Q3 이미지 없음", file=sys.stderr)
            continue

        print(f"\n[TEST/{food}] {os.path.basename(image_path)}")
        trajectory = {}
        for key, backend in backends.items():
            try:
                raw = backend.predict(image_path, config.TRAJECTORY_PROMPT)
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

    if not config.EVAL_SHOW_ES:
        for item in results:
            for outcome in item.get("outcome", {}).values():
                outcome.pop("ES", None)

    output = {
        "task": config.TASK,
        "trajectory_prompt": config.TRAJECTORY_PROMPT,
        "food_list": config.FOOD_LIST_TEST_PATH,
        "models": model_keys,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    os.makedirs(epoch_dir, exist_ok=True)
    out_path = os.path.join(epoch_dir, "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")
    return output


# ── C. 패턴 추출 + 스킬 합성 ─────────────────────────────────────────────────

def _build_extract_prompt(
    task: str, food: str, model_key: str, traj: dict, outcome: dict
) -> str:
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
    return context + config.EXTRACT_PROMPT.format(K=config.EXTRACT_K)


def _extract_all_patterns(data: dict) -> list[dict]:
    task = data.get("task", "")
    available = set(config.MODEL_REGISTRY.keys())
    model_keys = [m for m in data.get("models", []) if m in available]
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
                prompt = _build_extract_prompt(task, food, model_key, traj, outcome)
                patterns = _parse_json_list(_call_gemini(prompt))
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


def _merge_batch(pattern_sets: list[list]) -> list:
    combined = "\n\n".join(
        f"--- Set {i + 1} ---\n{json.dumps(ps, ensure_ascii=False)}"
        for i, ps in enumerate(pattern_sets)
    )
    return _parse_json_list(_call_gemini(config.MERGE_PROMPT + f"\n\nPattern sets:\n{combined}"))


def _hierarchical_merge(all_sets: list[list], H: int) -> list:
    current = all_sets
    round_num = 1
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), H):
            batch = current[i: i + H]
            print(f"  라운드 {round_num}: {len(batch)}세트 병합 ...", flush=True)
            try:
                merged = _merge_batch(batch)
            except Exception as e:
                merged = [p for ps in batch for p in ps]
                print(f"    병합 오류({e}), 단순 합치기 fallback")
            next_round.append(merged)
        print(f"  라운드 {round_num} 완료: {len(current)} → {len(next_round)} 세트")
        current = next_round
        round_num += 1
    return current[0] if current else []


def _synthesize_skills(final_patterns: list, max_skill_chars: int) -> list:
    instruction = config.SYNTHESIS_PROMPT.format(
        max_skills=config.SYNTHESIS_MAX_SKILLS,
        max_skill_chars=max_skill_chars,
    )
    prompt = instruction + f"\n\nPatterns:\n{json.dumps(final_patterns, ensure_ascii=False)}"
    print("스킬 합성 중 ...", flush=True)
    try:
        return _parse_json_list(_call_gemini(prompt))
    except Exception as e:
        print(f"합성 오류: {e}")
        return []


def _extract_and_synthesize(
    train_data: dict,
    max_skill_chars: int,
    epoch_dir: str,
) -> list:
    print("\n=== 패턴 추출 ===")
    entries = _extract_all_patterns(train_data)

    pattern_path = os.path.join(epoch_dir, "success_failure_pattern.json")
    with open(pattern_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"저장: {pattern_path}")

    print("\n=== 계층적 병합 ===")
    pattern_sets = [e["patterns"] for e in entries if e["patterns"]]
    if not pattern_sets:
        print("추출된 패턴 없음 — 병합 생략")
        final_patterns = []
    elif len(pattern_sets) == 1:
        final_patterns = pattern_sets[0]
    else:
        final_patterns = _hierarchical_merge(pattern_sets, config.MERGE_H)
    print(f"최종 패턴 수: {len(final_patterns)}")

    print("\n=== 스킬 합성 ===")
    skills = _synthesize_skills(final_patterns, max_skill_chars)

    skill_path = os.path.join(epoch_dir, "skill.json")
    with open(skill_path, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)
    print(f"저장: {skill_path}  ({len(skills)}개 스킬)")
    return skills


# ── D. 기존 스킬 + 새 스킬 병합 ─────────────────────────────────────────────

def _merge_skills(
    current_skill: dict,
    new_skill: dict,
    max_skill_chars: int,
) -> list:
    instruction = config.SYNTHESIS_MERGE_PROMPT.format(
        max_skills=config.SYNTHESIS_MAX_SKILLS,
        max_skill_chars=max_skill_chars,
    )
    skills_input = json.dumps([current_skill, new_skill], ensure_ascii=False)
    prompt = instruction + f"\n\nSkills to merge:\n{skills_input}"
    print("스킬 병합 중 ...", flush=True)
    try:
        return _parse_json_list(_call_gemini(prompt))
    except Exception as e:
        print(f"병합 오류: {e}")
        return [new_skill]


# ── E. TEST 추론 (스킬 적용) ──────────────────────────────────────────────────

def _run_eval_with_skill(
    model_keys: list[str],
    skill: dict,
    epoch_dir: str,
) -> dict:
    trajectory_prompt = skill.get("body", "") + "\n\n" + config.TRAJECTORY_PROMPT

    foods = parse_food_list(config.FOOD_LIST_TEST_PATH)

    backends = {}
    for key in model_keys:
        if key not in config.MODEL_REGISTRY:
            continue
        backends[key] = config.MODEL_REGISTRY[key]()

    results = []
    for food in foods:
        image_path = find_q3_image(food, config.IMAGE_ROOT)
        if image_path is None:
            print(f"[건너뜀] {food}: Q3 이미지 없음", file=sys.stderr)
            continue

        print(f"\n[SKILL/{food}] {os.path.basename(image_path)}")
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
        "task": config.TASK,
        "trajectory_prompt": trajectory_prompt,
        "skill_applied": True,
        "skill_name": skill.get("name", "unknown"),
        "food_list": config.FOOD_LIST_TEST_PATH,
        "models": model_keys,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    os.makedirs(epoch_dir, exist_ok=True)
    out_path = os.path.join(epoch_dir, "skill_after_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")
    return output


# ── EWC 평균 (전체 모델 × 전체 음식) ─────────────────────────────────────────

def _avg_ewc(result_data: dict) -> float:
    total, count = 0.0, 0
    for item in result_data.get("results", []):
        for outcome in item.get("outcome", {}).values():
            ewc = outcome.get("EWC")
            if ewc is not None:
                total += ewc
                count += 1
    return total / count if count > 0 else -float("inf")


# ── best epoch → log/best 복사 ───────────────────────────────────────────────

def _copy_to_best(best_epoch: int):
    src = f"log/epochs/{best_epoch}ep"
    dst = "log/best"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if os.path.exists(src):
        shutil.copytree(src, dst)
        print(f"log/best ← {src}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Skill improvement loop")
    parser.add_argument("--epochs", type=int, default=config.LOOP_EPOCHS)
    parser.add_argument("--models", nargs="+", default=list(config.MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    model_keys    = args.models
    patience      = config.EARLY_STOPPING_PATIENCE
    best_ewc      = -float("inf")
    best_skill    = None
    best_epoch    = 0
    current_skill = None  # None = 스킬 없음

    print(f"\n루프 시작: epochs={args.epochs}, patience={patience}, lr={config.LOOP_LEARNING_RATE}")
    print(f"모델: {model_keys}\n")

    for epoch in range(1, args.epochs + 1):
        max_chars = int(
            config.SYNTHESIS_MAX_SKILL_CHARS * (config.LOOP_LEARNING_RATE ** (epoch - 1))
        )

        print(f"\n{'═' * 64}")
        print(f"  Epoch {epoch} / {args.epochs}  (max_skill_chars={max_chars})")
        print(f"{'═' * 64}")

        if current_skill:
            train_prompt = current_skill["body"] + "\n\n" + config.TRAJECTORY_PROMPT
        else:
            train_prompt = config.TRAJECTORY_PROMPT

        epoch_dir = f"log/epochs/{epoch}ep"
        os.makedirs(epoch_dir, exist_ok=True)

        # A. TRAIN 추론
        print("\n--- A. TRAIN 추론 ---")
        train_data = _run_pipeline(model_keys, train_prompt, epoch_dir)

        # B. TEST 추론 (스킬 없음, 로그용)
        print("\n--- B. TEST 추론 (스킬 없음) ---")
        _run_evaluate(model_keys, epoch_dir)

        # C. 패턴 추출 + 스킬 합성
        print("\n--- C. 패턴 추출 + 스킬 합성 ---")
        new_skills = _extract_and_synthesize(train_data, max_chars, epoch_dir)

        # D. 기존 스킬 있으면 병합, 없으면 그대로 사용
        if current_skill and new_skills:
            print("\n--- D. 기존 스킬 병합 ---")
            candidate_skills = _merge_skills(current_skill, new_skills[0], max_chars)
        else:
            candidate_skills = new_skills

        candidate = candidate_skills[0] if candidate_skills else None

        # E. 스킬 적용 TEST 추론
        if candidate:
            print("\n--- E. TEST 추론 (스킬 적용) ---")
            skill_after = _run_eval_with_skill(model_keys, candidate, epoch_dir)
            new_ewc = _avg_ewc(skill_after)
        else:
            print("[경고] 합성된 스킬 없음 — 스킬 적용 스킵")
            new_ewc = -float("inf")

        print(f"\n[Epoch {epoch}] EWC {new_ewc:.4f}  (best {best_ewc:.4f})  patience {patience}")

        if new_ewc > best_ewc:
            best_ewc      = new_ewc
            best_skill    = candidate
            best_epoch    = epoch
            current_skill = candidate
            patience      = config.EARLY_STOPPING_PATIENCE
            print("  ▲ 개선 — 스킬 업데이트, patience 초기화")
        else:
            patience -= 1
            print(f"  ▼ 미개선 — 이전 스킬 유지, patience → {patience}")
            if patience <= 0:
                print("  조기 종료 (early stopping)")
                break

    print(f"\n{'═' * 64}")
    print(f"  완료. 최고 EWC {best_ewc:.4f} (epoch {best_epoch})")
    print(f"{'═' * 64}")

    if best_epoch > 0:
        _copy_to_best(best_epoch)
    os.makedirs("result", exist_ok=True)
    with open("result/best_skill.json", "w", encoding="utf-8") as f:
        json.dump([best_skill] if best_skill else [], f, ensure_ascii=False, indent=2)
    print("result/best_skill.json 저장됨")


if __name__ == "__main__":
    main()
