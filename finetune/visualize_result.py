"""
Compare before/after skill results (EWC + ES).

Usage:
  python3 visualize_result.py [--before result/test_results.json] [--after result/skill_after_result.json]
"""

import argparse
import json
import sys


def load_results(path: str) -> dict[str, dict]:
    """Return {food: {model: outcome_dict}}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["food"]: item.get("outcome", {}) for item in data.get("results", [])}


def print_separator(char: str = "─", width: int = 64):
    print(char * width)


def visualize(before_path: str, after_path: str):
    before = load_results(before_path)
    after  = load_results(after_path)

    common_foods = sorted(set(before) & set(after))
    if not common_foods:
        print("[오류] 두 파일에 공통 음식 항목이 없습니다.", file=sys.stderr)
        print(f"  before 음식: {sorted(before)[:5]} ...", file=sys.stderr)
        print(f"  after  음식: {sorted(after)[:5]} ...", file=sys.stderr)
        sys.exit(1)

    all_models = sorted(set(m for d in after.values() for m in d))

    # ── 헤더 ────────────────────────────────────────────────────────────────
    print()
    print_separator("═")
    print(f"  비교: {before_path}")
    print(f"     → {after_path}")
    print(f"  공통 음식 수: {len(common_foods)}")
    print_separator("═")

    summary_rows = []

    for model in all_models:
        ewc_pairs = []   # [(before_ewc, after_ewc), ...]
        # confusion (ES 기반): [[TT,TF],[FT,FF]]
        cm = [[0, 0], [0, 0]]
        has_es = False

        for food in common_foods:
            b = before.get(food, {}).get(model, {})
            a = after.get(food, {}).get(model, {})

            b_ewc = b.get("EWC")
            a_ewc = a.get("EWC")
            if b_ewc is not None and a_ewc is not None:
                ewc_pairs.append((b_ewc, a_ewc))

            b_es = b.get("ES")
            a_es = a.get("ES")
            if b_es is not None and a_es is not None:
                has_es = True
                cm[0 if b_es else 1][0 if a_es else 1] += 1

        print()
        print(f"  모델: {model}")
        print_separator()

        # ── EWC 비교 ─────────────────────────────────────────────────────
        if ewc_pairs:
            b_ewcs = [p[0] for p in ewc_pairs]
            a_ewcs = [p[1] for p in ewc_pairs]
            avg_b = sum(b_ewcs) / len(b_ewcs)
            avg_a = sum(a_ewcs) / len(a_ewcs)
            delta  = avg_a - avg_b
            arrow  = "▲" if delta > 0 else ("▼" if delta < 0 else "─")

            print(f"  [ EWC ]")
            print(f"  {'스킬 전 평균 EWC':>18}: {avg_b:.4f}")
            print(f"  {'스킬 후 평균 EWC':>18}: {avg_a:.4f}")
            print(f"  {'변화':>18}: {arrow} {delta:+.4f}")

            # 음식별 EWC 변화 (개선/퇴보 상위 표시)
            deltas = sorted(
                [(food, b, a, a - b) for food, (b, a) in zip(common_foods, ewc_pairs)],
                key=lambda x: x[3], reverse=True,
            )
            improved = [(f, d) for f, _, _, d in deltas if d > 0]
            regressed = [(f, d) for f, _, _, d in deltas if d < 0]
            if improved:
                top = ", ".join(f"{f}({d:+.3f})" for f, d in improved[:3])
                print(f"  {'EWC 개선':>18}: {top}")
            if regressed:
                bot = ", ".join(f"{f}({d:+.3f})" for f, d in regressed[:3])
                print(f"  {'EWC 퇴보':>18}: {bot}")
        else:
            avg_b = avg_a = delta = 0.0
            print(f"  EWC 데이터 없음")

        # ── Confusion Matrix (ES 있을 때만) ─────────────────────────────
        if has_es:
            tt, tf = cm[0]
            ft, ff = cm[1]
            before_es = tt + tf
            after_es  = tt + ft
            es_delta  = after_es - before_es
            es_arrow  = "▲" if es_delta > 0 else ("▼" if es_delta < 0 else "─")

            print()
            print(f"  [ ES (Exact Match) ]")
            print(f"  {'스킬 전 정답':>18}: {before_es} / {len(common_foods)}")
            print(f"  {'스킬 후 정답':>18}: {after_es} / {len(common_foods)}")
            print(f"  {'변화':>18}: {es_arrow} {es_delta:+d} ({es_delta / len(common_foods):+.1%})")
            print()
            print(f"  {'':22s}  {'스킬 후 정답':^10}  {'스킬 후 오답':^10}")
            print(f"  {'스킬 전 정답':>22s}  {tt:^10d}  {tf:^10d}")
            print(f"  {'스킬 전 오답':>22s}  {ft:^10d}  {ff:^10d}")
            if ft > 0:
                print(f"\n  ✓ ES 개선 (오답→정답): {ft}개")
            if tf > 0:
                print(f"  ✗ ES 퇴보 (정답→오답): {tf}개")
        else:
            before_es = after_es = es_delta = 0
            print(f"\n  ES 데이터 없음")

        print_separator()
        summary_rows.append((model, avg_b, avg_a, delta, before_es, after_es, es_delta, has_es))

    # ── 전체 요약 ────────────────────────────────────────────────────────────
    print()
    print_separator("═")
    print("  전체 요약")
    print_separator("═")
    print(f"  {'모델':<28}  {'EWC 전':>8}  {'EWC 후':>8}  {'ΔEWC':>8}  {'ES 전':>6}  {'ES 후':>6}  {'ΔES':>5}")
    print_separator()
    for model, avg_b, avg_a, delta, bef_es, aft_es, es_d, has_es in summary_rows:
        es_bef = f"{bef_es}" if has_es else "N/A"
        es_aft = f"{aft_es}" if has_es else "N/A"
        es_d_s = f"{es_d:+d}" if has_es else "N/A"
        print(f"  {model:<28}  {avg_b:>8.4f}  {avg_a:>8.4f}  {delta:>+8.4f}  {es_bef:>6}  {es_aft:>6}  {es_d_s:>5}")
    print_separator("═")
    print()


def main():
    parser = argparse.ArgumentParser(description="스킬 적용 전후 결과 비교 (EWC + ES)")
    parser.add_argument("--before", default="result/test_results.json")
    parser.add_argument("--after",  default="result/skill_after_result.json")
    args = parser.parse_args()
    visualize(args.before, args.after)


if __name__ == "__main__":
    main()
