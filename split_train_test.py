import argparse
import math
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent

# 기본 분류 대상/결과 경로. 필요하면 여기만 바꿔서 재사용할 수 있습니다.
SOURCE_FOOD_LIST_PATH = str(BASE_DIR / "food_list")
TRAIN_OUTPUT_PATH = str(BASE_DIR / "food_list_train.txt")
TEST_OUTPUT_PATH = str(BASE_DIR / "food_list_test.txt")
DOTENV_PATH = str(BASE_DIR / ".env")

# None으로 두면 매번 다른 무작위 분할이 됩니다.
RANDOM_SEED: int | None = 42
EMBEDDING_MODEL = "text-embedding-3-small"


def dedupe_foods(foods: list[str]) -> list[str]:
    return list(dict.fromkeys(food.strip() for food in foods if food.strip()))


def parse_food_list(path: str) -> list[str]:
    foods: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            foods.append(parts[1].strip() if len(parts) == 2 else parts[0].strip())
    return foods


def embed_all(texts: list[str], client: OpenAI) -> dict[str, list[float]]:
    unique = dedupe_foods(texts)
    if not unique:
        return {}
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=unique)
    return {text: item.embedding for text, item in zip(unique, response.data)}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("0 벡터는 코사인 유사도를 계산할 수 없습니다.")
    return dot / (norm_a * norm_b)


def split_foods_by_distance(
    foods: list[str],
    embeddings: dict[str, list[float]],
    seed: int | None = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    remaining = dedupe_foods(foods)
    missing = [food for food in remaining if food not in embeddings]
    if missing:
        raise ValueError(f"임베딩이 없는 음식이 있습니다: {', '.join(missing[:5])}")

    rng = random.Random(seed)
    train: list[str] = []
    test: list[str] = []

    while remaining:
        train_index = rng.randrange(len(remaining))
        train_food = remaining.pop(train_index)
        train.append(train_food)

        if not remaining:
            break

        train_embedding = embeddings[train_food]
        test_index, _ = min(
            enumerate(remaining),
            key=lambda item: cosine_similarity(train_embedding, embeddings[item[1]]),
        )
        test.append(remaining.pop(test_index))

    return train, test


def write_food_list(path: str, foods: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(foods) + "\n", encoding="utf-8")


def build_openai_client(dotenv_path: str) -> OpenAI:
    load_dotenv(dotenv_path)
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_KEY를 찾을 수 없습니다: {dotenv_path}")
    return OpenAI(api_key=api_key)


def run(
    source_path: str = SOURCE_FOOD_LIST_PATH,
    train_output_path: str = TRAIN_OUTPUT_PATH,
    test_output_path: str = TEST_OUTPUT_PATH,
    dotenv_path: str = DOTENV_PATH,
    seed: int | None = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    foods = dedupe_foods(parse_food_list(source_path))
    if not foods:
        raise ValueError(f"음식 목록이 비어 있습니다: {source_path}")

    print(f"[로드] 음식 목록: {source_path} ({len(foods)}개)")
    print(f"[임베딩] {EMBEDDING_MODEL} 모델로 음식명을 벡터화합니다.")

    client = build_openai_client(dotenv_path)
    embeddings = embed_all(foods, client)
    train, test = split_foods_by_distance(foods, embeddings, seed=seed)

    write_food_list(train_output_path, train)
    write_food_list(test_output_path, test)

    print(f"[저장] train: {train_output_path} ({len(train)}개)")
    print(f"[저장] test : {test_output_path} ({len(test)}개)")
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI 임베딩 기반 음식 train/test 분할 스크립트"
    )
    parser.add_argument("--input", default=SOURCE_FOOD_LIST_PATH, help="분류할 음식 목록 경로")
    parser.add_argument("--train-output", default=TRAIN_OUTPUT_PATH, help="train 목록 저장 경로")
    parser.add_argument("--test-output", default=TEST_OUTPUT_PATH, help="test 목록 저장 경로")
    parser.add_argument("--dotenv", default=DOTENV_PATH, help="OPENAI_KEY를 읽을 .env 경로")
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="무작위 train 선택 seed. 같은 seed면 같은 결과를 만듭니다.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="seed를 쓰지 않고 매번 다른 무작위 분할을 만듭니다.",
    )
    args = parser.parse_args()

    run(
        source_path=args.input,
        train_output_path=args.train_output,
        test_output_path=args.test_output,
        dotenv_path=args.dotenv,
        seed=None if args.random else args.seed,
    )


if __name__ == "__main__":
    main()
