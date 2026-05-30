from models.gemini import GeminiVLM
from models.openai_model import OpenAIVLM
from models.local_qwen import LocalQwenVLM

IMAGE_ROOT = (
    "/home/202421012/food/"
    "122.음식_분류를_위한_음식종류_및_양에_따른_칼로리_데이터셋(재료,_양념,_완제품_등)/"
    "01.데이터/2.Validation/원천데이터/image"
)

FOOD_LIST_TRAIN_PATH = "/home/202421012/food/finetune_prompt/food_list_train.txt"
FOOD_LIST_TEST_PATH  = "/home/202421012/food/finetune_prompt/food_list_test.txt"

PROMPT = "Say the name of the given food in Korean. Reply with the food name only, no extra text."


EVAL_SHOW_ES = False  # True 이면 outcome/predictions 에 ES 포함, False 이면 제외


def set_prompt(new_prompt: str):
    global PROMPT
    PROMPT = new_prompt


MODEL_REGISTRY = {
    # "qwen3-vl-8b":           lambda: LocalQwenVLM("Qwen/Qwen3-VL-8B-Instruct"),
    # "qwen3-vl-32b":          lambda: LocalQwenVLM("Qwen/Qwen3-VL-32B-Instruct"),
    "gemini-3.5-flash":      lambda: GeminiVLM("gemini-3.5-flash"),
    "gemini-3.1-flash-lite": lambda: GeminiVLM("gemini-3.1-flash-lite"),
    "gpt-5.4-mini":          lambda: OpenAIVLM("gpt-5.4-mini"),
}
