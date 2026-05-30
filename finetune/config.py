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

TASK = "주어진 음식 이미지를 보고 음식명을 한국어로 맞추시오."

TRAJECTORY_PROMPT = (
    "You are analyzing a food image. "
    "Respond ONLY with a JSON object in this exact format (no markdown, no extra text):\n"
    "{\n"
    '  "think": "<describe what you observe in the image and why you identify it as a specific food>",\n'
    '  "action": "<state your prediction in Korean>",\n'
    '  "obs": "<food name in Korean only>"\n'
    "}"
)


EVAL_SHOW_ES = False  # True 이면 outcome 에 ES 포함, False 이면 제외


def set_task(new_task: str):
    global TASK
    TASK = new_task


def set_trajectory_prompt(new_prompt: str):
    global TRAJECTORY_PROMPT
    TRAJECTORY_PROMPT = new_prompt


# ── extract.py 설정 ──────────────────────────────────────────────────────────

EXTRACT_MODEL = "gemini-3.5-flash"

EXTRACT_K = 3

EXTRACT_PROMPT = (
    "You analyse a single agent trajectory and extract [success patterns | failure patterns] -\n"
    "high-level, reusable, transferable behavioural patterns. Focus on genuinely novel,\n"
    "reusable patterns from THIS trajectory; do NOT try to be exhaustive.\n"
    "What is a pattern? A pattern is a high-level behaviour that is (1) transferable\n"
    "across a broad class of tasks, (2) actionable enough to follow, (3) non-obvious\n"
    "- going beyond common sense, and (4) self-contained - understandable without the\n"
    "original trajectory.\n"
    "Per-type guidance (success). Capture effective strategies, decision patterns, and\n"
    "methodological insights. Ask: \"What did this agent do RIGHT that other agents\n"
    "facing similar tasks should also do?\"\n"
    "Per-type guidance (failure). Capture error patterns, anti-patterns, and non-obvious\n"
    "pitfalls. Ask: \"What should an agent AVOID doing when facing similar tasks?\"\n"
    "Quality requirements. Each pattern must be (i) high-level and domain-general, (ii)\n"
    "maximally broad in coverage, (iii) information-dense with a concrete description,\n"
    "and (iv) free of task-specific details (no specific file names, identifiers, error\n"
    "messages, or API calls).\n"
    "Constraints. Extract at most [{K}] patterns from this trajectory; each as a pattern\n"
    "name and a 2-4 sentence description.\n"
    "Output format. A JSON list of {{\"type\", \"pattern\", \"description\"}} entries. "
    "If no useful patterns are found, return an empty list.\n"
    "Output ONLY the JSON list with no markdown fences."
)

MERGE_H = 10

MERGE_PROMPT = (
    "You receive several pattern sets, each extracted from a different agent trajectory,\n"
    "and merge them into a single consolidated pattern set.\n"
    "Guidelines.\n"
    "1. Deduplicate: if multiple patterns describe the same or overlapping behaviour,\n"
    "combine them into ONE stronger pattern with the best description.\n"
    "2. Generalise: raise the abstraction level to cover more scenarios; a single\n"
    "well-generalised pattern is worth more than several narrow ones.\n"
    "3. Preserve type: keep success and failure patterns separate; do NOT convert\n"
    "between types.\n"
    "4. Preserve quality: drop vague or low-value patterns; keep concrete, actionable ones.\n"
    "5. Prioritise: when there are too many patterns, retain the most important and\n"
    "broadly applicable ones.\n"
    "Quality requirements. Each merged pattern must be transferable across tasks,\n"
    "information-dense, non-obvious, and free of task-specific details.\n"
    "Output format. A JSON list of {{\"type\", \"pattern\", \"description\"}} entries.\n"
    "Output ONLY the JSON list with no markdown fences."
)

SYNTHESIS_MAX_SKILLS = 1
SYNTHESIS_MAX_SKILL_CHARS = 1000

SYNTHESIS_PROMPT = (
    "You receive a consolidated set of success and failure patterns and synthesise them\n"
    "into skills.\n"
    "Synthesis strategy.\n"
    "1. Integrate both polarities: a good skill includes both what TO DO (from success\n"
    "patterns) and what to AVOID (from failure patterns).\n"
    "2. Organise thematically: group related patterns into coherent skills around\n"
    "shared themes.\n"
    "3. Structure the body clearly: recommended approaches, common pitfalls, decision\n"
    "criteria for when to apply, and verification methods.\n"
    "4. Maintain information density: every sentence carries actionable content; no platitudes.\n"
    "5. Keep the description short: 1-2 sentences only; all detail goes in the body.\n"
    "Schema requirements. name (lowercase-hyphen slug, <= 64 chars); description (1-2\n"
    "sentences: what class of problems, when to apply); body (Markdown with strategies,\n"
    "pitfalls, decision criteria, verification).\n"
    "Budget. Maximum [{max_skills}] skills, each <= [{max_skill_chars}] characters.\n"
    "Output format. A JSON list of {{\"name\", \"description\", \"body\"}} objects.\n"
    "Output ONLY the JSON list with no markdown fences."
)


MODEL_REGISTRY = {
    # "qwen3-vl-8b":           lambda: LocalQwenVLM("Qwen/Qwen3-VL-8B-Instruct"),
    # "qwen3-vl-32b":          lambda: LocalQwenVLM("Qwen/Qwen3-VL-32B-Instruct"),
    "gemini-3.5-flash":      lambda: GeminiVLM("gemini-3.5-flash"),
    "gemini-3.1-flash-lite": lambda: GeminiVLM("gemini-3.1-flash-lite"),
    "gpt-5.4-mini":          lambda: OpenAIVLM("gpt-5.4-mini"),
}
