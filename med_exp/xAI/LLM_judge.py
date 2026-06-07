"""
LLM-as-a-Judge: score baseline / gnnexp text explanations produced by
GNNExplainer_get_med_explanation_from_edge_mask.py.

Rubric (1-5 each):
  - Plausibility  : clinical correctness and medical coherence
  - Completeness  : self-sufficiency – all context needed is present
  - Compactness   : no unnecessary narrative or redundancy

For every idx the judge sees:
  note text  + ICD/procedure text  (grounding context)
  one explanation at a time        (independent scoring)

Output: scores.csv  (one row per explanation file)

Anti-Self-Enhancement-Bias Architecture
----------------------------------------
The **generator** and the **judge** are deliberately kept as *separate models*
loaded independently.  Using the same model to score its own outputs introduces
Self-Enhancement Bias — the model systematically over-scores text that matches
its own generation style.

Three judge tiers are supported; select via JUDGE_TIER below:

  Option A — Selene 1 Mini  (JUDGE_TIER = "selene")   ← RECOMMENDED
    AtlaAI/Selene-1-Mini-Llama-3.1-8B  (~16 GB bf16)
    State-of-the-art 8B LLM-as-a-judge, outperforms GPT-4o-mini and
    Prometheus 2 across absolute scoring, classification, and pairwise tasks.
    Generates a chain-of-thought critique then a bare integer score after
    [RESULT], giving well-calibrated scores across the full 1-5 range.

    Key design change vs Prometheus: each criterion is scored in a **separate**
    inference call.  This prevents the model from collapsing all three scores
    to the same extreme (the "only 1 or 5" problem).

  Option B — Specialized Evaluator  (JUDGE_TIER = "prometheus")
    prometheus-eval/prometheus-8x7b-v2.0   (~26 GB bf16)
    prometheus-eval/prometheus-7b-v2.0     (memory-constrained alternative)
    Note: known to produce polarised 1/5 scores on absolute scoring tasks.

  Option C — High-Capacity General / Reasoning  (JUDGE_TIER = "general")
    Qwen/Qwen2.5-72B-Instruct              (default; needs 2× A100)
    meta-llama/Llama-3.3-70B-Instruct      (alternative)

Set JUDGE_TIER and (optionally) JUDGE_MODEL_ID before running.
"""

import gc
import json
import os
import re
import sys
import csv
import pickle
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from med_exp.graph.med_graph_utils import getNodeText  # noqa: E402

explainer = "pg"  # gnn | pg

category     = "tp"
EXPL_ROOT    = os.path.join(
    "med_exp", "graph", "results", "explanations",
    f"hetero_{explainer}_explainer", "NoDupEdges",
)
CATEGORY_DIR = os.path.join(EXPL_ROOT, category)
TEXT_DIR     = os.path.join(CATEGORY_DIR, "text_explanations")
OUTPUT_CSV   = os.path.join(TEXT_DIR, f"{explainer}_scores.csv")


# ---------------------------------------------------------------------------
# Judge model configuration
# ---------------------------------------------------------------------------
# Select JUDGE_TIER:
#   "selene"     → Option A: Selene 1 Mini (recommended — best calibration)
#   "prometheus" → Option B: specialized evaluator (prometheus-eval/*)
#   "general"    → Option C: high-capacity general model (Qwen / Llama)
#
# Override JUDGE_MODEL_ID to pin a specific checkpoint within the chosen tier.
# Leave as None to use the tier default.
# ---------------------------------------------------------------------------
JUDGE_TIER     = "flow"    # "selene" | "prometheus" | "general"
JUDGE_MODEL_ID = None        # None → use tier default below

hf_token = os.getenv("HF_TOKEN")


@dataclass
class JudgeConfig:
    """Per-model settings that affect prompt construction and generation."""
    model_id: str
    # Whether the model's chat template accepts a "system" role.
    supports_system_role: bool
    # "selene"     → critique + bare integer after [RESULT], one criterion/call
    # "prometheus" → JSON block in a single call
    # "json"       → plain JSON in a single call
    output_format: str
    # Safe token budget for combined context fields.
    max_context_tokens: int
    # Hard cap on generated tokens.
    max_new_tokens: int


_TIER_DEFAULTS: dict[str, JudgeConfig] = {
    "flow": JudgeConfig(
        model_id="flowaicom/Flow-Judge-v0.1",
        supports_system_role=True,
        output_format="flow",
        max_context_tokens=3072,
        max_new_tokens=512,   # feedback prose + <score> tag
    ),
    "selene": JudgeConfig(
        model_id="AtlaAI/Selene-1-Mini-Llama-3.1-8B",
        supports_system_role=True,
        output_format="selene",
        max_context_tokens=3072,
        max_new_tokens=512,   # critique + score; Selene reasons before scoring
    ),
    "prometheus": JudgeConfig(
        model_id="prometheus-eval/prometheus-8x7b-v2.0",
        supports_system_role=False,
        output_format="prometheus",
        max_context_tokens=3072,
        max_new_tokens=128,
    ),
    "general": JudgeConfig(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        supports_system_role=True,
        output_format="json",
        max_context_tokens=4096,
        max_new_tokens=64,
    ),
}

if JUDGE_TIER not in _TIER_DEFAULTS:
    raise ValueError(
        f"JUDGE_TIER must be 'flow', 'selene', 'prometheus', or 'general', got {JUDGE_TIER!r}"
    )

judge_cfg = _TIER_DEFAULTS[JUDGE_TIER]
if JUDGE_MODEL_ID is not None:
    judge_cfg.model_id = JUDGE_MODEL_ID

print(
    f"[Judge] Tier: {JUDGE_TIER!r} | Model: {judge_cfg.model_id}\n"
    f"[Judge] Output format: {judge_cfg.output_format} | "
    f"System role: {judge_cfg.supports_system_role}\n"
)

judge_tokenizer = AutoTokenizer.from_pretrained(
    judge_cfg.model_id, use_fast=True, token=hf_token
)
judge_model = AutoModelForCausalLM.from_pretrained(
    judge_cfg.model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=hf_token,
)
if judge_tokenizer.pad_token_id is None:
    judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id

MAX_CONTEXT_TOKENS = judge_cfg.max_context_tokens
MAX_NEW_TOKENS     = judge_cfg.max_new_tokens


def clear_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens using the judge tokenizer."""
    ids = judge_tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return judge_tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _generate_judge(messages: list[dict], do_sample: bool = False) -> str:
    """
    Run one forward pass through the **judge** model.

    For Selene we use do_sample=True with temperature 0.3 so the critique is
    coherent but the score is still calibrated (greedy can get anchored to
    extremes on short responses).
    For Prometheus and general models we use greedy decoding (do_sample=False).
    """
    prompt_str = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    encoded        = judge_tokenizer(prompt_str, return_tensors="pt", padding=True,
                                     return_attention_mask=True)
    input_ids      = encoded["input_ids"].to(judge_model.device)
    attention_mask = encoded["attention_mask"].to(judge_model.device)
    del encoded

    prompt_len = input_ids.shape[1]

    terminators = [judge_tokenizer.eos_token_id]
    for special in ("<|eot_id|>", "<|end_of_turn|>", "<|im_end|>"):
        tok_id = judge_tokenizer.convert_tokens_to_ids(special)
        if tok_id is not None and tok_id != judge_tokenizer.unk_token_id:
            terminators.append(tok_id)
    terminators = list(dict.fromkeys(t for t in terminators if t is not None))

    gen_kwargs: dict = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=terminators,
        pad_token_id=judge_tokenizer.pad_token_id,
        attention_mask=attention_mask,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=0.3, top_p=0.95)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = judge_model.generate(input_ids, **gen_kwargs)

    del input_ids, attention_mask

    generated_ids = outputs[0, prompt_len:]
    del outputs

    response = judge_tokenizer.decode(generated_ids, skip_special_tokens=True)
    del generated_ids

    clear_gpu_cache()
    return response


# ---------------------------------------------------------------------------
# Prompt builders — Selene
# ---------------------------------------------------------------------------
# Selene is trained on a format where:
#   system = evaluator role instructions
#   user   = [Instruction] block + [Response] block + [Scoring Rubric] block
#   model  = chain-of-thought critique ending with "[RESULT] <integer>"
#
# We score each of the three criteria in a **separate call** to avoid the
# "all scores collapse to 1 or 5" failure mode that comes from asking a model
# to produce multiple constrained values in one response.

_SELENE_SYSTEM = (
    "You are a fair and objective evaluator assessing the quality of "
    "AI-generated medical necessity arguments. "
    "You will be given an instruction (a patient note and the medical procedures "
    "requested) and a response (the argument to evaluate). "
    "Your task is to score the response on a single criterion using the "
    "1–5 rubric provided. "
    "First write a concise critique (2–4 sentences), then on a new line "
    "write exactly: [RESULT] <integer>"
)

_SELENE_RUBRICS: dict[str, str] = {
    "plausibility": (
        "Plausibility — clinical correctness and medical coherence.\n"
        "1 = Clinically wrong or medically incoherent.\n"
        "2 = Mostly incorrect; only minor correct elements.\n"
        "3 = Partially correct; some notable clinical errors.\n"
        "4 = Mostly correct with only minor gaps.\n"
        "5 = Clinically accurate and medically sound throughout."
    ),
    "completeness": (
        "Completeness — self-sufficiency; all context needed is present.\n"
        "1 = Almost no useful context; response cannot stand alone.\n"
        "2 = Major context missing; reader would be confused.\n"
        "3 = Some key context present but important gaps remain.\n"
        "4 = Nearly self-sufficient; only minor omissions.\n"
        "5 = Fully self-contained; no additional context needed."
    ),
    "compactness": (
        "Compactness — conciseness; no unnecessary repetition.\n"
        "1 = Extremely verbose; filled with repetition and filler.\n"
        "2 = Noticeably wordy; significant redundancy.\n"
        "3 = Moderate length; some unnecessary repetition.\n"
        "4 = Mostly concise; minor unnecessary passages.\n"
        "5 = Very concise; every sentence adds distinct value."
    ),
}


def _build_selene_messages(
    note_txt: str, icd_txt: str, explanation: str, criterion: str
) -> list[dict]:
    """Return a message list to score one criterion with Selene."""
    budget = MAX_CONTEXT_TOKENS // 3
    note_txt    = _truncate_to_tokens(note_txt,    budget)
    icd_txt     = _truncate_to_tokens(icd_txt,     budget)
    explanation = _truncate_to_tokens(explanation, budget)

    user_content = (
        "[Instruction]\n"
        f"## Patient Note\n{note_txt.strip()}\n\n"
        f"## Medical Procedures (ICD)\n{icd_txt.strip()}\n\n"
        "[Response]\n"
        f"{explanation.strip()}\n\n"
        "[Scoring Rubric]\n"
        f"{_SELENE_RUBRICS[criterion]}\n\n"
        "Write your critique, then end with:\n[RESULT] <integer 1-5>"
    )
    return [
        {"role": "system", "content": _SELENE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


_FLOW_RUBRICS_MED: dict[str, tuple[str, str]] = {
    "plausibility": (
        "Plausibility of the medical necessity argument on a 5-point Likert scale: "
        "how clinically correct and medically coherent is the explanation?",
        "Score 1: The argument is clinically wrong or medically incoherent.\n"
        "Score 2: The argument is mostly incorrect with only minor correct elements.\n"
        "Score 3: The argument is partially correct but contains notable clinical errors.\n"
        "Score 4: The argument is mostly correct with only minor gaps.\n"
        "Score 5: The argument is clinically accurate and medically sound throughout.",
    ),
    "completeness": (
        "Completeness of the medical necessity argument on a 5-point Likert scale: "
        "is the explanation self-sufficient with all necessary context present?",
        "Score 1: Almost no useful context; the response cannot stand alone.\n"
        "Score 2: Major context is missing; a reader would be confused.\n"
        "Score 3: Some key context is present but important gaps remain.\n"
        "Score 4: Nearly self-sufficient; only minor omissions.\n"
        "Score 5: Fully self-contained; no additional context is needed.",
    ),
    "compactness": (
        "Compactness of the medical necessity argument on a 5-point Likert scale: "
        "how concise is the explanation with no unnecessary repetition?",
        "Score 1: Extremely verbose; filled with repetition and irrelevant filler.\n"
        "Score 2: Noticeably wordy; significant redundancy present.\n"
        "Score 3: Moderate length; some unnecessary repetition.\n"
        "Score 4: Mostly concise; only minor unnecessary passages.\n"
        "Score 5: Very concise; every sentence adds distinct value.",
    ),
}


def _build_flow_messages_med(
    note_txt: str, icd_txt: str, explanation: str, criterion: str
) -> list[dict]:
    """Build a Flow-Judge message list for one medical criterion."""
    budget = MAX_CONTEXT_TOKENS // 3
    note_txt    = _truncate_to_tokens(note_txt,    budget)
    icd_txt     = _truncate_to_tokens(icd_txt,     budget)
    explanation = _truncate_to_tokens(explanation, budget)

    eval_criteria, rubric = _FLOW_RUBRICS_MED[criterion]

    user_content = (
        "# GOAL\n"
        "Your job is to evaluate a task carried out by an AI system powered by a "
        "large language model.\n"
        "You will be provided with the inputs and output of the task, as well as the "
        "evaluation criteria and scoring rubric. Your task is to evaluate the output "
        "of the AI system based on the evaluation criteria and scoring rubric provided.\n\n"
        "# INPUT\n"
        "Below are the inputs required for performing the task:\n"
        "<inputs>\n"
        f"<patient_note>\n{note_txt.strip()}\n</patient_note>\n"
        f"<medical_procedures_icd>\n{icd_txt.strip()}\n</medical_procedures_icd>\n"
        "</inputs>\n\n"
        "# OUTPUT\n"
        "Below is the output of the task:\n"
        f"<output>\n{explanation.strip()}\n</output>\n\n"
        "# EVALUATION CRITERIA AND SCORING RUBRIC\n"
        "Here are the evaluation criteria and the rubric that you need to use for "
        "evaluating the task:\n"
        f"<evaluation_criteria>\n{eval_criteria}\n</evaluation_criteria>\n\n"
        f"<scoring_rubric>\n{rubric}\n</scoring_rubric>\n\n"
        "# INSTRUCTIONS FOR THE EVALUATION\n"
        "1. Understand the task and criteria: Review the evaluation criteria and the "
        "5-point Likert scale scoring rubric.\n"
        "2. Review the inputs and output.\n"
        "3. Compare the output to each score description in the rubric and decide "
        "which best matches.\n"
        "4. Pay attention to small details that might shift the score up or down.\n"
        "5. Write verbal feedback justifying your evaluation.\n"
        "6. Assign a final score from 1 to 5 based on the Likert scale rubric.\n\n"
        "## FORMAT FOR THE EVALUATION\n"
        "- Write the verbal feedback inside <feedback> tags.\n"
        "- Write the numeric score (1-5) inside <score> tags, always after the feedback.\n\n"
        "Please accurately evaluate the task. Strictly adhere to the evaluation criteria "
        "and rubric."
    )
    return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# Prompt builders — Prometheus
# ---------------------------------------------------------------------------

_RUBRIC_PROMETHEUS = """\
[System]
You are a fair and objective evaluator that assesses the quality of AI-generated medical necessity explanations.

[Task Description]
An instruction (comprising a patient note and a medical procedure/ICD code) will be given along with a response (the explanation to score).
Your task is to evaluate the response by scoring it on exactly three dimensions and returning a JSON object.

<rubric>
  plausibility  (1-5): clinical correctness and medical coherence
  completeness  (1-5): self-sufficiency; all context needed is present
  compactness   (1-5): conciseness; no unnecessary repetition (5 = very concise)
</rubric>

Reply with ONLY a JSON object, no other text, no markdown:
{"plausibility": <int>, "completeness": <int>, "compactness": <int>}
[RESULT]"""


def _build_messages_prometheus(
    note_txt: str, icd_txt: str, explanation: str
) -> list[dict]:
    instruction = (
        "## Patient Note\n" + note_txt.strip() + "\n\n"
        "## Medical Procedures (ICD)\n" + icd_txt.strip()
    )
    user_content = (
        _RUBRIC_PROMETHEUS + "\n\n"
        "[Instruction]\n" + instruction + "\n\n"
        "<response>" + explanation.strip() + "</response>"
    )
    return [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": "{"},
    ]


# ---------------------------------------------------------------------------
# Prompt builders — General (JSON)
# ---------------------------------------------------------------------------

_RUBRIC_JSON = (
    "Score the following medical necessity explanation on exactly three criteria "
    "(integer 1-5 each):\n"
    "  plausibility  – clinical correctness and medical coherence\n"
    "  completeness  – self-sufficiency; all context needed is present\n"
    "  compactness   – conciseness; no unnecessary repetition (5=very concise)\n\n"
    "Reply with ONLY a JSON object, no other text, no markdown:\n"
    '{"plausibility": <int>, "completeness": <int>, "compactness": <int>}'
)


def _build_messages_general(system_content: str, user_content: str) -> list[dict]:
    messages: list[dict] = []
    if judge_cfg.supports_system_role:
        messages.append({"role": "system", "content": system_content})
    else:
        user_content = system_content + "\n\n" + user_content
    messages.append({"role": "user",      "content": user_content})
    messages.append({"role": "assistant", "content": "{"})
    return messages


def _build_messages_general_judge(
    note_txt: str, icd_txt: str, explanation: str
) -> list[dict]:
    budget_each = MAX_CONTEXT_TOKENS // 3
    user_content = (
        "## Patient Note\n" + _truncate_to_tokens(note_txt, budget_each).strip() + "\n\n"
        "## Medical Procedures (ICD)\n" + _truncate_to_tokens(icd_txt, budget_each).strip() + "\n\n"
        "## Explanation\n" + _truncate_to_tokens(explanation, budget_each).strip()
    )
    return _build_messages_general(
        system_content="You are an expert medical AI evaluator.\n\n" + _RUBRIC_JSON,
        user_content=user_content,
    )


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def _extract_selene_score(raw: str) -> int | None:
    """
    Extract the integer score from Selene's output.

    Selene outputs a chain-of-thought critique ending with e.g.:
        [RESULT] 3
    We try multiple patterns in descending order of confidence.
    """
    # Strategy 1: explicit [RESULT] tag
    m = re.search(r'\[RESULT\]\s*(\d)', raw, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score

    # Strategy 2: "Score: N" or "score of N" or "rating: N"
    m = re.search(r'(?:score|rating)\s*(?:of|:)?\s*(\d)', raw, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score

    # Strategy 3: last standalone 1–5 digit in the response
    digits = re.findall(r'\b([1-5])\b', raw)
    if digits:
        return int(digits[-1])

    return None


def _extract_flow_score(raw: str) -> int | None:
    """
    Extract the integer score from Flow-Judge's output.
    Flow-Judge outputs: <feedback>...</feedback>\n<score>N</score>
    """
    # Strategy 1: <score> tag
    m = re.search(r'<score>\s*(\d)\s*</score>', raw, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score

    # Strategy 2: bare <score>N (unclosed tag, truncated output)
    m = re.search(r'<score>\s*(\d)', raw, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score

    # Strategy 3: last standalone 1-5 digit
    digits = re.findall(r'\b([1-5])\b', raw)
    if digits:
        return int(digits[-1])

    return None


_KV_RE      = re.compile(r'"(\w+)"\s*:\s*"?(\d)"?')
_SCORE_KEYS = {"plausibility", "completeness", "compactness"}


def _obj_to_scores(obj: dict) -> dict | None:
    try:
        return {
            "plausibility": int(obj["plausibility"]),
            "completeness": int(obj["completeness"]),
            "compactness":  int(obj["compactness"]),
        }
    except (KeyError, ValueError, TypeError):
        return None


def _extract_scores_json(raw: str) -> dict | None:
    """Multi-strategy extraction for JSON-format outputs (Prometheus / general)."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned
    if cleaned.count("{") > cleaned.count("}"):
        cleaned += "}"

    for blob in re.finditer(r'\{[^{}]*\}', cleaned, re.DOTALL):
        try:
            result = _obj_to_scores(json.loads(blob.group()))
            if result:
                return result
        except Exception:
            pass

    try:
        result = _obj_to_scores(json.loads(cleaned))
        if result:
            return result
    except Exception:
        pass

    found: dict[str, int] = {}
    for key, val in _KV_RE.findall(cleaned):
        if key in _SCORE_KEYS:
            found[key] = int(val)
    if found.keys() == _SCORE_KEYS:
        return dict(found)

    m = re.compile(
        r'"plausibility"\s*:\s*"?(?P<p>\d)"?'
        r'.*?"completeness"\s*:\s*"?(?P<c>\d)"?'
        r'.*?"compactness"\s*:\s*"?(?P<k>\d)"?',
        re.DOTALL,
    ).search(cleaned)
    if m:
        return {
            "plausibility": int(m.group("p")),
            "completeness": int(m.group("c")),
            "compactness":  int(m.group("k")),
        }

    return None


# ---------------------------------------------------------------------------
# Scoring entry points
# ---------------------------------------------------------------------------

def _score_with_selene(
    explanation: str, note_txt: str, icd_txt: str, max_retries: int
) -> dict | None:
    """
    Score via Selene: **three independent criterion calls**.

    Running one criterion per LLM call avoids the compression-to-extremes
    effect seen when a model tries to emit three constrained integer scores in
    one pass.  Each call can reason freely about a single criterion and land
    anywhere on the 1–5 scale.
    """
    scores: dict[str, int] = {}
    for criterion in ("plausibility", "completeness", "compactness"):
        score = None
        for attempt in range(1, max_retries + 1):
            messages = _build_selene_messages(note_txt, icd_txt, explanation, criterion)
            raw      = _generate_judge(messages, do_sample=True)
            score    = _extract_selene_score(raw)
            if score is not None:
                if attempt > 1:
                    print(f"  [INFO] {criterion} recovered on attempt {attempt}: {score}")
                break
            print(
                f"  [WARN] {criterion} attempt {attempt}/{max_retries}: "
                f"could not parse score from: {raw[-200:]!r}"
            )
        if score is None:
            print(
                f"  [ERROR] Could not extract '{criterion}' score after "
                f"{max_retries} attempts."
            )
            return None
        scores[criterion] = score
    return scores


def _score_with_flow(
    explanation: str, note_txt: str, icd_txt: str, max_retries: int
) -> dict | None:
    """Score via Flow-Judge: three independent criterion calls."""
    scores: dict[str, int] = {}
    for criterion in ("plausibility", "completeness", "compactness"):
        score = None
        for attempt in range(1, max_retries + 1):
            messages = _build_flow_messages_med(note_txt, icd_txt, explanation, criterion)
            raw      = _generate_judge(messages, do_sample=False)
            score    = _extract_flow_score(raw)
            if score is not None:
                if attempt > 1:
                    print(f"  [INFO] {criterion} recovered on attempt {attempt}: {score}")
                break
            print(
                f"  [WARN] {criterion} attempt {attempt}/{max_retries}: "
                f"could not parse score from: {raw[-200:]!r}"
            )
        if score is None:
            print(f"  [ERROR] Could not extract '{criterion}' score after {max_retries} attempts.")
            return None
        scores[criterion] = score
    return scores


def score_file(
    filepath: str,
    note_txt: str,
    icd_txt: str,
    max_retries: int = 5,
) -> dict | None:
    """
    Load explanation text, call the judge, return score dict.

    Returns None only when all retries are exhausted (the file is not written
    to CSV, so it will be retried on the next run).
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        explanation = fh.read().strip()

    if not explanation:
        print(f"  [WARN] Empty file: {filepath}")
        return None

    # ── Selene path: three separate criterion calls ──────────────────────────
    if judge_cfg.output_format == "selene":
        return _score_with_selene(explanation, note_txt, icd_txt, max_retries)

    if judge_cfg.output_format == "flow":
        return _score_with_flow(explanation, note_txt, icd_txt, max_retries)

    # ── Prometheus / general path: single JSON call with repair retries ──────
    last_raw = ""
    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            if judge_cfg.output_format == "prometheus":
                messages = _build_messages_prometheus(note_txt, icd_txt, explanation)
            else:
                messages = _build_messages_general_judge(note_txt, icd_txt, explanation)
        else:
            repair_suffix = (
                "\n\n## Previous (invalid) output\n"
                f"{last_raw.strip()}\n\n"
                "The previous output was not valid JSON. "
                "Return ONLY the corrected JSON object with keys "
                "plausibility, completeness, compactness (integers 1-5). "
                "No other text."
            )
            if judge_cfg.output_format == "prometheus":
                instruction = (
                    "## Patient Note\n" + note_txt.strip() + "\n\n"
                    "## Medical Procedures (ICD)\n" + icd_txt.strip()
                )
                user_content = (
                    _RUBRIC_PROMETHEUS + "\n\n"
                    "[Instruction]\n" + instruction + "\n\n"
                    "<response>" + explanation.strip() + "</response>"
                    + repair_suffix
                )
                messages = [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": "{"},
                ]
            else:
                budget_each = MAX_CONTEXT_TOKENS // 4
                user_content = (
                    "## Patient Note\n" + _truncate_to_tokens(note_txt, budget_each).strip() + "\n\n"
                    "## Medical Procedures (ICD)\n" + _truncate_to_tokens(icd_txt, budget_each).strip() + "\n\n"
                    "## Explanation\n" + _truncate_to_tokens(explanation, budget_each).strip()
                    + repair_suffix
                )
                messages = _build_messages_general(
                    system_content="You are an expert medical AI evaluator.\n\n" + _RUBRIC_JSON,
                    user_content=user_content,
                )

        last_raw = _generate_judge(messages, do_sample=False)
        raw_for_extraction = "{" + last_raw if not last_raw.startswith("{") else last_raw
        scores = _extract_scores_json(raw_for_extraction)
        if scores is not None:
            if attempt > 1:
                print(f"  [INFO] Recovered scores on attempt {attempt}: {scores}")
            return scores

        print(
            f"  [WARN] Attempt {attempt}/{max_retries}: "
            f"failed to parse JSON from: {last_raw!r}"
        )

    print(f"  [ERROR] Exhausted {max_retries} attempts for {filepath}. Will retry next run.")
    return None


# ---------------------------------------------------------------------------
# Index discovery
# ---------------------------------------------------------------------------

def discover_indices(text_dir: str) -> list[int]:
    indices = set()
    for fname in os.listdir(text_dir):
        m = re.match(r"(?:baseline|gnnexp|pgexp)(\d+)\.txt$", fname)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    indices = discover_indices(TEXT_DIR)
    if not indices:
        print(f"No explanation files found in {TEXT_DIR}. Exiting.")
        return

    print(f"Found {len(indices)} indices: {indices}")

    completed: set[str] = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if all(row.get(k, "NA") not in ("", "NA") for k in
                       ("plausibility", "completeness", "compactness")):
                    completed.add(row["filename"])
        print(f"Resuming – {len(completed)} files already successfully scored.")

    fieldnames = ["idx", "variant", "filename", "judge_model",
                  "plausibility", "completeness", "compactness"]

    write_header = not os.path.exists(OUTPUT_CSV)
    csv_fh  = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer  = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    if explainer == "gnn":
        VARIANTS = ["baseline", "gnnexp"]
    elif explainer == "pg":
        VARIANTS = ["pgexp"]
    else:
        VARIANTS = []

    pickle_filename = os.path.join(
        "med_exp", "graph", "results", "explanations",
        "hetero_gnn_explainer", "NoDupEdges", category,
        f"gnnexp_gnn_hier_med_graph_3GATConv_pred_edge_to_comp_g_edge_mask_{category}.pkl",
    )

    try:
        for idx in tqdm(indices, desc="Indices"):
            note_txt = ""
            icd_txt  = ""
            try:
                edge_masks_raw = pickle.load(open(pickle_filename, "rb"))

                link_idx = 0
                links_to_skip: list[int] = []

                for nodes, _ in edge_masks_raw.items():
                    link_idx += 1
                    if link_idx > 100:
                        break
                    if link_idx in links_to_skip:
                        continue
                    if link_idx != idx:
                        continue

                    src_ntype = nodes[0][0]
                    src_nid   = nodes[0][1].item()
                    tgt_ntype = nodes[1][0]
                    tgt_nid   = nodes[1][1].item()

                    note_txt = getNodeText(src_nid, src_ntype)
                    icd_txt  = getNodeText(tgt_nid, tgt_ntype)
                    break

                del edge_masks_raw
                clear_gpu_cache()

            except Exception as exc:
                print(f"  [WARN] Could not load note/ICD text for idx={idx}: {exc}")
                note_txt = "[Patient note unavailable]"
                icd_txt  = "[Procedure text unavailable]"

            for variant in VARIANTS:
                fpath = os.path.join(TEXT_DIR, f"{variant}{idx}.txt")
                fname = os.path.basename(fpath)

                if fname in completed:
                    continue
                if not os.path.exists(fpath):
                    print(f"  [SKIP] Missing: {fpath}")
                    continue

                print(f"  Scoring {fname} …")
                scores = score_file(fpath, note_txt, icd_txt)

                if scores is not None:
                    row = {"idx": idx, "variant": variant, "filename": fname,
                           "judge_model": judge_cfg.model_id, **scores}
                    writer.writerow(row)
                    csv_fh.flush()
                else:
                    print(f"  [SKIP-WRITE] {fname} — no valid scores; will retry next run.")

    finally:
        csv_fh.close()

    print(f"\nDone. Scores written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()