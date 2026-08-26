"""AdaptEval 评测指标（对齐 TLM 仓库 scripts/eval）。"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ReasoningBench vs Domain/InstructionBench（按 AdaptEval 子集文件名）
REASONING_SUBSET_KEYS = ("gsm8k", "logiqa", "metamath")


def bench_type_for_subset(subset_tag: str) -> str:
    s = subset_tag.lower()
    if any(k in s for k in REASONING_SUBSET_KEYS):
        return "reasoning"
    return "similarity"


def get_reference_label(row: Dict[str, Any]) -> str:
    if row.get("output") is not None:
        return str(row["output"]).strip()
    if row.get("answers") is not None:
        return str(row["answers"]).strip()
    if row.get("solution") is not None:
        return str(row["solution"]).strip()
    return ""


# --- 以下摘自 TLM scripts/eval/eval_utils.py（精简保留） ---


def _strip_thinking_blocks(text: str) -> str:
    """去掉 Qwen3.5 thinking 段，避免从中间推理步骤抽数字。"""
    if not text or text.strip().startswith("!"):
        return ""
    think_open = "<" + "think" + ">"
    think_close = "</" + "think" + ">"
    for pat in (
        re.escape(think_open) + r".*?" + re.escape(think_close),
        re.escape(think_open) + r".*",
    ):
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def _parse_numeric_token(s: str) -> Optional[str]:
    s = s.replace(",", "").strip().strip(".")
    if not s or not re.match(r"^-?\d*\.?\d*$", s):
        return None
    try:
        return str(round(float(s)))
    except ValueError:
        return None


def extract_gsm8k_answer_number(completion: str) -> Optional[str]:
    """从生成文本抽取 GSM8K 最终数字答案（对齐 TLM eval_utils，并适配 Qwen thinking）。"""
    if not completion or completion.strip().startswith("!"):
        return None

    text = _strip_thinking_blocks(completion)

    # GSM8K 常见 #### 答案行
    hash_nums = re.findall(r"####\s*(-?[\d,]+\.?\d*)", text)
    if hash_nums:
        return _parse_numeric_token(hash_nums[-1])

    for pat in (
        r"(?:the answer is|final answer is|answer is:?)\s*[#$\\]*\s*(-?[\d,]+\.?\d*)",
        r"\\boxed\{(-?[\d,]+\.?\d*)\}",
    ):
        ms = re.findall(pat, text, flags=re.IGNORECASE)
        if ms:
            parsed = _parse_numeric_token(ms[-1])
            if parsed is not None:
                return parsed

    # 优先在「Response」段之后取最后一个数（Alpaca 格式）
    parts = re.split(r"###\s*Response\s*:", text, flags=re.IGNORECASE)
    tail = parts[-1] if len(parts) > 1 else text

    tokens = re.split(r"[\s\n]+", tail)
    tokens_with_numbers = [t for t in tokens if re.search(r"-?\d", t)]
    cleaned_numbers = [re.sub(r"[^\d,\.-]", "", t) for t in tokens_with_numbers]
    if not cleaned_numbers:
        return None
    return _parse_numeric_token(cleaned_numbers[-1])


def extract_logiqa_option(completion: str) -> str:
    result = ""
    if re.match(r"^[A-D]\.\s", completion):
        result = completion[0]
    elif all(k in completion for k in ("Answer: ", "Explanation")):
        match = re.search(r"Answer:\s*(.*?)\s*Explanation", completion, re.DOTALL)
        tmp = match.group(1) if match else re.split(r"Answer: ", completion)[1]
        m = re.search(r"\b[A-D]\b", tmp)
        if m:
            result = m.group()
    elif "Explanation" in completion:
        tmp = re.split(r"Explanation", completion)[0]
        m = re.search(r"\b[A-D]\b", tmp)
        if m:
            result = m.group()
    elif any(k in completion for k in ("Answer: ", "answer is")):
        tmp = re.split(r"Answer:\s*|answer is\s*", completion, maxsplit=1)[1]
        m = re.search(r"\b[A-D]\b", tmp)
        if m:
            result = m.group()
    else:
        m = re.search(r"\b[A-D]\b", completion)
        if m:
            result = m.group()
    return result


def _post_process_math_string(string: str) -> str:
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("\\%", "").replace("%", "")
    return string.replace(" ", "")


def extract_math_answer(completion: str, label: str) -> Optional[str]:
    if any(key in label for key in ("\\text{", ":")):
        if "\\text{" in label:
            m = re.search(r"\\text{(.*?)}", label)
            content = m.group(1) if m else label
        else:
            content = label
        parts = re.split(r"[Tt]herefore|[Ss]o", completion)
        if len(parts) > 1 and content in parts[-1]:
            return label
        return None
    if label.startswith("("):
        m = re.search(r"\((.*?)", label)
        content = m.group(1) if m else label
        parts = re.split(r"[Tt]herefore|[Ss]o", completion)
        if len(parts) > 1 and content in parts[-1].replace(" ", ""):
            return label
        return None

    match = re.search(r"\\boxed\{(.*)\}", completion)
    if match:
        return _post_process_math_string(match.group(1))
    fracs = re.findall(r"\\frac\{\d\}\{\d\}", completion)
    if fracs:
        return _post_process_math_string(fracs[-1])
    return extract_gsm8k_answer_number(completion)


def extract_prediction(subset_tag: str, completion: str, label: str) -> Optional[str]:
    s = subset_tag.lower()
    if "logiqa" in s:
        return extract_logiqa_option(completion) or None
    if "gsm8k" in s:
        return extract_gsm8k_answer_number(completion)
    if "metamath" in s:
        return extract_math_answer(completion, label)
    return completion.strip()


def accuracy_score(
    subset_tag: str,
    predictions: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    extracted: List[Optional[str]] = []
    for pred, lab in zip(predictions, labels):
        extracted.append(extract_prediction(subset_tag, pred, lab))
    n = len(labels)
    valid = [(a, b) for a, b in zip(extracted, labels) if a is not None]
    if not valid:
        return {"accuracy": 0.0, "n": n, "n_valid_extractions": 0}
    cmp = [str(a).strip() == str(b).strip() for a, b in valid]
    return {
        "accuracy": sum(cmp) / len(cmp),
        "n": n,
        "n_valid_extractions": len(valid),
    }


def _pre_rouge_processing(summary: str) -> str:
    from nltk import sent_tokenize

    summary = summary.replace(" ", " ")
    return "\n".join(sent_tokenize(summary))


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True, split_summaries=True)
    sums = {k: 0.0 for k in rouge_types}
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, _pre_rouge_processing(pred))
        for k in rouge_types:
            sums[k] += scores[k].fmeasure
    return {k: sums[k] / n if n else 0.0 for k in rouge_types}


def bleu_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    smooth = SmoothingFunction().method4
    total = 0.0
    for pred, ref in zip(predictions, references):
        ref_tok = [nltk.word_tokenize(ref)]
        pred_tok = nltk.word_tokenize(pred)
        total += sentence_bleu(ref_tok, pred_tok, smoothing_function=smooth)
    return total / len(predictions) if predictions else 0.0


def evaluate_subset(
    subset_tag: str,
    predictions: Sequence[str],
    labels: Sequence[str],
    *,
    use_rouge: bool = True,
    use_bleu: bool = True,
) -> Dict[str, Any]:
    btype = bench_type_for_subset(subset_tag)
    out: Dict[str, Any] = {"subset": subset_tag, "bench_type": btype, "n": len(predictions)}
    if btype == "reasoning":
        out.update(accuracy_score(subset_tag, predictions, labels))
    else:
        if use_rouge:
            out.update(rouge_scores(predictions, labels))
        if use_bleu:
            out["bleu"] = bleu_score(predictions, labels)
    return out


def aggregate_metrics(per_subset: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    reasoning = [m for m in per_subset.values() if m.get("bench_type") == "reasoning"]
    similarity = [m for m in per_subset.values() if m.get("bench_type") == "similarity"]

    agg: Dict[str, Any] = {}
    if reasoning:
        n = sum(m.get("n", 0) for m in reasoning)
        acc_sum = sum(m.get("accuracy", 0.0) * m.get("n", 0) for m in reasoning)
        agg["reasoning_accuracy"] = acc_sum / n if n else 0.0
        agg["reasoning_n"] = n
    if similarity:
        for key in ("rouge1", "rouge2", "rougeL", "rougeLsum", "bleu"):
            vals = [m[key] for m in similarity if key in m]
            if vals:
                agg[f"similarity_{key}"] = sum(vals) / len(vals)
        agg["similarity_n_subsets"] = len(similarity)
    return agg
