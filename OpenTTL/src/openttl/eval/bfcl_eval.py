"""BFCL 轻量评测：解析模型输出与官方 ground_truth AST，做函数调用结构匹配。

覆盖类别：
- AST 类（simple/multiple/parallel/parallel_multiple/live_*，java/javascript 宽松解析）
- irrelevance/live_irrelevance（不应发起调用）、relevance/live_relevance（应发起调用）
- exec_*/rest/sql/multi_turn/chatable 需执行环境或多轮交互，仅生成预测，返回 None。

与官方 gorilla-bfcl 的 AST eval 为近似对齐（官方还处理参数默认值、类型强转等细节），
如需与 Leaderboard 完全一致的分数，请用官方 berkeley-function-call-leaderboard 包
对本脚本产出的 result jsonl 做评测。
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

from openttl.data.bfcl import (
    BFCL_AST_CATEGORIES,
    BFCL_EXEC_CATEGORIES,
    BFCL_IRRELEVANCE_CATEGORIES,
    BFCL_MULTI_TURN_CATEGORIES,
    BFCL_RELEVANCE_CATEGORIES,
)

Call = Dict[str, Any]  # {"name": str, "arguments": dict, "positional": list}


# ---------------------------------------------------------------- 值归一化
def _norm_value(v: Any) -> Any:
    """归一化参数值以便跨类型比较（数字转 float、递归处理 list/dict）。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return v.strip()
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [_norm_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k).strip(): _norm_value(x) for k, x in v.items()}
    return v


def _values_equal(a: Any, b: Any) -> bool:
    na, nb = _norm_value(a), _norm_value(b)
    if isinstance(na, float) and isinstance(nb, float):
        return na == nb
    if isinstance(na, (list, dict)) != isinstance(nb, (list, dict)):
        # 类型不同但字符串化后一致（如 "3" vs 3 不做等价，仅 list/dict 直接比）
        return str(a).strip() == str(b).strip()
    return na == nb


# ---------------------------------------------------------------- 模型输出解析
def _extract_json_block(text: str) -> Optional[str]:
    """提取第一个 ``` 代码块内容；无代码块时返回 None。"""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else None


def _first_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    i = text.find(open_ch)
    while i != -1:
        depth = 0
        for j in range(i, len(text)):
            c = text[j]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
        i = text.find(open_ch, i + 1)
    return None


def _coerce_call_object(obj: Any) -> Optional[Call]:
    """将 {"name": ..., "arguments": {...}} / {"func": ...} / {"name": args} 归一为 Call。"""
    if not isinstance(obj, dict) or not obj:
        return None
    name = None
    for k in ("name", "function", "func", "tool", "function_name", "tool_name"):
        if k in obj and isinstance(obj[k], str):
            name = obj[k]
            break
    if name is not None:
        args = None
        for k in ("arguments", "parameters", "args", "params"):
            if k in obj and isinstance(obj[k], dict):
                args = obj[k]
                break
        return {"name": name.strip(), "arguments": args or {}, "positional": []}
    # {func_name: {args}} 形式
    if len(obj) == 1:
        (name, args), = obj.items()
        if isinstance(args, dict):
            return {"name": str(name).strip(), "arguments": args, "positional": []}
    return None


def _parse_py_call_expr(expr: ast.expr) -> Optional[Call]:
    if not isinstance(expr, ast.Call):
        return None
    fn = expr.func
    if isinstance(fn, ast.Name):
        name = fn.id
    elif isinstance(fn, ast.Attribute):
        try:
            name = ast.unparse(fn)
        except Exception:
            return None
    else:
        return None
    kwargs: Dict[str, Any] = {}
    for kw in expr.keywords:
        if kw.arg is None:
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            kwargs[kw.arg] = ast.unparse(kw.value)
    positional = []
    for a in expr.args:
        try:
            positional.append(ast.literal_eval(a))
        except Exception:
            positional.append(ast.unparse(a))
    return {"name": name, "arguments": kwargs, "positional": positional}


_CALL_RE = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*\s*\(")


def _parse_python_style_calls(text: str) -> List[Call]:
    """逐段解析 name(arg=..., ...) 形式的调用（兼容 java/javascript）。"""
    calls: List[Call] = []
    for m in _CALL_RE.finditer(text):
        name = m.group(0)[:-1].strip()
        # 平衡括号截取参数串
        depth, start = 0, m.end() - 1
        end = None
        for j in range(m.end() - 1, len(text)):
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            continue
        arg_str = text[start + 1 : end].strip()
        kwargs: Dict[str, Any] = {}
        positional: List[Any] = []
        if arg_str:
            for piece in _split_top_level(arg_str):
                if "=" in piece and not piece.lstrip().startswith(("'", '"', "[", "{")):
                    k, v = piece.split("=", 1)
                    if re.fullmatch(r"[A-Za-z_]\w*", k.strip()):
                        kwargs[k.strip()] = _loose_literal(v.strip())
                        continue
                positional.append(_loose_literal(piece.strip()))
        calls.append({"name": name, "arguments": kwargs, "positional": positional})
    return calls


def _split_top_level(s: str) -> List[str]:
    parts, depth, cur, quote = [], 0, [], None
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur and "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _loose_literal(s: str) -> Any:
    s = s.strip()
    if not s:
        return None
    low = s.lower()
    if low in ("null", "none", "undefined"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def parse_model_calls(text: str) -> List[Call]:
    """从模型自由文本中尽力提取函数调用列表；无法提取返回 []。"""
    if not text or not text.strip():
        return []
    candidates = []
    block = _extract_json_block(text)
    candidates.append(block if block is not None else text)
    candidates.append(text)

    for cand in candidates:
        # 1) JSON 数组 / 对象
        for snippet in (cand, _first_balanced(cand, "[", "]"), _first_balanced(cand, "{", "}")):
            if not snippet:
                continue
            try:
                obj = json.loads(snippet)
            except Exception:
                continue
            if isinstance(obj, list):
                calls = [c for c in (_coerce_call_object(o) for o in obj) if c]
                if calls or not obj:
                    return calls
            elif isinstance(obj, dict):
                c = _coerce_call_object(obj)
                if c:
                    return [c]
        # 2) Python/Java/JS 风格调用
        calls = _parse_python_style_calls(cand)
        if calls:
            return calls
    return []


# ---------------------------------------------------------------- ground truth 解析
def _parse_gt_python(ast_str: str) -> Optional[List[Call]]:
    s = ast_str.strip()
    if not s.startswith("["):
        s = "[" + s + "]"
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None
    expr = tree.body
    if not isinstance(expr, ast.List):
        return None
    calls = [_parse_py_call_expr(e) for e in expr.elts]
    if any(c is None for c in calls):
        return None
    return calls  # type: ignore[return-value]


def parse_ground_truth(gt: Any, language: str = "python") -> List[List[Call]]:
    """ground_truth 为多个候选 AST 字符串；返回每个候选解析出的调用列表。"""
    if isinstance(gt, str):
        gt = [gt]
    out: List[List[Call]] = []
    for item in gt or []:
        item = str(item)
        calls = None
        if language in ("python",):
            calls = _parse_gt_python(item)
        if calls is None:
            calls = _parse_python_style_calls(item)
        if calls:
            out.append(calls)
    return out


# ---------------------------------------------------------------- 匹配
def _args_match(pred: Call, gold: Call) -> bool:
    pk, gk = pred["arguments"], gold["arguments"]
    if gold["positional"] and not gold["arguments"]:
        # gold 为位置参数：优先与 pred 位置参数比，其次按值多重集宽松比
        if pred["positional"]:
            if len(pred["positional"]) != len(gold["positional"]):
                return False
            return all(
                _values_equal(a, b) for a, b in zip(pred["positional"], gold["positional"])
            )
        if len(pk) != len(gold["positional"]):
            return False
        pred_vals = sorted(str(_norm_value(v)) for v in pk.values())
        gold_vals = sorted(str(_norm_value(v)) for v in gold["positional"])
        return pred_vals == gold_vals
    if set(k.strip() for k in pk) != set(k.strip() for k in gk):
        return False
    gk_n = {k.strip(): v for k, v in gk.items()}
    pk_n = {k.strip(): v for k, v in pk.items()}
    return all(_values_equal(pk_n[k], v) for k, v in gk_n.items())


def _calls_equal(pred: List[Call], gold: List[Call]) -> bool:
    """顺序敏感的调用序列匹配（名称 + 参数）。"""
    if len(pred) != len(gold):
        return False
    for p, g in zip(pred, gold):
        if p["name"].strip() != g["name"].strip():
            # 容忍 ClassName.method 与 method 的写法差异
            if p["name"].strip().split(".")[-1] != g["name"].strip().split(".")[-1]:
                return False
        if not _args_match(p, g):
            return False
    return True


def match_ast(pred_calls: List[Call], gold_alternatives: List[List[Call]]) -> bool:
    return any(_calls_equal(pred_calls, alt) for alt in gold_alternatives)


# ---------------------------------------------------------------- 单样本评测
def evaluate_bfcl_sample(
    category: str, row: Dict[str, Any], pred_text: str
) -> Optional[bool]:
    """返回 True/False；None 表示该类别不由内置评测器计分（exec/multi_turn 等）。"""
    pred_calls = parse_model_calls(pred_text)

    if category in BFCL_IRRELEVANCE_CATEGORIES:
        return len(pred_calls) == 0
    if category in BFCL_RELEVANCE_CATEGORIES:
        return len(pred_calls) > 0
    if category in BFCL_AST_CATEGORIES:
        language = "python"
        if category == "java":
            language = "java"
        elif category == "javascript":
            language = "javascript"
        gold = parse_ground_truth(row.get("ground_truth"), language=language)
        if not gold:
            return None
        return match_ast(pred_calls, gold)
    # exec / rest / sql / multi_turn / chatable：需执行环境或多轮交互
    return None


def summarize_bfcl_results(
    category: str, verdicts: List[Optional[bool]], total: int
) -> Dict[str, Any]:
    scored = [v for v in verdicts if v is not None]
    metrics: Dict[str, Any] = {"category": category, "total": total}
    if scored:
        metrics["scored"] = len(scored)
        metrics["correct"] = sum(scored)
        metrics["accuracy"] = sum(scored) / len(scored)
    else:
        metrics["scored"] = 0
        metrics["accuracy"] = None
        metrics["note"] = (
            f"类别 {category} 需执行环境/多轮交互，内置评测不计分；"
            "如需官方分数请用 gorilla-bfcl 包评测 result jsonl"
        )
    return metrics
