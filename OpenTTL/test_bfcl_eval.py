"""BFCL 解析与评测逻辑冒烟测试：python test_bfcl_eval.py"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

try:
    from openttl.data.bfcl import build_bfcl_messages, build_bfcl_plain_prompt
    from openttl.eval.bfcl_eval import (
        evaluate_bfcl_sample,
        parse_ground_truth,
        parse_model_calls,
    )
except ImportError:
    # 本地无 datasets 等依赖时：以孤立模块方式加载，跳过包 __init__
    for _sub in ("openttl", "openttl.data", "openttl.eval"):
        if _sub not in sys.modules:
            _m = types.ModuleType(_sub)
            _m.__path__ = [str(_ROOT / "src" / _sub.replace(".", "/"))]
            sys.modules[_sub] = _m

    def _load(name: str, path: str):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _bfcl = _load("openttl.data.bfcl", str(_ROOT / "src/openttl/data/bfcl.py"))
    _ev = _load("openttl.eval.bfcl_eval", str(_ROOT / "src/openttl/eval/bfcl_eval.py"))
    build_bfcl_messages = _bfcl.build_bfcl_messages
    build_bfcl_plain_prompt = _bfcl.build_bfcl_plain_prompt
    evaluate_bfcl_sample = _ev.evaluate_bfcl_sample
    parse_ground_truth = _ev.parse_ground_truth
    parse_model_calls = _ev.parse_model_calls


def check(name: str, cond: bool) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise SystemExit(f"smoke test failed: {name}")


def main() -> None:
    # 1. JSON 数组输出 -> simple 命中
    row = {"ground_truth": ["get_weather(city='San Francisco')"]}
    out = '[{"name": "get_weather", "arguments": {"city": "San Francisco"}}]'
    check("simple json", evaluate_bfcl_sample("simple", row, out) is True)

    # 2. Python 风格调用输出
    out = "get_weather(city='San Francisco')"
    check("simple python style", evaluate_bfcl_sample("simple", row, out) is True)

    # 3. 代码块包裹 + 多余解释文字
    out = (
        "I will call the function:\n```json\n"
        '[{"name": "get_weather", "arguments": {"city": "San Francisco"}}]\n```'
    )
    check("simple fenced json", evaluate_bfcl_sample("simple", row, out) is True)

    # 4. 参数值不一致 -> False
    out = '[{"name": "get_weather", "arguments": {"city": "New York"}}]'
    check("simple wrong arg", evaluate_bfcl_sample("simple", row, out) is False)

    # 5. parallel：两个调用，顺序敏感
    row = {
        "ground_truth": [
            "[get_weather(city='London'), get_weather(city='Paris')]"
        ]
    }
    out = (
        '[{"name": "get_weather", "arguments": {"city": "London"}}, '
        '{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    )
    check("parallel two calls", evaluate_bfcl_sample("parallel", row, out) is True)

    # 6. 数字参数等价（int vs float）
    row = {"ground_truth": ["calculate_area(length=5.0, width=3)"]}
    out = '[{"name": "calculate_area", "arguments": {"length": 5, "width": 3.0}}]'
    check("numeric equivalence", evaluate_bfcl_sample("simple", row, out) is True)

    # 7. irrelevance：不应调用
    row = {"ground_truth": []}
    check("irrelevance no call", evaluate_bfcl_sample("irrelevance", row, "No applicable function.") is True)
    check(
        "irrelevance with call",
        evaluate_bfcl_sample("irrelevance", row, 'func_a(x=1)') is False,
    )

    # 8. relevance：应有调用
    check(
        "relevance with call",
        evaluate_bfcl_sample("relevance", {"ground_truth": []}, 'func_a(x=1)') is True,
    )

    # 9. java 类别宽松解析
    row = {"ground_truth": ["Message.send(to='+1234567890', message='Hello')"]}
    out = '[{"name": "Message.send", "arguments": {"to": "+1234567890", "message": "Hello"}}]'
    check("java gt parse", evaluate_bfcl_sample("java", row, out) is True)

    # 10. exec 类别不计分（None）
    check(
        "exec not scored",
        evaluate_bfcl_sample("exec_simple", {"ground_truth": []}, "whatever") is None,
    )

    # 11. 位置参数 GT 与 JSON kwargs 的值多重集匹配
    gt = parse_ground_truth(["convert(100, 'USD', 'EUR')"])
    check("positional gt parsed", len(gt) == 1 and gt[0][0]["positional"] == [100, "USD", "EUR"])
    calls = parse_model_calls('[{"name": "convert", "arguments": {"amount": 100, "from": "USD", "to": "EUR"}}]')
    check("positional multiset match", len(calls) == 1)

    # 12. Qwen3.5 原生 XML tool_call
    out = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nSan Francisco\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    check(
        "qwen xml tool_call",
        evaluate_bfcl_sample(
            "simple",
            {"ground_truth": ["get_weather(city='San Francisco')"]},
            out,
        )
        is True,
    )

    # 13. 提示构造包含函数文档与问题
    row = {
        "function": [{"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}],
        "question": "do it",
    }
    msgs = build_bfcl_messages(row)
    check("messages structure", msgs[0]["role"] == "system" and "do it" in msgs[1]["content"])
    check("plain prompt", "do it" in build_bfcl_plain_prompt(row))

    # 14. 官方 possible_answer 格式（参数允许值列表，"" 表示可选）
    row = {
        "ground_truth": [
            {
                "calculate_triangle_area": {
                    "base": [10],
                    "height": [5],
                    "unit": ["units", ""],
                }
            }
        ]
    }
    out = '[{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}]'
    check("official gt optional omitted", evaluate_bfcl_sample("simple", row, out) is True)
    out = '[{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}]'
    check("official gt optional present", evaluate_bfcl_sample("simple", row, out) is True)
    out = '[{"name": "calculate_triangle_area", "arguments": {"base": 11, "height": 5}}]'
    check("official gt wrong value", evaluate_bfcl_sample("simple", row, out) is False)

    # 15. 嵌套 question 抽文本
    row = {
        "function": [{"name": "f", "description": "d", "parameters": {}}],
        "question": [[{"role": "user", "content": "nested q"}]],
    }
    check("nested question extracted", "nested q" in build_bfcl_messages(row)[1]["content"])

    print("All BFCL smoke tests passed.")


if __name__ == "__main__":
    main()
