from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from openttl.inference.base import InferenceEngine

LOG = logging.getLogger(__name__)


def _load_lora_state_dict(adapter_dir: str) -> Dict[str, Any]:
    """Load all tensors from the first safetensors or .bin file in adapter_dir."""
    sf_files = glob.glob(os.path.join(adapter_dir, "*.safetensors"))
    if sf_files:
        from safetensors.torch import load_file as _st_load
        return _st_load(sf_files[0], device="cpu")
    bin_files = glob.glob(os.path.join(adapter_dir, "*.bin"))
    if bin_files:
        import torch as _torch
        return _torch.load(bin_files[0], map_location="cpu", weights_only=True)
    return {}


def _infer_lora_arch_dims(adapter_dir: str) -> Dict[str, int]:
    """Infer architectural dimensions from saved LoRA adapter weights.

    Returns a dict with:
      ``hidden_size``      – INPUT dimension of linear projections (lora_A input, mode)
      ``max_lora_b_output`` – max OUTPUT dimension across all lora_B tensors
                              (= q_proj output = num_q_heads * head_dim for GQA models)

    Why two separate values?
    Some models use *expanded Q projections* where q_proj output > hidden_size
    (e.g. hidden_size=2048 but num_q_heads*head_dim=5120).  SGLang's LoRA buffer
    pre-allocates:
      - lora_A buffers sized [max_rank, hidden_size]  (INPUT dim)
      - lora_B buffers sized [q_proj_output, max_rank] (OUTPUT dim)
    These require two different ground-truth values from the adapter.
    """
    from collections import Counter
    a_dims: List[int] = []
    b_dims: List[int] = []
    try:
        sd = _load_lora_state_dict(adapter_dir)
        for k, v in sd.items():
            if not (hasattr(v, "shape") and len(v.shape) == 2):
                continue
            if "lora_A" in k:
                a_dims.append(int(v.shape[-1]))   # [rank, in_features]
            elif "lora_B" in k:
                b_dims.append(int(v.shape[0]))    # [out_features, rank]
    except Exception as _e:
        LOG.debug("_infer_lora_arch_dims failed: %s", _e)
    result: Dict[str, int] = {}
    if a_dims:
        # Most common lora_A input dim = hidden_size
        result["hidden_size"] = Counter(a_dims).most_common(1)[0][0]
    if b_dims:
        result["max_lora_b_output"] = max(b_dims)
    return result


def _infer_hidden_size_from_lora(adapter_dir: str) -> Optional[int]:
    """Compatibility shim – returns hidden_size only."""
    return _infer_lora_arch_dims(adapter_dir).get("hidden_size")


def _sum_token_logprobs_from_meta(meta: Any) -> Optional[float]:
    if not isinstance(meta, dict):
        return None
    for key in (
        "output_token_logprobs",
        "token_logprobs",
        "completion_token_logprobs",
    ):
        raw = meta.get(key)
        if raw is None:
            continue
        total = 0.0
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    v = item.get("logprob")
                    if v is not None:
                        total += float(v)
                    else:
                        return None
                elif isinstance(item, (int, float)):
                    total += float(item)
                else:
                    return None
            return total
    # 部分版本把 logprobs 放在顶层
    alt = meta.get("input_token_logprobs")
    if isinstance(alt, (list, tuple)) and alt:
        return _sum_token_logprobs_from_meta({"output_token_logprobs": alt})
    return None


def _text_from_generate_output(out: Any) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        t = out.get("text")
        if isinstance(t, str):
            return t
        if isinstance(t, list) and t and isinstance(t[0], str):
            return t[0]
    return str(out)


class SGLangOfflineEngine(InferenceEngine):
    """封装 ``sglang.Engine``：动态 LoRA、文本/多模态 generate、logprob 打分。"""

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer: Any,
        initial_lora_name: Optional[str] = None,
        initial_lora_path: Optional[str] = None,
        dtype: str = "bfloat16",
        tp_size: int = 1,
        mem_fraction_static: float = 0.45,
        disable_cuda_graph: bool = True,
        trust_remote_code: bool = True,
        max_lora_rank: int = 8,
        lora_target_modules: Optional[List[str]] = None,
        context_length: Optional[int] = None,
        log_level: str = "error",
        model_cfg_overrides: Optional[str] = None,
    ) -> None:
        from sglang import Engine

        self._tokenizer = tokenizer
        self._initial_lora_name = initial_lora_name
        use_lora = bool(initial_lora_path and initial_lora_name)
        self._current_lora_name: Optional[str] = initial_lora_name if use_lora else None

        kwargs: Dict[str, Any] = dict(
            model_path=model_path,
            dtype=dtype,
            tp_size=int(tp_size),
            mem_fraction_static=float(mem_fraction_static),
            disable_cuda_graph=bool(disable_cuda_graph),
            trust_remote_code=bool(trust_remote_code),
            enable_lora=use_lora,
            log_level=log_level,
        )
        if use_lora:
            kwargs["lora_paths"] = [
                {
                    "lora_name": initial_lora_name,
                    "lora_path": initial_lora_path,
                    "pinned": True,
                }
            ]
            kwargs["max_lora_rank"] = int(max_lora_rank)
            if lora_target_modules:
                kwargs["lora_target_modules"] = list(lora_target_modules)
            # Pass architecture overrides (hoisted from nested sub-configs) so that
            # SGLang's LoRA manager can find attributes like vocab_size / num_hidden_layers
            # even when the custom config class (e.g. Qwen3_5Config) stores them nested.
            if model_cfg_overrides:
                try:
                    kwargs["json_model_override_args"] = model_cfg_overrides
                except Exception:
                    pass
        if context_length is not None:
            kwargs["context_length"] = int(context_length)

        self._engine = Engine(**kwargs)

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def current_lora_name(self) -> Optional[str]:
        return self._current_lora_name

    def sync_lora(self, local_dir: str, new_name: str) -> str:
        prev = self._current_lora_name
        self._engine.load_lora_adapter(new_name, local_dir, pinned=False)
        self._current_lora_name = new_name
        if prev and prev != new_name:
            try:
                self._engine.unload_lora_adapter(prev)
            except Exception as e:  # pragma: no cover
                LOG.warning("unload_lora_adapter(%s) failed: %s", prev, e)
        return new_name

    def _lora_req_list(self, name: Optional[str]) -> Optional[List[Optional[str]]]:
        n = name or self._current_lora_name
        if not n:
            return None
        return [n]

    def generate(
        self,
        prompt: Union[str, List[str]],
        *,
        image_data: Optional[Any] = None,
        sampling_params: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        lora_name: Optional[str] = None,
    ) -> Union[str, List[str]]:
        single = isinstance(prompt, str)
        prompts: List[str] = [prompt] if single else list(prompt)
        lp = self._lora_req_list(lora_name)  # returns [name] or None
        # SGLang Engine API:
        #   single request → lora_path must be a str (or None), NOT a list
        #   batch requests → lora_path must be a list aligned with prompts
        if single:
            lora_arg: Any = lp[0] if lp else None
        else:
            lora_arg = lp * len(prompts) if lp else None
        out = self._engine.generate(
            prompt=prompts[0] if single else prompts,
            sampling_params=sampling_params,
            image_data=image_data,
            lora_path=lora_arg,
        )
        if single:
            return _text_from_generate_output(out)
        # batch：期望 list-like
        if isinstance(out, list):
            return [_text_from_generate_output(x) for x in out]
        if isinstance(out, dict) and isinstance(out.get("text"), list):
            return [str(x) for x in out["text"]]
        return [_text_from_generate_output(out)]

    def score_logprob_sum(
        self,
        *,
        full_text: str,
        prefix_len_tokens: int,
        lora_name: Optional[str] = None,
    ) -> float:
        toks = self._tokenizer(
            full_text,
            return_tensors=None,
            add_special_tokens=True,
            truncation=False,
        )
        ids: List[int] = list(toks["input_ids"])
        if prefix_len_tokens < 0 or prefix_len_tokens > len(ids):
            return float("-inf")
        lp = self._lora_req_list(lora_name)
        # 与 HF CLM 对齐：continuation 首 token 由 logits[plen-1] 预测
        start = max(0, int(prefix_len_tokens) - 1)
        out = self._engine.generate(
            input_ids=ids,
            sampling_params={"max_new_tokens": 0},
            return_logprob=True,
            logprob_start_len=start,
            lora_path=lp,
        )
        meta = out.get("meta_info") if isinstance(out, dict) else None
        s = _sum_token_logprobs_from_meta(meta) if meta else None
        if s is None and isinstance(out, dict):
            s = _sum_token_logprobs_from_meta(out)
        if s is None:
            LOG.warning(
                "SGLang logprob parse failed; keys=%s",
                list(out.keys()) if isinstance(out, dict) else type(out),
            )
            return float("-inf")
        return float(s)

    def shutdown(self) -> None:
        if hasattr(self._engine, "shutdown"):
            try:
                self._engine.shutdown()
            except Exception:  # pragma: no cover
                pass


def build_sglang_engine_from_omegaconf(
    model_cfg: Any,
    inference_cfg: Any,
    tokenizer: Any,
    initial_lora_path: Optional[str] = None,
) -> SGLangOfflineEngine:
    """从 Hydra / OmegaConf 构造引擎。"""
    from omegaconf import OmegaConf

    mp = OmegaConf.select(model_cfg, "pretrained_model_name_or_path")
    mp = str(mp or "")
    if not mp:
        raise ValueError("model.pretrained_model_name_or_path required for SGLang")
    init_name = getattr(inference_cfg, "initial_lora_name", None)
    if init_name is not None:
        init_name = str(init_name)
        if init_name.lower() in ("none", "null", ""):
            init_name = None
    # When a LoRA path is provided (online TTA) but no name is configured,
    # fall back to "tta_v0" — consistent with OnlineTTARunner.initial_adapter_path.
    if initial_lora_path and not init_name:
        init_name = "tta_v0"
    lora_tm = OmegaConf.select(inference_cfg, "lora_target_modules")
    if lora_tm is not None:
        lora_tm = list(OmegaConf.to_container(lora_tm, resolve=True))
    ctx = OmegaConf.select(inference_cfg, "context_length")
    use_lora = bool(initial_lora_path)

    # Some custom configs (e.g. Qwen3_5Config, Qwen2_5VLConfig) store model-architecture
    # attributes inside a nested sub-config (text_config / llm_config / language_config)
    # rather than as direct top-level attributes.  SGLang's LoRA manager accesses these
    # attributes directly on the config object, so it will raise AttributeError for every
    # missing field.  We fix this once by reading config.json, finding all nested sub-configs,
    # and hoisting their keys to the top-level override dict so SGLang always finds them.
    model_cfg_overrides: Optional[str] = None
    if use_lora:
        try:
            cfg_json_path = os.path.join(mp, "config.json")
            with open(cfg_json_path, "r", encoding="utf-8") as _f:
                _raw: Dict[str, Any] = json.load(_f)

            # Sub-config priority: llm_config (language decoder) > language_config >
            # language_model_config > text_config (may be a different encoder/tokenizer
            # sub-model with different hidden dimensions for multimodal models).
            _SUB_KEYS = (
                "llm_config",
                "language_config",
                "language_model_config",
                "text_config",
            )

            # Step 1 – Scalar attrs: hoist from nested sub-configs when missing at top level.
            # llm_config is tried before text_config because for VL models text_config
            # may describe a visual encoder with different dimensions.
            _SCALAR_ATTRS = (
                "vocab_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "max_position_embeddings",
                "head_dim",
                "rms_norm_eps",
                # hidden_size is fixed up in Step 2 using ground-truth adapter dims
            )
            _override: Dict[str, Any] = {}
            for _attr in _SCALAR_ATTRS:
                if _attr in _raw:
                    continue
                for _sub in _SUB_KEYS:
                    _val = (_raw.get(_sub) or {}).get(_attr)
                    if _val is not None:
                        _override[_attr] = _val
                        break

            # Step 2 – hidden_size and num_attention_heads from adapter weights.
            #
            # Problem: some models use *expanded Q projections* where
            #   q_proj output_dim  (= num_q_heads × head_dim)  >>  hidden_size
            # e.g. hidden_size=2048 but q_proj output=5120 (40 heads × 128 dim).
            #
            # SGLang allocates:
            #   lora_A buffer: [max_rank, hidden_size]      ← needs real hidden_size
            #   lora_B buffer: [num_attention_heads×head_dim, max_rank] ← needs real Q output
            #
            # The sub-config's num_attention_heads may correspond to a different
            # sub-model (e.g. text encoder with 24 heads) and give the wrong Q output.
            # We use adapter weights as ground truth for both values.
            _lora_dims = _infer_lora_arch_dims(str(initial_lora_path))
            _hs   = _lora_dims.get("hidden_size")
            _max_b = _lora_dims.get("max_lora_b_output")

            if "hidden_size" not in _raw:
                if _hs is not None:
                    _override["hidden_size"] = _hs
                    LOG.info("SGLang LoRA shim: hidden_size=%d inferred from lora_A dims", _hs)
                else:
                    for _sub in _SUB_KEYS:
                        _val = (_raw.get(_sub) or {}).get("hidden_size")
                        if _val is not None:
                            _override["hidden_size"] = _val
                            LOG.warning("SGLang LoRA shim: hidden_size=%s from sub-config %s "
                                        "(adapter unreadable; may be wrong)", _val, _sub)
                            break

            # Recompute num_attention_heads from adapter lora_B output / head_dim.
            # This overrides the sub-config value when the model has expanded Q heads.
            _hd = _override.get("head_dim") or _raw.get("head_dim")
            if _max_b and _hd and int(_hd) > 0 and (_max_b % int(_hd) == 0):
                _inferred_nheads = _max_b // int(_hd)
                if _inferred_nheads != _override.get("num_attention_heads"):
                    LOG.info(
                        "SGLang LoRA shim: overriding num_attention_heads "
                        "%s → %d (max_lora_b=%d / head_dim=%d from adapter weights)",
                        _override.get("num_attention_heads"), _inferred_nheads, _max_b, _hd,
                    )
                    _override["num_attention_heads"] = _inferred_nheads

            if _override:
                model_cfg_overrides = json.dumps(_override)
                LOG.info("SGLang LoRA shim: final overrides=%s", _override)
        except Exception as _exc:
            LOG.debug("Could not read model config.json for LoRA override: %s", _exc)

    return SGLangOfflineEngine(
        model_path=mp,
        tokenizer=tokenizer,
        initial_lora_name=init_name if use_lora else None,
        initial_lora_path=initial_lora_path if use_lora else None,
        dtype=str(getattr(inference_cfg, "dtype", "bfloat16")),
        tp_size=int(getattr(inference_cfg, "tp_size", 1)),
        mem_fraction_static=float(getattr(inference_cfg, "mem_fraction_static", 0.45)),
        disable_cuda_graph=bool(getattr(inference_cfg, "disable_cuda_graph", True)),
        trust_remote_code=bool(getattr(inference_cfg, "trust_remote_code", True)),
        max_lora_rank=int(OmegaConf.select(inference_cfg, "max_lora_rank") or 8),
        lora_target_modules=lora_tm,
        context_length=int(ctx) if ctx is not None else None,
        log_level=str(getattr(inference_cfg, "log_level", "error")),
        model_cfg_overrides=model_cfg_overrides,
    )
