
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_adapteval")
def main(cfg: DictConfig) -> None:
    from peft import PeftModel

    from openttl.adapters.registry import extract_model_cfg
    from openttl.models.loader import load_adapter

    mc = extract_model_cfg(cfg)
    adapter = load_adapter(cfg)
    adapter.load_processor(mc)
    tokenizer = adapter.tokenizer()
    model = adapter.load_model(mc)
    ap = OmegaConf.select(cfg, "adapter_path")
    if ap:
        model = PeftModel.from_pretrained(model, ap)
    if bool(OmegaConf.select(cfg, "merge_lora") or False):
        model = model.merge_and_unload()
    model.eval()
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    mnt = int(OmegaConf.select(cfg, "max_new_tokens") or 32)
    out = model.generate(**inputs, max_new_tokens=mnt)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    metrics = {"sample_decode": text}
    outp = Path(str(cfg.output_json))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
