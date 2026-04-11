"""在 EmbodiedBench 评测循环中注入 on_episode_start / env.step / on_episode_end 钩子。

通过 ContextVar 传递钩子实例，并在各 Env 的 ``__init__`` 结束时包装 ``reset`` / ``step``；
在 Evaluator 实例上包装 ``save_episode_metric`` 以在写盘前触发 ``on_episode_end``。
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional, Protocol

_hooks_ctx: ContextVar[Optional["EmbodiedBenchHooks"]] = ContextVar(
    "openttl_embodiedbench_hooks", default=None
)

_PATCHED_INITS: dict[type, Callable[..., None]] = {}
_PATCH_LOCK = False


class EmbodiedBenchHooks(Protocol):
    def on_episode_start(self, env: Any, reset_output: Any) -> None: ...

    def on_env_step_end(self, env: Any, step_output: Any) -> None: ...

    def on_episode_end(self, evaluator: Any, episode_info: dict) -> None: ...


class NoOpEmbodiedBenchHooks:
    def on_episode_start(self, env: Any, reset_output: Any) -> None:
        return None

    def on_env_step_end(self, env: Any, step_output: Any) -> None:
        return None

    def on_episode_end(self, evaluator: Any, episode_info: dict) -> None:
        return None


def current_hooks() -> Optional[EmbodiedBenchHooks]:
    return _hooks_ctx.get()


def _instrument_env_instance(env: Any, hooks: EmbodiedBenchHooks) -> None:
    orig_reset = env.reset
    orig_step = env.step

    def reset_wrapped(*args: Any, **kwargs: Any) -> Any:
        out = orig_reset(*args, **kwargs)
        hooks.on_episode_start(env, out)
        return out

    def step_wrapped(*args: Any, **kwargs: Any) -> Any:
        out = orig_step(*args, **kwargs)
        hooks.on_env_step_end(env, out)
        return out

    env.reset = types.MethodType(reset_wrapped, env)  # type: ignore[method-assign]
    env.step = types.MethodType(step_wrapped, env)  # type: ignore[method-assign]


def _make_patched_init(orig_cls: type, orig_init: Callable[..., None]) -> Callable[..., None]:
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        h = current_hooks()
        if h is not None and not isinstance(h, NoOpEmbodiedBenchHooks):
            _instrument_env_instance(self, h)

    return patched_init


def ensure_embodiedbench_env_patches() -> None:
    """幂等：为 EB 四个 Env 类打补丁，在构造完成后自动包装 reset/step。"""
    global _PATCH_LOCK
    if _PATCH_LOCK:
        return

    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
    from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
    from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
    from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv

    for cls in (EBAlfEnv, EBHabEnv, EBNavigationEnv, EBManEnv):
        if cls in _PATCHED_INITS:
            continue
        _PATCHED_INITS[cls] = cls.__init__  # type: ignore[assignment]
        cls.__init__ = _make_patched_init(cls, _PATCHED_INITS[cls])  # type: ignore[method-assign]

    _PATCH_LOCK = True


def patch_evaluator_save_episode_metric(
    evaluator: Any, hooks: Optional[EmbodiedBenchHooks]
) -> None:
    if hooks is None or isinstance(hooks, NoOpEmbodiedBenchHooks):
        return
    orig = evaluator.save_episode_metric

    def save_wrapped(episode_info: dict) -> None:
        hooks.on_episode_end(evaluator, episode_info)
        return orig(episode_info)

    evaluator.save_episode_metric = save_wrapped  # type: ignore[method-assign]


@contextmanager
def embodiedbench_hooks(hooks: Optional[EmbodiedBenchHooks]):
    token = _hooks_ctx.set(hooks)
    try:
        yield
    finally:
        _hooks_ctx.reset(token)
