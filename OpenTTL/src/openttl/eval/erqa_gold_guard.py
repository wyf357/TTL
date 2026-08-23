"""Helpers for ERQA evaluation protocols (e.g., no ground-truth read before online TTA)."""

from __future__ import annotations


class DelayedGoldExample(dict):
    """Block reads of ``answer`` until :meth:`allow_gold_read` (strict no-label-feedback mode)."""

    __slots__ = ("_gold_ok",)

    def __init__(self, raw: dict, *, strict: bool):
        super().__init__(raw)
        object.__setattr__(self, "_gold_ok", not strict)

    def allow_gold_read(self) -> None:
        object.__setattr__(self, "_gold_ok", True)

    def __getitem__(self, key: str):
        if key == "answer" and not self._gold_ok:
            raise RuntimeError(
                "erqa.strict_no_label_feedback: ground-truth 'answer' was read before "
                "the TTA update completed (no-label-feedback protocol violation)."
            )
        return super().__getitem__(key)
