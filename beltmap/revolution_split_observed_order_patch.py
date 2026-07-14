"""Select automatic holdout revolutions by observed order, not label value.

``build_revolution_split`` documents automatic evaluation selection as every
``eval_every``-th observed revolution.  The original implementation instead
applied modulo arithmetic directly to the integer revolution labels.  Sparse or
non-zero-based labels can therefore place every observed revolution in the same
modulo class, leaving either the training or evaluation split empty.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

from . import revolution_split as _revolution_split

_PATCHED_ATTR = "_beltmap_observed_order_revolution_split_patched"
_ORIGINAL_ATTR = "_beltmap_original_build_revolution_split"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the split builder behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_build_revolution_split = _unwrap_patched_callable(
    _revolution_split.build_revolution_split
)


def observed_order_build_revolution_split(
    revolution_by_frame: Sequence[int],
    *,
    eval_every: int = 3,
    eval_offset: int = 0,
    eval_revolutions: Sequence[int] = (),
    min_train_revolutions: int = 1,
    min_eval_revolutions: int = 1,
) -> _revolution_split.RevolutionSplit:
    """Assign automatic holdouts by ordinal position among observed revolutions.

    Explicit ``eval_revolutions`` retain their label-based semantics.  For the
    automatic path, observed revolution labels are mapped to contiguous ordinal
    indices before delegating to the original validated implementation, then
    mapped back into the returned split.  Contiguous zero-based inputs therefore
    remain unchanged while sparse and non-zero-based inputs follow the documented
    "every N-th observed revolution" behavior.
    """

    explicit_eval = tuple(eval_revolutions)
    if explicit_eval:
        return _original_build_revolution_split(
            revolution_by_frame,
            eval_every=eval_every,
            eval_offset=eval_offset,
            eval_revolutions=explicit_eval,
            min_train_revolutions=min_train_revolutions,
            min_eval_revolutions=min_eval_revolutions,
        )

    revolutions = _revolution_split._normalize_revolution_by_frame(
        revolution_by_frame
    )
    observed_revolutions = tuple(sorted(set(revolutions)))
    ordinal_by_revolution = {
        revolution: ordinal
        for ordinal, revolution in enumerate(observed_revolutions)
    }
    ordinal_split = _original_build_revolution_split(
        tuple(ordinal_by_revolution[revolution] for revolution in revolutions),
        eval_every=eval_every,
        eval_offset=eval_offset,
        eval_revolutions=(),
        min_train_revolutions=min_train_revolutions,
        min_eval_revolutions=min_eval_revolutions,
    )

    return _revolution_split.RevolutionSplit(
        revolution_by_frame=revolutions,
        frame_split=ordinal_split.frame_split,
        train_revolutions=tuple(
            observed_revolutions[ordinal]
            for ordinal in ordinal_split.train_revolutions
        ),
        eval_revolutions=tuple(
            observed_revolutions[ordinal]
            for ordinal in ordinal_split.eval_revolutions
        ),
        train_frame_indices=ordinal_split.train_frame_indices,
        eval_frame_indices=ordinal_split.eval_frame_indices,
    )


setattr(observed_order_build_revolution_split, _PATCHED_ATTR, True)
setattr(
    observed_order_build_revolution_split,
    _ORIGINAL_ATTR,
    _original_build_revolution_split,
)
_revolution_split.build_revolution_split = observed_order_build_revolution_split

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(
        _package,
        "build_revolution_split",
        observed_order_build_revolution_split,
    )
