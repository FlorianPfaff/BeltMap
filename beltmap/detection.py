"""Particle detection from normalized belt residual images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .residual import ResidualImage


_IMPORT_UNCHECKED = object()
_IMPORT_MISSING = object()
_SCIPY_NDIMAGE: Any = _IMPORT_UNCHECKED


@dataclass(frozen=True)
class ParticleMaskCleanupConfig:
    """Optional morphology applied after residual thresholding.

    Defaults preserve the historical behavior: a pixel is detected when it is
    valid and above threshold.  The cleanup operations are intentionally small
    and conservative; they are meant to join threshold fragments and fill tiny
    holes caused by static-background-corrected residual texture, not to invent
    large particle shapes.
    """

    closing_radius_px: int = 0
    fill_holes: bool = False
    opening_radius_px: int = 0
    min_component_area_px: int = 0
    connectivity: int = 8


def detect_particles_from_residual(
    residual: ArrayLike | ResidualImage,
    *,
    threshold: float,
    mask: ArrayLike | None = None,
    cleanup: ParticleMaskCleanupConfig | None = None,
) -> NDArray[np.bool_]:
    """Detect bright particles by thresholding a normalized residual image.

    For bright particles on a darker belt this is intentionally just:

    ``particle_mask = residual > threshold``

    Invalid pixels, non-finite residuals, and any pixels excluded by ``mask`` are
    returned as ``False``.

    Set ``cleanup`` to close one-pixel gaps, fill holes, or remove very small
    threshold components before connected-component extraction.  This is useful
    for additive static-background residuals that otherwise fragment particles.
    """

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if isinstance(residual, ResidualImage):
        values = np.asarray(residual.normalized, dtype=np.float64)
        valid = np.asarray(residual.mask, dtype=bool).copy()
    else:
        values = np.asarray(residual, dtype=np.float64)
        if values.size == 0:
            raise ValueError("residual must not be empty")
        valid = np.ones(values.shape, dtype=bool)

    valid &= np.isfinite(values)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != values.shape:
            raise ValueError("mask must have the same shape as residual")
        valid &= user_mask

    threshold_mask = valid & (values > threshold)
    return cleanup_particle_mask(threshold_mask, valid_mask=valid, config=cleanup)


def cleanup_particle_mask(
    particle_mask: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
    config: ParticleMaskCleanupConfig | None = None,
) -> NDArray[np.bool_]:
    """Clean a threshold mask before connected-component extraction."""

    cfg = config or ParticleMaskCleanupConfig()
    _validate_cleanup_config(cfg)

    cleaned = np.asarray(particle_mask, dtype=bool).copy()
    if cleaned.ndim != 2 or cleaned.size == 0:
        raise ValueError("particle_mask must be a non-empty 2-D array")

    if valid_mask is None:
        valid = np.ones(cleaned.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != cleaned.shape:
            raise ValueError("valid_mask must have the same shape as particle_mask")
        cleaned &= valid

    if cfg.closing_radius_px > 0:
        cleaned = _binary_erosion(
            _binary_dilation(cleaned, radius=cfg.closing_radius_px),
            radius=cfg.closing_radius_px,
        )
        cleaned &= valid

    if cfg.fill_holes:
        cleaned = _binary_fill_holes(cleaned, valid=valid)
        cleaned &= valid

    if cfg.opening_radius_px > 0:
        cleaned = _binary_dilation(
            _binary_erosion(cleaned, radius=cfg.opening_radius_px),
            radius=cfg.opening_radius_px,
        )
        cleaned &= valid

    if cfg.min_component_area_px > 1:
        cleaned = _remove_small_components(
            cleaned,
            min_area_px=cfg.min_component_area_px,
            connectivity=cfg.connectivity,
        )
        cleaned &= valid

    return cleaned


def _validate_cleanup_config(config: ParticleMaskCleanupConfig) -> None:
    if config.closing_radius_px < 0:
        raise ValueError("closing_radius_px must be non-negative")
    if config.opening_radius_px < 0:
        raise ValueError("opening_radius_px must be non-negative")
    if config.min_component_area_px < 0:
        raise ValueError("min_component_area_px must be non-negative")
    if config.connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")


def _binary_dilation(mask: NDArray[np.bool_], *, radius: int) -> NDArray[np.bool_]:
    if radius <= 0:
        return mask.copy()
    ndimage = _load_scipy_ndimage()
    if ndimage is not None:
        return np.asarray(
            ndimage.binary_dilation(mask, structure=_square_structure(radius)),
            dtype=bool,
        )
    return _binary_morphology_numpy(mask, radius=radius, operation="dilation")


def _binary_erosion(mask: NDArray[np.bool_], *, radius: int) -> NDArray[np.bool_]:
    if radius <= 0:
        return mask.copy()
    ndimage = _load_scipy_ndimage()
    if ndimage is not None:
        return np.asarray(
            ndimage.binary_erosion(mask, structure=_square_structure(radius)),
            dtype=bool,
        )
    return _binary_morphology_numpy(mask, radius=radius, operation="erosion")


def _binary_morphology_numpy(
    mask: NDArray[np.bool_],
    *,
    radius: int,
    operation: str,
) -> NDArray[np.bool_]:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    height, width = mask.shape
    slices = [
        padded[row_offset : row_offset + height, col_offset : col_offset + width]
        for row_offset in range(2 * radius + 1)
        for col_offset in range(2 * radius + 1)
    ]
    if operation == "dilation":
        out = np.zeros(mask.shape, dtype=bool)
        for part in slices:
            out |= part
        return out
    if operation == "erosion":
        out = np.ones(mask.shape, dtype=bool)
        for part in slices:
            out &= part
        return out
    raise ValueError(f"unsupported morphology operation {operation!r}")


def _binary_fill_holes(
    mask: NDArray[np.bool_],
    *,
    valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Fill holes inside ``mask`` while treating invalid pixels as outside."""

    domain = ~mask
    seed = np.zeros(mask.shape, dtype=bool)
    seed[0, :] |= domain[0, :]
    seed[-1, :] |= domain[-1, :]
    seed[:, 0] |= domain[:, 0]
    seed[:, -1] |= domain[:, -1]
    seed |= ~valid
    seed &= domain

    ndimage = _load_scipy_ndimage()
    if ndimage is not None:
        outside = np.asarray(ndimage.binary_propagation(seed, mask=domain), dtype=bool)
    else:
        outside = _binary_propagation_numpy(seed, domain=domain)
    holes = valid & ~mask & ~outside
    return mask | holes


def _binary_propagation_numpy(
    seed: NDArray[np.bool_],
    *,
    domain: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    visited = np.asarray(seed & domain, dtype=bool).copy()
    height, width = visited.shape
    stack = [(int(row), int(col)) for row, col in np.argwhere(visited)]
    while stack:
        row, col = stack.pop()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and domain[next_row, next_col]
                and not visited[next_row, next_col]
            ):
                visited[next_row, next_col] = True
                stack.append((next_row, next_col))
    return visited


def _remove_small_components(
    mask: NDArray[np.bool_],
    *,
    min_area_px: int,
    connectivity: int,
) -> NDArray[np.bool_]:
    kept = np.zeros(mask.shape, dtype=bool)
    ndimage = _load_scipy_ndimage()
    if ndimage is not None:
        labels, component_count = ndimage.label(mask, structure=_component_structure(connectivity))
        for label in range(1, int(component_count) + 1):
            component = labels == label
            if int(np.count_nonzero(component)) >= min_area_px:
                kept |= component
        return kept

    for rows, cols in _connected_components_numpy(mask, connectivity=connectivity):
        if rows.size >= min_area_px:
            kept[rows, cols] = True
    return kept


def _connected_components_numpy(
    mask: NDArray[np.bool_],
    *,
    connectivity: int,
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]]:
    offsets = (
        [(-1, 0), (0, -1), (0, 1), (1, 0)]
        if connectivity == 4
        else [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
    )
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[NDArray[np.integer], NDArray[np.integer]]] = []
    height, width = mask.shape
    for start_row, start_col in np.argwhere(mask):
        row = int(start_row)
        col = int(start_col)
        if visited[row, col]:
            continue
        stack = [(row, col)]
        visited[row, col] = True
        rows: list[int] = []
        cols: list[int] = []
        while stack:
            current_row, current_col = stack.pop()
            rows.append(current_row)
            cols.append(current_col)
            for row_offset, col_offset in offsets:
                next_row = current_row + row_offset
                next_col = current_col + col_offset
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and mask[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
        components.append((np.asarray(rows), np.asarray(cols)))
    return components


def _square_structure(radius: int) -> NDArray[np.bool_]:
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)


def _component_structure(connectivity: int) -> NDArray[np.bool_]:
    if connectivity == 4:
        return np.array(
            [[False, True, False], [True, True, True], [False, True, False]],
            dtype=bool,
        )
    if connectivity == 8:
        return np.ones((3, 3), dtype=bool)
    raise ValueError("connectivity must be 4 or 8")


def _load_scipy_ndimage() -> Any | None:
    global _SCIPY_NDIMAGE

    if _SCIPY_NDIMAGE is _IMPORT_UNCHECKED:
        try:
            from scipy import ndimage
        except ImportError:
            _SCIPY_NDIMAGE = _IMPORT_MISSING
        else:
            _SCIPY_NDIMAGE = ndimage
    return None if _SCIPY_NDIMAGE is _IMPORT_MISSING else _SCIPY_NDIMAGE
