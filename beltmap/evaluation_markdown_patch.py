"""Keep evaluation Markdown tables valid for arbitrary run labels."""

from __future__ import annotations

from html import escape
from typing import Any

from . import evaluation as _evaluation

_PATCHED_ATTR = "_beltmap_evaluation_markdown_table_escaping_patched"
_ORIGINAL_ATTR = "_beltmap_original_build_markdown"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the report builder behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_build_markdown = _unwrap_patched_callable(_evaluation.build_markdown)


def markdown_table_cell(value: Any) -> str:
    """Return text safe for one GitHub-Flavored Markdown table cell."""

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = escape(text, quote=False)
    return text.replace("|", "&#124;").replace("\n", "<br>")


def safe_build_markdown(summaries: list[dict[str, Any]]) -> str:
    """Build an evaluation report without allowing labels to split table rows."""

    lines = [
        "# BeltMap evaluation summary",
        "",
        "This report compares completed `beltmap-apply` output directories. It is intended for ablations such as baseline vs. phase feedback, static background/noise learning, threshold settings, and tracker settings.",
        "",
        "## Run comparison",
        "",
        "| Run | Images | Detections | Tracks | Median abs correction px | Median score | Mean detections/frame | Median velocity ratio | Observed map | Missing files |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        run = f"<code>{markdown_table_cell(summary['run'])}</code>"
        missing_files = markdown_table_cell(summary.get("missing_files") or "none")
        lines.append(
            "| "
            + " | ".join(
                [
                    run,
                    _evaluation.format_markdown_value(summary.get("n_images")),
                    _evaluation.format_markdown_value(summary.get("n_detections")),
                    _evaluation.format_markdown_value(summary.get("n_tracks")),
                    _evaluation.format_markdown_value(
                        summary.get("phase_correction_abs_median_px")
                    ),
                    _evaluation.format_markdown_value(
                        summary.get("registration_score_median")
                    ),
                    _evaluation.format_markdown_value(
                        summary.get("detections_per_frame_mean")
                    ),
                    _evaluation.format_markdown_value(
                        summary.get("velocity_ratio_y_median")
                    ),
                    _evaluation.format_markdown_value(
                        summary.get("belt_map_observed_fraction")
                    ),
                    missing_files,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpreting ablations",
            "",
            "- Lower absolute phase corrections and higher registration scores usually indicate a cleaner phase model and belt map.",
            "- Large detection-count changes should be checked against residual previews before interpreting them as improvements.",
            "- Velocity-ratio outliers outside the plausible physical range are a useful proxy for fragmented or mismatched tracks.",
            "- Low observed-map or contributed-map fractions indicate that map quality may be limited by insufficient belt coverage or excessive masking.",
            "",
        ]
    )
    return "\n".join(lines)


setattr(safe_build_markdown, _PATCHED_ATTR, True)
setattr(safe_build_markdown, _ORIGINAL_ATTR, _original_build_markdown)
_evaluation.build_markdown = safe_build_markdown
