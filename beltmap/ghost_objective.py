from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw

from .compare_runs import (
    DEFAULT_FROC_MAX_THRESHOLDS,
    RunSpec,
    load_labeled_detection_truth,
    load_run_data,
    read_csv_rows,
    summarize_run,
)


NON_BELTMAP_BASELINES = {"raw_robust_zscore", "static_average_background"}


@dataclass(frozen=True)
class GhostObjectiveWeights:
    """Weights for the ghost-aware configuration-selection objective."""

    f1: float = 1.0
    fp_frame: float = 0.01
    map_false_detections: float = 0.05
    map_false_long: float = 1.0
    map_false_accepted: float = 1.0
    small_accepted: float = 0.1
    mask_burden: float = 0.1


@dataclass(frozen=True)
class GhostObjectiveArtifacts:
    """Files written by a ghost-objective evaluation."""

    table_csv: Path
    report_md: Path
    plot_png: Path
    config_selection_json: Path


def finite_number(value: Any) -> float | None:
    """Parse a finite float, accepting common missing-value spellings."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "n/a",
        "nan",
        "none",
        "null",
    }:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite_integer(value: Any) -> int | None:
    parsed = finite_number(value)
    return None if parsed is None else int(round(parsed))


def first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return row[key]
    return None


def format_metric(value: Any, digits: int = 4) -> str:
    parsed = finite_number(value)
    if parsed is None:
        return "n/a"
    if abs(parsed - round(parsed)) < 1e-9:
        return str(int(round(parsed)))
    return f"{parsed:.{digits}f}"


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("expected LABEL=PATH")
    label, path_text = value.split("=", 1)
    label = label.strip()
    path_text = path_text.strip()
    if not label:
        raise ValueError("label before '=' must not be empty")
    if not path_text:
        raise ValueError("path after '=' must not be empty")
    return label, Path(path_text)


def row_variant(row: Mapping[str, Any]) -> str | None:
    value = first_present(row, ("variant", "label", "run", "config", "name"))
    return None if value is None else str(value)


def labeled_evidence_from_summary_csv(path: Path) -> dict[str, dict[str, Any]]:
    """Read labeled run metrics from a beltmap-compare summary CSV."""

    evidence: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        variant = row_variant(row)
        if not variant:
            continue
        evidence[variant] = labeled_evidence_from_summary_row(
            variant,
            row,
            source=str(path),
        )
    return evidence


def labeled_evidence_from_summary_row(
    variant: str,
    row: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "labeled_precision": finite_number(
            first_present(row, ("labeled_precision", "precision"))
        ),
        "labeled_recall": finite_number(first_present(row, ("labeled_recall", "recall"))),
        "labeled_f1": finite_number(first_present(row, ("labeled_f1", "f1"))),
        "fp_per_frame": finite_number(
            first_present(
                row,
                (
                    "labeled_false_positives_per_frame",
                    "fp_per_frame",
                    "false_positives_per_frame",
                ),
            )
        ),
        "froc_auc_le1": finite_number(
            first_present(
                row,
                ("labeled_froc_auc_fp_per_frame_le_1", "froc_auc_le1"),
            )
        ),
        "recall_at_0_1_fp_frame": finite_number(
            first_present(
                row,
                (
                    "labeled_froc_recall_at_0_1_fp_per_frame",
                    "recall_at_0_1_fp_frame",
                ),
            )
        ),
        "recall_at_0_5_fp_frame": finite_number(
            first_present(
                row,
                (
                    "labeled_froc_recall_at_0_5_fp_per_frame",
                    "recall_at_0_5_fp_frame",
                ),
            )
        ),
        "recall_at_1_0_fp_frame": finite_number(
            first_present(
                row,
                (
                    "labeled_froc_recall_at_1_0_fp_per_frame",
                    "recall_at_1_0_fp_frame",
                ),
            )
        ),
        "small_accepted_tracks": finite_integer(
            first_present(row, ("small_accepted_tracks_lt_50", "small_accepted_tracks"))
        ),
        "long_tracks_ge10": finite_integer(
            first_present(row, ("long_velocity_tracks_ge_10", "long_tracks_ge10"))
        ),
        "n_tracks": finite_integer(first_present(row, ("n_tracks", "tracks_500"))),
        "n_detections": finite_integer(
            first_present(row, ("n_detections", "detections_500"))
        ),
        "labeled_metrics_source": source,
        "output_dir": str(first_present(row, ("output_dir", "run_dir")) or ""),
    }


def labeled_evidence_from_runs(
    specs: list[RunSpec],
    *,
    truth_path: Path,
    truth_iou_threshold: float = 0.25,
    froc_max_thresholds: int | None = DEFAULT_FROC_MAX_THRESHOLDS,
) -> dict[str, dict[str, Any]]:
    """Compute labeled evidence directly from run directories and a truth file."""

    truth = load_labeled_detection_truth(truth_path)
    evidence: dict[str, dict[str, Any]] = {}
    for spec in specs:
        row = summarize_run(
            load_run_data(spec),
            labeled_truth=truth,
            truth_iou_threshold=truth_iou_threshold,
            froc_max_thresholds=froc_max_thresholds,
        )
        evidence[spec.label] = labeled_evidence_from_summary_row(
            spec.label,
            row,
            source=f"{truth_path} + {spec.output_dir}",
        )
    return evidence


def map_only_evidence_from_row(
    variant: str,
    row: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "map_only_false_detections": finite_integer(
            first_present(row, ("map_only_false_detections", "false_detections"))
        ),
        "map_only_false_long_tracks": finite_integer(
            first_present(row, ("map_only_false_long_tracks", "false_long_tracks"))
        ),
        "map_only_false_accepted_tracks": finite_integer(
            first_present(row, ("map_only_false_accepted_tracks", "false_accepted_tracks"))
        ),
        "masked_pixel_fraction": finite_number(
            first_present(
                row,
                (
                    "masked_pixel_fraction",
                    "mask_fraction",
                    "excessive_mask_fraction",
                ),
            )
        ),
        "map_only_metrics_source": source,
        "map_only_note": str(first_present(row, ("source_note", "selection_read", "note")) or ""),
    }


def load_one_map_only_path(label: str, path: Path) -> dict[str, Any]:
    """Load map-only metrics for one label from a JSON or CSV path."""

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        candidate = data
    elif path.suffix.lower() == ".csv":
        rows = read_csv_rows(path)
        if not rows:
            raise ValueError(f"{path} contains no rows")
        matching = [
            row
            for row in rows
            if (row_variant(row) or label) == label
        ]
        candidate = matching[0] if matching else rows[0]
    else:
        raise ValueError(f"{path} must be a JSON or CSV metrics file")
    return map_only_evidence_from_row(label, candidate, source=str(path))


def map_only_evidence_from_summary_csv(path: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        variant = row_variant(row)
        if not variant:
            continue
        evidence[variant] = map_only_evidence_from_row(variant, row, source=str(path))
    return evidence


def map_only_evidence_from_label_paths(values: Iterable[str]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for value in values:
        label, path = parse_label_path(value)
        evidence[label] = load_one_map_only_path(label, path)
    return evidence


def measured_mask_fraction(row: Mapping[str, Any]) -> float | None:
    direct = finite_number(row.get("masked_pixel_fraction"))
    if direct is not None:
        return direct
    recurrent = finite_number(row.get("recurrent_artifact_fraction_from_map"))
    return recurrent


def score_variant(row: Mapping[str, Any], weights: GhostObjectiveWeights) -> tuple[float | None, list[str]]:
    """Return ``(score, missing_terms)`` for one merged evidence row."""

    missing: list[str] = []
    f1 = finite_number(row.get("labeled_f1"))
    fp_frame = finite_number(row.get("fp_per_frame"))
    false_det = finite_number(row.get("map_only_false_detections"))
    false_long = finite_number(row.get("map_only_false_long_tracks"))
    false_accepted = finite_number(row.get("map_only_false_accepted_tracks"))
    for name, value in [
        ("labeled_f1", f1),
        ("fp_per_frame", fp_frame),
        ("map_only_false_detections", false_det),
        ("map_only_false_long_tracks", false_long),
        ("map_only_false_accepted_tracks", false_accepted),
    ]:
        if value is None:
            missing.append(name)
    if str(row.get("variant")) in NON_BELTMAP_BASELINES:
        missing.append("not_a_beltmap_config")
    if missing:
        return None, missing

    small_tracks = finite_number(row.get("small_accepted_tracks")) or 0.0
    n_tracks = finite_number(row.get("n_tracks")) or 0.0
    small_share = 0.0 if n_tracks <= 0 else small_tracks / n_tracks
    mask_burden = measured_mask_fraction(row) or 0.0
    assert f1 is not None
    assert fp_frame is not None
    assert false_det is not None
    assert false_long is not None
    assert false_accepted is not None
    score = (
        weights.f1 * f1
        - weights.fp_frame * fp_frame
        - weights.map_false_detections * false_det
        - weights.map_false_long * false_long
        - weights.map_false_accepted * false_accepted
        - weights.small_accepted * small_share
        - weights.mask_burden * mask_burden
    )
    return float(score), []


def merge_evidence(
    *,
    labeled: Mapping[str, Mapping[str, Any]],
    map_only: Mapping[str, Mapping[str, Any]],
    weights: GhostObjectiveWeights,
    variants: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Merge labeled and map-only rows, scoring eligible BeltMap variants."""

    ordered_variants = list(dict.fromkeys([*(variants or []), *labeled.keys(), *map_only.keys()]))
    rows: list[dict[str, Any]] = []
    for variant in ordered_variants:
        row: dict[str, Any] = {"variant": variant}
        row.update(map_only.get(variant, {}))
        row.update(labeled.get(variant, {}))
        n_tracks = finite_number(row.get("n_tracks")) or 0.0
        small_tracks = finite_number(row.get("small_accepted_tracks")) or 0.0
        row["small_accepted_tracks_share"] = 0.0 if n_tracks <= 0 else small_tracks / n_tracks
        row["mask_burden"] = measured_mask_fraction(row) or 0.0
        score, missing = score_variant(row, weights)
        row["ghost_objective_score"] = score
        row["eligible_for_selection"] = not missing
        row["missing_score_terms"] = ";".join(missing)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            0 if row["eligible_for_selection"] else 1,
            -float(row["ghost_objective_score"] if row["ghost_objective_score"] is not None else -np.inf),
            str(row["variant"]),
        )
    )
    return rows


def select_winner(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(row) for row in rows if row.get("eligible_for_selection")]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: float(row["ghost_objective_score"]),
    )


TABLE_FIELDS = [
    "variant",
    "eligible_for_selection",
    "ghost_objective_score",
    "labeled_precision",
    "labeled_recall",
    "labeled_f1",
    "fp_per_frame",
    "froc_auc_le1",
    "recall_at_0_1_fp_frame",
    "recall_at_0_5_fp_frame",
    "recall_at_1_0_fp_frame",
    "map_only_false_detections",
    "map_only_false_long_tracks",
    "map_only_false_accepted_tracks",
    "small_accepted_tracks",
    "small_accepted_tracks_share",
    "long_tracks_ge10",
    "n_tracks",
    "n_detections",
    "mask_burden",
    "missing_score_terms",
    "labeled_metrics_source",
    "map_only_metrics_source",
    "map_only_note",
    "output_dir",
]


def write_csv_table(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TABLE_FIELDS})


def write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    weights: GhostObjectiveWeights,
    selection: Mapping[str, Any] | None,
) -> None:
    lines = [
        "# Ghost-Aware BeltMap Objective",
        "",
        "This report scores BeltMap configurations with labeled detection quality and map-only ghost-control penalties.",
        "",
        "## Objective",
        "",
        "```text",
        f"score = {weights.f1:g} * F1",
        f"      - {weights.fp_frame:g} * FP_per_frame",
        f"      - {weights.map_false_detections:g} * map_only_false_detections",
        f"      - {weights.map_false_long:g} * map_only_false_long_tracks",
        f"      - {weights.map_false_accepted:g} * map_only_false_accepted_tracks",
        f"      - {weights.small_accepted:g} * small_accepted_tracks_share",
        f"      - {weights.mask_burden:g} * mask_burden",
        "```",
        "",
        "## Selection",
        "",
    ]
    if selection is None:
        lines.append("No variant had all required score terms.")
    else:
        lines.append(
            f"Selected `{selection['variant']}` with score {format_metric(selection.get('ghost_objective_score'))}."
        )
    lines.extend(
        [
            "",
            "## Table",
            "",
            "| variant | eligible | score | F1 | FP/frame | map-only false det/long/accepted | small share | missing terms |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in rows:
        triplet = "/".join(
            format_metric(row.get(field), 0)
            for field in (
                "map_only_false_detections",
                "map_only_false_long_tracks",
                "map_only_false_accepted_tracks",
            )
        )
        lines.append(
            f"| {row['variant']} | {row['eligible_for_selection']} | "
            f"{format_metric(row.get('ghost_objective_score'))} | "
            f"{format_metric(row.get('labeled_f1'))} | "
            f"{format_metric(row.get('fp_per_frame'))} | {triplet} | "
            f"{format_metric(row.get('small_accepted_tracks_share'))} | "
            f"{row.get('missing_score_terms') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a transparent configuration-selection objective, not a formal statistical significance test.",
            "- Variants missing map-only ghost metrics are ineligible rather than assumed clean.",
            "- Empty-frame specificity still requires reviewed empty or no-particle frames in the supplied truth set.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_selection_json(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    weights: GhostObjectiveWeights,
    selection: Mapping[str, Any] | None,
) -> None:
    payload = {
        "selected_variant": None if selection is None else selection["variant"],
        "selected_score": None if selection is None else selection["ghost_objective_score"],
        "weights": weights.__dict__,
        "eligible_variants": [row["variant"] for row in rows if row["eligible_for_selection"]],
        "ineligible_variants": {
            row["variant"]: row["missing_score_terms"]
            for row in rows
            if not row["eligible_for_selection"]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def selected_variant_from_rows(rows: Iterable[Mapping[str, Any]]) -> str | None:
    selection = select_winner(rows)
    return None if selection is None else str(selection["variant"])


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact PNG summary without requiring matplotlib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1200
    row_h = 44
    top = 72
    left = 250
    height = max(260, top + row_h * len(rows) + 50)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), "Ghost-aware BeltMap objective", fill="black")
    draw.text((left, 48), "objective score", fill="black")
    draw.text((left + 390, 48), "map-only false detections / long / accepted", fill="black")

    scores = [finite_number(row.get("ghost_objective_score")) for row in rows]
    finite_scores = [score for score in scores if score is not None]
    min_score = min([0.0, *finite_scores]) if finite_scores else 0.0
    max_score = max([1.0, *finite_scores]) if finite_scores else 1.0
    span = max_score - min_score if max_score != min_score else 1.0
    bar_w = 320
    selected_variant = selected_variant_from_rows(rows)

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        variant = str(row["variant"])
        selected = variant == selected_variant
        color = (49, 130, 73) if selected else (75, 120, 168)
        draw.text((20, y + 10), variant, fill="black")
        score = finite_number(row.get("ghost_objective_score"))
        if score is None:
            draw.text((left, y + 10), "not eligible", fill=(90, 90, 90))
        else:
            x0 = left + int((min(0.0, score) - min_score) / span * bar_w)
            x1 = left + int((max(0.0, score) - min_score) / span * bar_w)
            draw.rectangle((min(x0, x1), y + 8, max(x0, x1), y + 28), fill=color)
            draw.text((left + bar_w + 12, y + 10), format_metric(score), fill="black")
        triplet = "/".join(
            format_metric(row.get(field), 0)
            for field in (
                "map_only_false_detections",
                "map_only_false_long_tracks",
                "map_only_false_accepted_tracks",
            )
        )
        draw.text((left + 390, y + 10), triplet, fill="black")
        f1 = format_metric(row.get("labeled_f1"))
        fp_frame = format_metric(row.get("fp_per_frame"))
        draw.text((left + 660, y + 10), f"F1 {f1} | FP/frame {fp_frame}", fill="black")
    image.save(path)


def run_ghost_objective(
    *,
    output_dir: Path,
    labeled: Mapping[str, Mapping[str, Any]],
    map_only: Mapping[str, Mapping[str, Any]],
    weights: GhostObjectiveWeights,
    variants: Iterable[str] | None = None,
) -> GhostObjectiveArtifacts:
    rows = merge_evidence(labeled=labeled, map_only=map_only, weights=weights, variants=variants)
    selection = select_winner(rows)
    table_csv = output_dir / "ghost_objective_table.csv"
    report_md = output_dir / "ghost_objective_report.md"
    plot_png = output_dir / "ghost_objective_plot.png"
    selection_json = output_dir / "config_selection.json"
    write_csv_table(table_csv, rows)
    write_report(report_md, rows=rows, weights=weights, selection=selection)
    write_plot(plot_png, rows)
    write_selection_json(selection_json, rows=rows, weights=weights, selection=selection)
    return GhostObjectiveArtifacts(
        table_csv=table_csv,
        report_md=report_md,
        plot_png=plot_png,
        config_selection_json=selection_json,
    )
