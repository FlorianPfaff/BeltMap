from __future__ import annotations

import json
from pathlib import Path

from beltmap.cli.validate import build_parser, generate_validation_report, load_run_data
from beltmap.visual_qc import VisualQcArtifacts, generate_visual_qc


def markdown_link(path: Path, *, relative_to: Path) -> str:
    """Return a path suitable for Markdown links from ``relative_to``."""

    try:
        target = path.relative_to(relative_to)
    except ValueError:
        target = path
    return str(target).replace("\\", "/")


def image_gallery_lines(
    paths: list[Path],
    *,
    title: str,
    report_dir: Path,
    max_images: int = 3,
) -> list[str]:
    """Render a compact Markdown gallery for generated QC images."""

    if not paths:
        return [f"### {title}", "", "No sample images were generated.", ""]
    lines = [f"### {title}", ""]
    for path in paths[:max_images]:
        target = markdown_link(path, relative_to=report_dir)
        lines.extend([f"![{path.name}]({target})", ""])
    if len(paths) > max_images:
        lines.extend([f"- ... and {len(paths) - max_images} more image(s)", ""])
    return lines


def append_visual_qc_section(report_path: Path, artifacts: VisualQcArtifacts) -> None:
    """Append visual QC artifact links to the existing validation report."""

    report_dir = report_path.parent
    residual_histogram = artifacts.plots.get("residual_histogram")
    belt_map_coverage = artifacts.plots.get("belt_map_coverage")
    overlay_contact_sheet = artifacts.plots.get("overlay_contact_sheet")
    lines = [
        "",
        "## Visual quality-control artifacts",
        "",
        "These images are intended for manual sanity checking of real conveyor data.",
        "They help answer whether residual detections correspond to particles and",
        "whether reconstructed tracks connect the right components.",
        "",
    ]
    if residual_histogram is not None:
        lines.extend(
            [
                "### Residual histogram",
                "",
                f"![Residual histogram]({markdown_link(residual_histogram, relative_to=report_dir)})",
                "",
            ]
        )
    if belt_map_coverage is not None:
        lines.extend(
            [
                "### Belt-map coverage",
                "",
                f"![Belt-map coverage]({markdown_link(belt_map_coverage, relative_to=report_dir)})",
                "",
            ]
        )
    if overlay_contact_sheet is not None:
        lines.extend(
            [
                "### Overlay contact sheet",
                "",
                f"![Overlay contact sheet]({markdown_link(overlay_contact_sheet, relative_to=report_dir)})",
                "",
            ]
        )
    lines.extend(
        image_gallery_lines(
            artifacts.images.get("detections_overlay", []),
            title="Detection overlays",
            report_dir=report_dir,
        )
    )
    lines.extend(
        image_gallery_lines(
            artifacts.images.get("tracks_overlay", []),
            title="Track overlays",
            report_dir=report_dir,
        )
    )
    lines.extend(
        [
            "### Interpretation notes",
            "",
            "- `residual_histogram.png` summarizes saved residual preview PNG intensities.",
            "- `belt_map_coverage.png` is nominal phase-trajectory coverage, not the exact accumulation mask.",
            "- `overlay_contact_sheet.png` pairs detection and track overlays for quick review.",
            "- `detections_overlay_sample_*.png` overlays current-frame boxes and centroids on residual previews.",
            "- `tracks_overlay_sample_*.png` reconstructs PyRecEst-backed tracks for visual sanity checks.",
            "",
        ]
    )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    """Create the standard validation report plus visual QC artifacts."""

    parser = build_parser()
    args = parser.parse_args(argv)
    standard_artifacts = generate_validation_report(
        args.output_dir,
        report_path=args.report_path,
        make_plots=not args.no_plots,
    )
    visual_artifacts = VisualQcArtifacts(plots={}, images={})
    if not args.no_plots:
        visual_artifacts = generate_visual_qc(args.output_dir, load_run_data(args.output_dir))
        append_visual_qc_section(standard_artifacts.report, visual_artifacts)

    if not args.quiet:
        print(
            json.dumps(
                {
                    "report": str(standard_artifacts.report),
                    "summary": str(standard_artifacts.summary),
                    "plots": {
                        **{key: str(path) for key, path in standard_artifacts.plots.items()},
                        **{key: str(path) for key, path in visual_artifacts.plots.items()},
                    },
                    "extra_images": {
                        key: [str(path) for path in value]
                        for key, value in visual_artifacts.images.items()
                    },
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
