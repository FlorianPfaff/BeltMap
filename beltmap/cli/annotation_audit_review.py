from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import tempfile
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def crop_name(frame_index: int) -> str:
    return f"frame_{frame_index:06d}.png"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def crop_box_from_reference(reference: dict[str, Any]) -> tuple[int, int, int, int]:
    coord = reference.get("coordinate_system") or {}
    region = coord.get("crop_region_in_source_image") or {}
    top = int(region.get("top", 0))
    left = int(region.get("left", 220))
    height = int(region.get("height", 1330))
    width = int(region.get("width", 1800))
    return left, top, left + width, top + height


def crop_size_from_reference(reference: dict[str, Any]) -> tuple[int, int]:
    left, top, right, bottom = crop_box_from_reference(reference)
    return right - left, bottom - top


def find_source_image(source_image_dir: Path, frame_index: int) -> Path | None:
    suffix5 = f"{frame_index:05d}"
    suffix6 = f"{frame_index:06d}"
    candidates = [
        source_image_dir
        / "ZiegelzuKalk50zu50_20gpros"
        / f"ZiegelzuKalk50zu50_20gpros{suffix5}.bmp",
        source_image_dir / f"ZiegelzuKalk50zu50_20gpros{suffix5}.bmp",
        source_image_dir / f"frame_{suffix6}.png",
        source_image_dir / f"frame_{suffix6}.bmp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(source_image_dir.rglob(f"*{suffix5}.bmp"))
    if matches:
        return matches[0]
    matches = sorted(source_image_dir.rglob(f"*{suffix6}.png"))
    if matches:
        return matches[0]
    return None


@dataclass
class AuditServerState:
    payload: dict[str, Any]
    image_paths: dict[str, Path]
    review_path: Path


def load_reference_by_frame(audit_dir: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    reference_path = audit_dir / "audit_original_reference_labels.json"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing {reference_path}")
    reference = read_json(reference_path)
    frames = {int(row["frame_index"]): row for row in reference.get("frames", [])}
    return reference, frames


def ensure_review_data(
    *,
    review_path: Path,
    audit_dir: Path,
    selected_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if review_path.is_file():
        return read_json(review_path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {
        "purpose": "brick20g_annotation_audit_click_review",
        "status": "pending_review",
        "created_at": now,
        "updated_at": now,
        "source_audit_dir": str(audit_dir.as_posix()),
        "frames": [
            {
                "frame_index": int(row["frame_index"]),
                "primary_bucket": row.get("primary_bucket", ""),
                "review_status": "unreviewed",
                "accept_existing_boxes": None,
                "mistake_points": [],
                "notes": "",
                "reviewer_id": "",
            }
            for row in selected_rows
        ],
    }


def update_review_status(review_data: dict[str, Any]) -> None:
    frames = review_data.get("frames", [])
    if frames and all(row.get("review_status") != "unreviewed" for row in frames):
        review_data["status"] = "reviewed_with_click_flags"
    else:
        review_data["status"] = "in_progress"
    review_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")


def image_id_for(frame_index: int, offset: int) -> str:
    return f"f{frame_index:06d}_o{offset:+d}"


def load_or_make_context_crop(
    *,
    audit_dir: Path,
    crop_dir: Path,
    source_image_dir: Path | None,
    crop_box: tuple[int, int, int, int],
    target_frame: int,
    context_frame: int,
    offset: int,
) -> Path | None:
    if offset == 0:
        target_crop = audit_dir / "raw_crops" / crop_name(target_frame)
        if target_crop.is_file():
            return target_crop

    cached = audit_dir / "context_crops" / f"frame_{target_frame:06d}" / crop_name(context_frame)
    if cached.is_file():
        return cached

    reviewed_crop = crop_dir / crop_name(context_frame)
    if reviewed_crop.is_file():
        return reviewed_crop

    if source_image_dir is None:
        return None

    source_path = find_source_image(source_image_dir, context_frame)
    if source_path is None:
        return None

    from PIL import Image

    cached.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source_path).convert("L").crop(crop_box)
    image.save(cached)
    return cached


def build_payload(
    *,
    audit_dir: Path,
    crop_dir: Path,
    source_image_dir: Path | None,
    review_path: Path,
    context_radius: int,
) -> AuditServerState:
    selection_path = audit_dir / "audit_frame_selection.csv"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing {selection_path}")
    selected_rows = read_csv_rows(selection_path)
    reference, reference_by_frame = load_reference_by_frame(audit_dir)
    crop_box = crop_box_from_reference(reference)
    crop_width, crop_height = crop_size_from_reference(reference)
    review_data = ensure_review_data(
        review_path=review_path,
        audit_dir=audit_dir,
        selected_rows=selected_rows,
    )
    review_by_frame = {
        int(row["frame_index"]): row for row in review_data.get("frames", [])
    }

    image_paths: dict[str, Path] = {}
    frames: list[dict[str, Any]] = []
    for ordinal, row in enumerate(selected_rows, start=1):
        frame_index = int(row["frame_index"])
        reference_row = reference_by_frame.get(frame_index, {})
        context: list[dict[str, Any]] = []
        for offset in range(-context_radius, context_radius + 1):
            context_frame = frame_index + offset
            if context_frame < 0:
                context.append(
                    {
                        "offset": offset,
                        "frame_index": context_frame,
                        "missing": True,
                        "image_url": "",
                    }
                )
                continue
            image_path = load_or_make_context_crop(
                audit_dir=audit_dir,
                crop_dir=crop_dir,
                source_image_dir=source_image_dir,
                crop_box=crop_box,
                target_frame=frame_index,
                context_frame=context_frame,
                offset=offset,
            )
            if image_path is None:
                context.append(
                    {
                        "offset": offset,
                        "frame_index": context_frame,
                        "missing": True,
                        "image_url": "",
                    }
                )
                continue
            image_id = image_id_for(frame_index, offset)
            image_paths[image_id] = image_path
            context.append(
                {
                    "offset": offset,
                    "frame_index": context_frame,
                    "missing": False,
                    "image_url": f"/image/{image_id}",
                }
            )

        review_row = review_by_frame.get(frame_index, {})
        frames.append(
            {
                "ordinal": ordinal,
                "frame_index": frame_index,
                "primary_bucket": row.get("primary_bucket", ""),
                "context": context,
                "original_particles": reference_row.get("original_particles", []),
                "review": review_row,
            }
        )

    payload = {
        "audit_dir": str(audit_dir.as_posix()),
        "review_path": str(review_path.as_posix()),
        "crop_size": {"width": crop_width, "height": crop_height},
        "context_radius": context_radius,
        "review_data": review_data,
        "frames": frames,
    }
    return AuditServerState(payload=payload, image_paths=image_paths, review_path=review_path)


def merge_frame_review(review_data: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    frame_index = int(incoming["frame_index"])
    frames = review_data.setdefault("frames", [])
    for idx, row in enumerate(frames):
        if int(row["frame_index"]) == frame_index:
            merged = dict(row)
            break
    else:
        idx = len(frames)
        merged = {"frame_index": frame_index}
        frames.append(merged)

    allowed = {
        "primary_bucket",
        "review_status",
        "accept_existing_boxes",
        "mistake_points",
        "notes",
        "reviewer_id",
    }
    for key in allowed:
        if key in incoming:
            merged[key] = incoming[key]
    merged["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    frames[idx] = merged
    update_review_status(review_data)
    return merged


def make_handler(state: AuditServerState) -> type[BaseHTTPRequestHandler]:
    class AuditReviewHandler(BaseHTTPRequestHandler):
        server_version = "BeltMapAuditReview/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_bytes(
            self,
            body: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, value: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(value).encode("utf-8"),
                content_type="application/json; charset=utf-8",
                status=status,
            )

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(
                    REVIEW_HTML.encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                )
                return
            if parsed.path == "/api/state":
                self._send_json(state.payload)
                return
            if parsed.path.startswith("/image/"):
                image_id = parsed.path.removeprefix("/image/")
                image_path = state.image_paths.get(image_id)
                if image_path is None or not image_path.is_file():
                    self._send_json({"error": "image not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                self._send_bytes(image_path.read_bytes(), content_type=content_type)
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/review":
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                incoming = json.loads(self.rfile.read(length).decode("utf-8"))
                review_data = state.payload["review_data"]
                merged = merge_frame_review(review_data, incoming)
                for frame in state.payload["frames"]:
                    if int(frame["frame_index"]) == int(merged["frame_index"]):
                        frame["review"] = merged
                        break
                write_json_atomic(state.review_path, review_data)
            except Exception as exc:  # pragma: no cover - browser-facing fallback
                self._send_json(
                    {"error": f"could not save review: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json({"ok": True, "review": merged, "review_data": review_data})

    return AuditReviewHandler


REVIEW_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BeltMap Annotation Audit Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f4;
      --panel: #ffffff;
      --ink: #1b1d1f;
      --muted: #606a73;
      --line: #cfd5d9;
      --accent: #0f766e;
      --warn: #c2410c;
      --bad: #be123c;
      --ok: #15803d;
      --target: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    #app {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #ecefed;
      padding: 14px;
      overflow: auto;
    }
    main {
      padding: 16px;
      overflow: auto;
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 20px; line-height: 1.2; }
    h2 { font-size: 18px; }
    .muted { color: var(--muted); }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    .frame-list {
      display: grid;
      gap: 6px;
      margin-top: 14px;
    }
    button, select, textarea, input {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }
    button:hover { border-color: #8c969d; }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    button.warn {
      background: var(--warn);
      color: white;
      border-color: var(--warn);
    }
    button.ghost {
      background: transparent;
    }
    .frame-button {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      width: 100%;
      text-align: left;
    }
    .frame-button.active {
      border-color: var(--target);
      outline: 2px solid rgba(37, 99, 235, 0.2);
    }
    .status {
      min-width: 18px;
      height: 18px;
      border-radius: 50%;
      border: 1px solid var(--line);
      background: #f8fafc;
    }
    .status.checked_ok { background: var(--ok); border-color: var(--ok); }
    .status.needs_correction { background: var(--bad); border-color: var(--bad); }
    .progress {
      height: 8px;
      background: #dbe1e5;
      border-radius: 99px;
      margin-top: 10px;
      overflow: hidden;
    }
    .progress > div {
      height: 100%;
      width: 0;
      background: var(--accent);
    }
    .context-grid {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) minmax(120px, 1fr) minmax(320px, 2.4fr) minmax(120px, 1fr) minmax(120px, 1fr);
      gap: 12px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-width: 0;
    }
    .panel.target {
      border-color: var(--target);
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .image-wrap {
      position: relative;
      background: #111;
      border-radius: 4px;
      overflow: hidden;
      min-height: 120px;
    }
    .image-wrap img {
      display: block;
      width: 100%;
      height: auto;
      user-select: none;
    }
    .image-wrap svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }
    .missing {
      min-height: 180px;
      display: grid;
      place-items: center;
      color: #cbd5e1;
      background: #1f2937;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(240px, 1fr) minmax(240px, 1fr);
      gap: 12px;
      margin-top: 12px;
    }
    .control-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
    }
    textarea {
      width: 100%;
      min-height: 82px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: white;
      color: var(--ink);
    }
    .radio-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    label.option {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #f8fafc;
      cursor: pointer;
    }
    label.option input { margin-right: 6px; }
    .issue-list {
      display: grid;
      gap: 6px;
      margin-top: 8px;
      max-height: 180px;
      overflow: auto;
      font-size: 13px;
    }
    .issue {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 6px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
    }
    @media (max-width: 1100px) {
      #app { grid-template-columns: 1fr; }
      aside { max-height: 280px; border-right: 0; border-bottom: 1px solid var(--line); }
      .context-grid { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div id="app">
    <aside>
      <h1>Annotation Audit</h1>
      <p class="muted" id="reviewPath"></p>
      <div class="progress"><div id="progressBar"></div></div>
      <p class="muted" id="progressText"></p>
      <div class="frame-list" id="frameList"></div>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h2 id="frameTitle">Loading</h2>
          <p class="muted" id="frameSubtitle"></p>
        </div>
        <div class="control-row">
          <button class="ghost" id="prevBtn">Previous</button>
          <button class="ghost" id="nextBtn">Next</button>
        </div>
      </div>
      <div class="context-grid" id="contextGrid"></div>
      <div class="controls">
        <section class="panel">
          <h3>Click Flag</h3>
          <div class="radio-row">
            <label class="option"><input type="radio" name="kind" value="false_box" checked>False box</label>
            <label class="option"><input type="radio" name="kind" value="missed_particle">Missed particle</label>
            <label class="option"><input type="radio" name="kind" value="adjust_box">Adjust box</label>
            <label class="option"><input type="radio" name="kind" value="unclear">Unclear</label>
          </div>
          <div class="control-row">
            <button id="undoBtn">Undo point</button>
            <button id="clearBtn">Clear points</button>
          </div>
          <div class="issue-list" id="issueList"></div>
        </section>
        <section class="panel">
          <h3>Decision</h3>
          <textarea id="notes" placeholder="Notes"></textarea>
          <div class="control-row">
            <input id="reviewer" placeholder="Reviewer ID">
            <button class="primary" id="confirmBtn">Confirm checked</button>
            <button class="warn" id="mistakeBtn">Needs correction</button>
            <button id="saveBtn">Save</button>
          </div>
        </section>
      </div>
    </main>
  </div>
  <script>
    let appState = null;
    let currentIndex = 0;
    let draft = null;

    function statusOf(frame) {
      return frame.review?.review_status || "unreviewed";
    }

    function cloneReview(frame) {
      return {
        frame_index: frame.frame_index,
        primary_bucket: frame.primary_bucket,
        review_status: frame.review?.review_status || "unreviewed",
        accept_existing_boxes: frame.review?.accept_existing_boxes ?? null,
        mistake_points: [...(frame.review?.mistake_points || [])],
        notes: frame.review?.notes || "",
        reviewer_id: frame.review?.reviewer_id || ""
      };
    }

    function kindLabel(kind) {
      return {
        false_box: "False box",
        missed_particle: "Missed particle",
        adjust_box: "Adjust box",
        unclear: "Unclear"
      }[kind] || kind;
    }

    function renderSidebar() {
      const list = document.getElementById("frameList");
      list.innerHTML = "";
      let done = 0;
      appState.frames.forEach((frame, idx) => {
        const status = statusOf(frame);
        if (status !== "unreviewed") done += 1;
        const btn = document.createElement("button");
        btn.className = "frame-button" + (idx === currentIndex ? " active" : "");
        btn.innerHTML = `<span>${String(frame.frame_index).padStart(6, "0")}<br><small>${frame.primary_bucket}</small></span><span class="status ${status}"></span>`;
        btn.addEventListener("click", () => {
          currentIndex = idx;
          renderCurrent();
        });
        list.appendChild(btn);
      });
      document.getElementById("progressBar").style.width = `${100 * done / appState.frames.length}%`;
      document.getElementById("progressText").textContent = `${done} / ${appState.frames.length} reviewed`;
    }

    function renderContextPanel(context, frame, isTarget) {
      const panel = document.createElement("section");
      panel.className = "panel" + (isTarget ? " target" : "");
      const rel = context.offset === 0 ? "target" : `t${context.offset > 0 ? "+" : ""}${context.offset}`;
      panel.innerHTML = `<div class="panel-header"><strong>${rel}</strong><span>frame ${String(context.frame_index).padStart(6, "0")}</span></div>`;
      const wrap = document.createElement("div");
      wrap.className = "image-wrap";
      if (context.missing) {
        wrap.innerHTML = `<div class="missing">missing context</div>`;
        panel.appendChild(wrap);
        return panel;
      }
      const img = document.createElement("img");
      img.src = context.image_url;
      img.alt = `${rel} frame ${context.frame_index}`;
      wrap.appendChild(img);
      if (isTarget) {
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", `0 0 ${appState.crop_size.width} ${appState.crop_size.height}`);
        svg.addEventListener("click", onTargetClick);
        for (const box of frame.original_particles) {
          const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          rect.setAttribute("x", box.left);
          rect.setAttribute("y", box.top);
          rect.setAttribute("width", box.right - box.left);
          rect.setAttribute("height", box.bottom - box.top);
          rect.setAttribute("fill", "none");
          rect.setAttribute("stroke", "#22c55e");
          rect.setAttribute("stroke-width", "3");
          rect.setAttribute("vector-effect", "non-scaling-stroke");
          svg.appendChild(rect);
        }
        for (const point of draft.mistake_points) {
          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", point.x);
          circle.setAttribute("cy", point.y);
          circle.setAttribute("r", "10");
          circle.setAttribute("fill", point.kind === "missed_particle" ? "#f59e0b" : "#e11d48");
          circle.setAttribute("stroke", "white");
          circle.setAttribute("stroke-width", "3");
          circle.setAttribute("vector-effect", "non-scaling-stroke");
          svg.appendChild(circle);
        }
        wrap.appendChild(svg);
      }
      panel.appendChild(wrap);
      return panel;
    }

    function renderIssues() {
      const list = document.getElementById("issueList");
      list.innerHTML = "";
      if (!draft.mistake_points.length) {
        list.innerHTML = `<p class="muted">No click flags.</p>`;
        return;
      }
      draft.mistake_points.forEach((point, idx) => {
        const row = document.createElement("div");
        row.className = "issue";
        row.innerHTML = `<span>${idx + 1}. ${kindLabel(point.kind)} at ${Math.round(point.x)}, ${Math.round(point.y)}</span><button data-idx="${idx}">Remove</button>`;
        row.querySelector("button").addEventListener("click", () => {
          draft.mistake_points.splice(idx, 1);
          renderCurrent(false);
          saveDraft();
        });
        list.appendChild(row);
      });
    }

    function renderCurrent(resetDraft = true) {
      const frame = appState.frames[currentIndex];
      if (resetDraft) draft = cloneReview(frame);
      document.getElementById("frameTitle").textContent = `Frame ${String(frame.frame_index).padStart(6, "0")}`;
      document.getElementById("frameSubtitle").textContent = `${frame.primary_bucket} | ${frame.original_particles.length} original boxes`;
      document.getElementById("notes").value = draft.notes;
      document.getElementById("reviewer").value = draft.reviewer_id;
      const grid = document.getElementById("contextGrid");
      grid.innerHTML = "";
      frame.context.forEach((context) => {
        grid.appendChild(renderContextPanel(context, frame, context.offset === 0));
      });
      renderIssues();
      renderSidebar();
    }

    function onTargetClick(event) {
      const svg = event.currentTarget;
      const rect = svg.getBoundingClientRect();
      const x = (event.clientX - rect.left) * appState.crop_size.width / rect.width;
      const y = (event.clientY - rect.top) * appState.crop_size.height / rect.height;
      const kind = document.querySelector("input[name=kind]:checked").value;
      draft.mistake_points.push({x, y, kind});
      draft.review_status = "needs_correction";
      draft.accept_existing_boxes = false;
      renderCurrent(false);
      saveDraft();
    }

    async function saveDraft() {
      draft.notes = document.getElementById("notes").value;
      draft.reviewer_id = document.getElementById("reviewer").value;
      const response = await fetch("/api/review", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(draft)
      });
      const result = await response.json();
      if (!result.ok) throw new Error(result.error || "Save failed");
      const frame = appState.frames[currentIndex];
      frame.review = result.review;
      appState.review_data = result.review_data;
      renderSidebar();
    }

    function move(delta) {
      currentIndex = Math.max(0, Math.min(appState.frames.length - 1, currentIndex + delta));
      renderCurrent();
    }

    async function init() {
      const response = await fetch("/api/state");
      appState = await response.json();
      document.getElementById("reviewPath").textContent = appState.review_path;
      document.getElementById("prevBtn").addEventListener("click", () => move(-1));
      document.getElementById("nextBtn").addEventListener("click", () => move(1));
      document.getElementById("undoBtn").addEventListener("click", () => {
        draft.mistake_points.pop();
        if (!draft.mistake_points.length && draft.review_status === "needs_correction") {
          draft.review_status = "unreviewed";
          draft.accept_existing_boxes = null;
        }
        renderCurrent(false);
        saveDraft();
      });
      document.getElementById("clearBtn").addEventListener("click", () => {
        draft.mistake_points = [];
        draft.review_status = "unreviewed";
        draft.accept_existing_boxes = null;
        renderCurrent(false);
        saveDraft();
      });
      document.getElementById("confirmBtn").addEventListener("click", () => {
        if (draft.mistake_points.length) {
          alert("Remove click flags before confirming as checked.");
          return;
        }
        draft.review_status = "checked_ok";
        draft.accept_existing_boxes = true;
        saveDraft();
      });
      document.getElementById("mistakeBtn").addEventListener("click", () => {
        draft.review_status = "needs_correction";
        draft.accept_existing_boxes = false;
        saveDraft();
      });
      document.getElementById("saveBtn").addEventListener("click", saveDraft);
      document.getElementById("notes").addEventListener("change", () => { draft.notes = document.getElementById("notes").value; });
      document.getElementById("reviewer").addEventListener("change", () => { draft.reviewer_id = document.getElementById("reviewer").value; });
      document.addEventListener("keydown", (event) => {
        if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT") return;
        if (event.key === "ArrowLeft") move(-1);
        if (event.key === "ArrowRight") move(1);
      });
      renderCurrent();
    }
    init();
  </script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve an interactive annotation-audit review UI with temporal context.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/brick20g_annotation_audit_pack"),
        help="Audit pack created by scripts/brick20g_annotation_audit_pack.py.",
    )
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=Path("outputs/brick20g_adversarial_review_pack/images"),
        help="Crop directory used as a fallback for context frames.",
    )
    parser.add_argument(
        "--source-image-dir",
        type=Path,
        help="Optional original image root for generating missing before/after crops.",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        help="Path where click-review decisions are saved.",
    )
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--print-state-summary",
        action="store_true",
        help="Build the review state, print a small JSON summary, and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_dir = args.audit_dir
    review_path = args.review_json or audit_dir / "audit_click_review.json"
    state = build_payload(
        audit_dir=audit_dir,
        crop_dir=args.crop_dir,
        source_image_dir=args.source_image_dir,
        review_path=review_path,
        context_radius=args.context_radius,
    )
    if args.print_state_summary:
        frames = state.payload["frames"]
        missing_context = sum(
            1
            for frame in frames
            for context in frame["context"]
            if context.get("missing")
        )
        print(
            json.dumps(
                {
                    "frames": len(frames),
                    "images": len(state.image_paths),
                    "missing_context_panels": missing_context,
                    "review_path": str(review_path.as_posix()),
                },
                indent=2,
            )
        )
        return 0

    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving annotation audit review at {url}")
    print(f"Saving review decisions to {review_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped annotation audit review server")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
