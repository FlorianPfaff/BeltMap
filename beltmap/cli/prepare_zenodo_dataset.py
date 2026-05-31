from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse
from zipfile import ZipFile

CHUNK_SIZE = 1024 * 1024
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "datasets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-prepare-zenodo",
        description=(
            "Download or reuse a cached Zenodo image archive, extract it "
            "atomically, and expose it through data/images."
        ),
    )
    parser.add_argument("--url", required=True, help="Zenodo file URL or local zip path.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Cache stem for the extracted dataset and zip file.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Dataset cache root. Default: {DEFAULT_CACHE_ROOT}",
    )
    parser.add_argument(
        "--image-link",
        type=Path,
        default=Path("data/images"),
        help="Path that should point at the extracted image directory.",
    )
    parser.add_argument(
        "--zip-link",
        type=Path,
        help="Optional path that should point at the cached zip archive.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Optional JSON path for dataset preparation metadata.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download/copy the zip again even if a cached zip exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract the zip again even if an extracted cache exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved cache paths without downloading or linking.",
    )
    return parser


def validate_dataset_name(name: str) -> str:
    cleaned = name.strip()
    if cleaned in {"", ".", ".."}:
        raise ValueError("dataset name must not be empty, '.' or '..'")
    if Path(cleaned).name != cleaned:
        raise ValueError("dataset name must be a single path component")
    return cleaned


def directory_has_files(path: Path) -> bool:
    return path.exists() and any(candidate.is_file() for candidate in path.rglob("*"))


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def local_source_path(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme == "":
        return Path(url).expanduser()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser()
    return None


def download_or_copy(url: str, destination: Path) -> None:
    source = local_source_path(url)
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(f"local dataset zip not found: {source}")
        shutil.copy2(source, destination)
        return

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BeltMap dataset preparation"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, CHUNK_SIZE)


def ensure_cached_zip(
    *,
    url: str,
    cache_zip: Path,
    force_download: bool,
) -> None:
    if cache_zip.is_file() and cache_zip.stat().st_size > 0 and not force_download:
        print(f"Local dataset zip cache hit: {cache_zip}")
        return

    print(f"Local dataset zip cache miss; storing archive at {cache_zip}")
    cache_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = cache_zip.with_suffix(cache_zip.suffix + ".tmp")
    remove_path(tmp_zip)
    try:
        download_or_copy(url, tmp_zip)
        if tmp_zip.stat().st_size == 0:
            raise ValueError(f"downloaded empty dataset zip: {tmp_zip}")
        tmp_zip.replace(cache_zip)
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()


def extract_cached_zip(*, cache_zip: Path, cache_images: Path) -> None:
    cache_tmp = cache_images.with_name(cache_images.name + ".tmp")
    remove_path(cache_tmp)
    cache_tmp.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(cache_zip) as archive:
            archive.extractall(cache_tmp)
        if not directory_has_files(cache_tmp):
            raise ValueError(f"dataset archive did not contain files: {cache_zip}")
        remove_path(cache_images)
        cache_tmp.replace(cache_images)
    finally:
        if cache_tmp.exists():
            shutil.rmtree(cache_tmp)


def expose_path(*, target: Path, link: Path, target_is_directory: bool) -> None:
    remove_path(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        if target_is_directory:
            shutil.copytree(target, link)
        else:
            shutil.copy2(target, link)


def tree_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            file_count += 1
            byte_count += candidate.stat().st_size
    return file_count, byte_count


def write_manifest(
    *,
    manifest_path: Path,
    source_url: str,
    dataset_name: str,
    cache_images: Path,
    cache_zip: Path,
    image_link: Path,
    zip_link: Path | None,
) -> None:
    file_count, byte_count = tree_stats(cache_images)
    manifest = {
        "dataset_name": dataset_name,
        "source_url": source_url,
        "cache_images": str(cache_images),
        "cache_zip": str(cache_zip),
        "image_link": str(image_link),
        "zip_link": str(zip_link) if zip_link is not None else None,
        "file_count": file_count,
        "extracted_bytes": byte_count,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset_name = validate_dataset_name(args.dataset_name)
        cache_root = args.cache_root.expanduser()
        cache_images = cache_root / dataset_name
        cache_zip = cache_root / f"{dataset_name}.zip"
        image_link = args.image_link
        zip_link = args.zip_link

        print(f"Dataset name: {dataset_name}")
        print(f"Source URL: {args.url}")
        print(f"Cache root: {cache_root}")
        print(f"Extracted cache: {cache_images}")
        print(f"Zip cache: {cache_zip}")
        print(f"Image link: {image_link}")
        if zip_link is not None:
            print(f"Zip link: {zip_link}")
        if args.dry_run:
            return 0

        cache_root.mkdir(parents=True, exist_ok=True)
        if directory_has_files(cache_images) and not args.force_extract:
            print(f"Local extracted dataset cache hit: {cache_images}")
        else:
            print(f"Local extracted dataset cache miss or empty: {cache_images}")
            ensure_cached_zip(
                url=args.url,
                cache_zip=cache_zip,
                force_download=args.force_download,
            )
            print(f"Extracting dataset atomically to {cache_images}")
            extract_cached_zip(cache_zip=cache_zip, cache_images=cache_images)

        expose_path(target=cache_images, link=image_link, target_is_directory=True)
        if zip_link is not None and cache_zip.is_file():
            expose_path(target=cache_zip, link=zip_link, target_is_directory=False)

        file_count, byte_count = tree_stats(cache_images)
        print(f"Extracted image/cache files: {file_count}")
        print(f"Extracted byte count: {byte_count}")
        if args.manifest_path is not None:
            write_manifest(
                manifest_path=args.manifest_path,
                source_url=args.url,
                dataset_name=dataset_name,
                cache_images=cache_images,
                cache_zip=cache_zip,
                image_link=image_link,
                zip_link=zip_link,
            )
        return 0
    except Exception as exc:  # pragma: no cover - keeps workflow failures readable.
        print(f"beltmap-prepare-zenodo: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
