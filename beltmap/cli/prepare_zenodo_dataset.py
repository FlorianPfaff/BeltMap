from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse
from urllib.request import url2pathname
from zipfile import ZipFile, ZipInfo

CHUNK_SIZE = 1024 * 1024
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "datasets"
ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"
ZENODO_RECORD_FILE_URL = (
    "https://zenodo.org/records/{record_id}/files/{filename}?download=1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beltmap-prepare-zenodo",
        description=(
            "Download or reuse a cached Zenodo image archive, extract it "
            "atomically, and expose it through data/images."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Zenodo file URL or local zip path.")
    source.add_argument(
        "--record-id",
        help="Zenodo record ID containing the zip file.",
    )
    parser.add_argument(
        "--record-file-glob",
        default="*.zip",
        help="Glob used with --record-id to select the archive file. Default: *.zip",
    )
    parser.add_argument(
        "--record-file-name",
        help="Exact file name used with --record-id; overrides --record-file-glob.",
    )
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
    candidate = Path(url).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    parsed = urlparse(url)
    if parsed.scheme == "":
        return candidate
    if parsed.scheme == "file":
        return Path(url2pathname(unquote(parsed.path))).expanduser()
    return None


def request_url(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": "BeltMap dataset preparation"},
    )


def fetch_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(request_url(url), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def record_file_name(file_info: object) -> str:
    if not isinstance(file_info, dict):
        raise ValueError(f"invalid Zenodo file metadata entry: {file_info!r}")
    key = file_info.get("key")
    if isinstance(key, str) and key:
        return key
    filename = file_info.get("filename")
    if isinstance(filename, str) and filename:
        return filename
    raise ValueError(f"Zenodo file metadata entry has no file name: {file_info!r}")


def record_file_size(file_info: object) -> int:
    if not isinstance(file_info, dict):
        return 0
    size = file_info.get("size")
    if isinstance(size, int):
        return size
    if isinstance(size, str) and size.isdigit():
        return int(size)
    return 0


def select_record_file(
    metadata: dict[str, object],
    *,
    file_name: str | None,
    file_glob: str,
) -> str:
    files = metadata.get("files")
    if not isinstance(files, list):
        raise ValueError("Zenodo record metadata does not contain a files list")

    if file_name:
        matches = [
            file_info
            for file_info in files
            if record_file_name(file_info) == file_name
        ]
    else:
        matches = [
            file_info
            for file_info in files
            if fnmatch.fnmatch(record_file_name(file_info), file_glob)
        ]

    if not matches:
        selector = file_name if file_name else file_glob
        available = ", ".join(record_file_name(file_info) for file_info in files)
        raise ValueError(
            f"Zenodo record file selector {selector!r} matched no files. "
            f"Available files: {available}"
        )
    if len(matches) > 1:
        matches = sorted(matches, key=record_file_size, reverse=True)
        largest_size = record_file_size(matches[0])
        if sum(record_file_size(match) == largest_size for match in matches) > 1:
            names = ", ".join(record_file_name(file_info) for file_info in matches)
            raise ValueError(
                "Zenodo record file selector matched multiple same-size files: "
                f"{names}; pass --record-file-name"
            )
    return record_file_name(matches[0])


def resolve_source_url(
    *,
    url: str | None,
    record_id: str | None,
    record_file_name_arg: str | None,
    record_file_glob: str,
) -> tuple[str, str | None]:
    if url is not None:
        return url, None
    if record_id is None:
        raise ValueError("either --url or --record-id is required")

    api_url = ZENODO_RECORD_API.format(record_id=record_id)
    metadata = fetch_json(api_url)
    file_name = select_record_file(
        metadata,
        file_name=record_file_name_arg,
        file_glob=record_file_glob,
    )
    source_url = ZENODO_RECORD_FILE_URL.format(
        record_id=record_id,
        filename=quote(file_name, safe=""),
    )
    return source_url, file_name


def download_or_copy(url: str, destination: Path) -> None:
    source = local_source_path(url)
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(f"local dataset zip not found: {source}")
        shutil.copy2(source, destination)
        return

    with urllib.request.urlopen(request_url(url), timeout=60) as response:
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


def normalized_member_path(member: ZipInfo) -> PurePosixPath:
    name = member.filename.replace("\\", "/")
    if not name or "\x00" in name:
        raise ValueError("dataset archive contains an empty or invalid path")
    member_path = PurePosixPath(name)
    if member_path.is_absolute() or any(
        part in {"", ".", ".."} for part in member_path.parts
    ):
        raise ValueError(f"dataset archive contains unsafe path: {member.filename}")
    return member_path


def safe_extract_zip(archive: ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        member_path = normalized_member_path(member)
        target = (destination / Path(*member_path.parts)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"dataset archive contains unsafe path: {member.filename}"
            ) from exc

        if member.is_dir() or member.filename.endswith(("/", "\\")):
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, CHUNK_SIZE)


def extract_cached_zip(*, cache_zip: Path, cache_images: Path) -> None:
    cache_tmp = cache_images.with_name(cache_images.name + ".tmp")
    remove_path(cache_tmp)
    cache_tmp.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(cache_zip) as archive:
            safe_extract_zip(archive, cache_tmp)
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
    source_record_id: str | None,
    source_record_file: str | None,
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
        "source_record_id": source_record_id,
        "source_record_file": source_record_file,
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
        source_url, source_record_file = resolve_source_url(
            url=args.url,
            record_id=args.record_id,
            record_file_name_arg=args.record_file_name,
            record_file_glob=args.record_file_glob,
        )
        cache_root = args.cache_root.expanduser()
        cache_images = cache_root / dataset_name
        cache_zip = cache_root / f"{dataset_name}.zip"
        image_link = args.image_link
        zip_link = args.zip_link

        print(f"Dataset name: {dataset_name}")
        print(f"Source URL: {source_url}")
        if args.record_id is not None:
            print(f"Zenodo record ID: {args.record_id}")
            print(f"Zenodo record file: {source_record_file}")
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
                url=source_url,
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
                source_url=source_url,
                source_record_id=args.record_id,
                source_record_file=source_record_file,
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
