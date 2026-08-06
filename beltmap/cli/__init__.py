"""Command-line interfaces for BeltMap."""

# Import for side effect: keep recursive CLI image discovery restricted to regular
# files so image-suffixed directories cannot consume frame limits or reach Pillow.
from .. import image_path_file_patch as _image_path_file_patch  # noqa: F401,E402

# Import for side effect: make annotation-audit context lookup honor all image
# formats supported by the BeltMap image/YOLO export path.
from . import annotation_audit_review_source_image_patch as _annotation_audit_review_source_image_patch  # noqa: F401,E402

# Import for side effect: keep prepare-zenodo links valid when --cache-root is
# supplied as a relative path.
from . import prepare_zenodo_relative_cache_patch as _prepare_zenodo_relative_cache_patch  # noqa: F401,E402

# Import for side effect: reject exposure paths that can delete or recursively
# copy the prepared Zenodo cache into itself.
from . import prepare_zenodo_link_overlap_patch as _prepare_zenodo_link_overlap_patch  # noqa: F401,E402

# Import for side effect: reconstruct fallback track membership directly from
# observed absolute frame IDs instead of allocating every preceding frame.
from . import filter_tracks_sparse_frame_patch as _filter_tracks_sparse_frame_patch  # noqa: F401,E402
