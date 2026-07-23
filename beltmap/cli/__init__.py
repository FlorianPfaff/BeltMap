"""Command-line interfaces for BeltMap."""

# Import for side effect: make annotation-audit context lookup honor all image
# formats supported by the BeltMap image/YOLO export path.
from . import annotation_audit_review_source_image_patch as _annotation_audit_review_source_image_patch  # noqa: F401,E402

# Import for side effect: keep prepare-zenodo links valid when --cache-root is
# supplied as a relative path.
from . import prepare_zenodo_relative_cache_patch as _prepare_zenodo_relative_cache_patch  # noqa: F401,E402

# Import for side effect: reconstruct fallback track membership directly from
# observed absolute frame IDs instead of allocating every preceding frame.
from . import filter_tracks_sparse_frame_patch as _filter_tracks_sparse_frame_patch  # noqa: F401,E402

# Import for side effect: reject fractional --belt-region coordinates instead
# of silently truncating them in the YOLO recurrence CLI.
from .. import yolo_recurrence_region_patch as _yolo_recurrence_region_patch  # noqa: F401,E402
