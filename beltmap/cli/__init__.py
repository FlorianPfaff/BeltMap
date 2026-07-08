"""Command-line interfaces for BeltMap."""

# Import for side effect: make annotation-audit context lookup honor all image
# formats supported by the BeltMap image/YOLO export path.
from . import annotation_audit_review_source_image_patch as _annotation_audit_review_source_image_patch  # noqa: F401,E402
