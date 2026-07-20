import csv

import pytest

from beltmap.cli.filter_revolution_recurrence import accepted_track_ids, bool_value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (" TRUE ", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_runtime_acceptance_parser_accepts_explicit_boolean_tokens(value, expected):
    assert bool_value(value) is expected


@pytest.mark.parametrize("value", ["", "treu", "2", None])
def test_runtime_acceptance_parser_rejects_unrecognized_values(value):
    with pytest.raises(ValueError, match="accepted must be one of"):
        bool_value(value)


def test_accepted_track_ids_rejects_malformed_acceptance_instead_of_dropping_track(
    tmp_path,
):
    with (tmp_path / "track_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "accepted"])
        writer.writeheader()
        writer.writerow({"track_id": 7, "accepted": "treu"})

    with pytest.raises(ValueError, match="accepted must be one of"):
        accepted_track_ids(tmp_path)
