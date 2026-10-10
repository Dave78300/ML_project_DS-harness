import math

import pandas as pd
import pytest

from ml_project_ds_harness import stage3


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ([], "nofile"),
        (["data/train.csv", "data/test.csv"], "fit"),
        (["data/TRAIN.CSV"], "fit"),
        (["data/sales.csv"], "table"),
    ],
)
def test_route(files, expected):
    assert stage3.route(files) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 234,5 kg", 1234.5),
        ("1,234.5", 1234.5),
        ("-2,5 kg", -2.5),
    ],
)
def test_num_from_text_parses_localized_numbers(text, expected):
    assert stage3._num_from_text(text) == pytest.approx(expected)


def test_num_from_text_returns_nan_for_non_numeric_text():
    assert math.isnan(stage3._num_from_text("n/a"))


def test_read_table_detects_separator_converts_numbers_and_missing_values(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    csv_path.write_text(
        "name;weight\n"
        "Alice;1 234,5 kg\n"
        "Bob;2,5 kg\n"
        "Cara;NA\n",
        encoding="utf-8",
    )

    table = stage3.read_table(csv_path)

    assert list(table.columns) == ["name", "weight"]
    assert table.loc[0, "weight"] == pytest.approx(1234.5)
    assert table.loc[1, "weight"] == pytest.approx(2.5)
    assert pd.isna(table.loc[2, "weight"])


def test_extract_code_removes_trailing_code_fence():
    assert stage3.extract_code("answer = 42\n```\nextra text") == "answer = 42"


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("Round the result to 2 decimals.", 2),
        ("Give the nearest integer.", 0),
        ("Calculate the result.", None),
    ],
)
def test_decimals_detects_requested_precision(objective, expected):
    assert stage3.decimals(objective) == expected


def test_vote_returns_value_from_heaviest_rounded_group():
    candidates = [(1.1, 1.0), (1.12, 2.0), (9.0, 1.0)]

    assert stage3.vote(candidates, nd=1) == pytest.approx(1.1)


def test_calc_rules_solves_least_common_multiple_without_model():
    objective = "What is the least common multiple of 6 and 8?"

    assert stage3.calc_rules(objective) == pytest.approx(24.0)
