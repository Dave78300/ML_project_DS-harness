"""Smoke tests for agent/agent_v10.py (the 97.5 % submission). No model is loaded: only the deterministic layers run.

Run with:  python -m pytest -q tests/test_agent_v10.py
"""
import importlib.util
import math
from pathlib import Path

import pytest

AGENT_PATH = Path(__file__).resolve().parents[1] / "agent" / "agent_v10.py"
_spec = importlib.util.spec_from_file_location("agent_v10", AGENT_PATH)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


@pytest.mark.parametrize(
    ("files", "expected"),
    [([], "nofile"), (["data/train.csv", "data/test.csv"], "fit"), (["data/sales.csv"], "table")],
)
def test_route(files, expected):
    assert A.route(files) == expected


def test_submission_defaults():
    assert A.TABLE_MODE == "v1_v2wins"
    assert A.LLM_HINTS and A.SANITY and A.TABLE_SAMPLES == 8
    assert not A.RULES_OFF and not A.HINT_WEIGHT and not A.EXTRA_ROUND and not A.NOFILE_HINTS


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("What is the least common multiple of 6 and 8? Give a whole number.", 24.0),
        ("What is the base-10 logarithm of 2,804? Round to 4 decimals.", math.log10(2804)),
        ("How many days are there from 15 April 2017 to April 28, 2017? Count both the first and the last day. "
         "Give a whole number.", 14.0),
        ("A savings account starts at 5000 euros and earns 4% per year, interest reinvested. How much interest has been "
         "earned after 4 years? Round to 2 decimals.", 5000 * (1.04 ** 4 - 1)),
    ],
)
def test_calc_rules(objective, expected):
    assert A.calc_rules(objective) == pytest.approx(expected)


def test_calc_rules_abstain_on_sub_list_statistic():
    # A statistic of a sub-list is left to the LLM rather than answered with the mean of the whole list.
    objective = "The delays were 10.5, 20.25, 30.0, 40.75 (hours). What is the mean of the delays above 15? Round to 2 decimals."
    assert A.calc_rules(objective) is None


def test_sanity_layer_filters_impossible_llm_answers():
    objective = "Among the loans of the Lyon branch, what fraction (between 0 and 1) have a default flag of '1'? Report it with 3 decimals."
    cands = A._sanitize(objective, [(42.5, 1.5), (42.5, 1.0)])
    assert A.vote(cands, A.decimals(objective)) == pytest.approx(0.425)


def test_table_rule_with_sentinel_duplicates_and_filter(tmp_path):
    (tmp_path / "shipments.csv").write_text(
        "id;w;km\n1;100,5;300\n2;-1;500\n3;200,5;700\n3;200,5;700\n4;400,0;900\n", encoding="utf-8")
    (tmp_path / "data_dictionary.csv").write_text(
        "file,column,description,unit\nshipments.csv,id,Shipment number,\nshipments.csv,w,Weight of the load,kg\n"
        "shipments.csv,km,Distance by road,km\n", encoding="utf-8")
    objective = ("What is the mean load weight of the shipments with a distance of at least 400 km (in kg)? "
                 "shipments.csv uses ';' to separate columns and a comma as the decimal mark. "
                 "A load weight of -1 means it was not recorded. Ignore values that are missing or not recorded. "
                 "Some shipments appear twice in shipments.csv: count each shipment once. Round to 2 decimals.")
    files = [str(tmp_path / "shipments.csv"), str(tmp_path / "data_dictionary.csv")]
    dfs, dico = A.load_task(objective, files)
    assert A.table_filter_rule(objective, dfs, dico) == pytest.approx((200.5 + 400.0) / 2)
