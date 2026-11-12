import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcad.brief import check_grounded, llm_brief, numbers_in, template_brief, top_rows  # noqa: E402

FLAGS = pd.DataFrame({
    "state": ["Johor", "Johor", "Selangor"], "district": ["Batu Pahat", "Muar", "Klang"],
    "premise": ["HYPERMARKET PACIFIC", "KEDAI A", "PASAR B"],
    "item": ["CILI AKAR MERAH", "BAWANG", "AYAM"], "unit": ["1kg", "1kg", "1kg"],
    "price": [29.9, 11.99, 15.0], "ref_price": [9.99, 7.98, 9.5], "ratio_to_ref": [2.99, 1.5, 1.58],
    "p_value": [5e-6, 4e-5, 9e-5], "score": [54.8, 20.4, 18.0],
    "likely_cause": ["pack/unit mismatch?", "persistent premium", "new jump"],
})


class FakeLLM:
    def __init__(self, reply):
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])))


def test_numbers_are_normalised():
    assert numbers_in("RM29.90 and 1,234.50 and 3") == {"29.9", "1234.5", "3"}


def test_template_brief_lists_strongest_first():
    text = template_brief(FLAGS, "2026-10-19", 0.0001, 100_000)
    assert "3 of 100,000 prices" in text and text.index("HYPERMARKET PACIFIC") < text.index("PASAR B")


def test_grounded_brief_passes():
    reply = "- Chilli at HYPERMARKET PACIFIC: RM29.90, 2.99x the RM9.99 reference [#1]."
    text, problems = llm_brief(FLAGS, "2026-10-19", 0.0001, 100_000, client=FakeLLM(reply))
    assert problems == [] and "2.99x" in text


def test_invented_number_falls_back_to_template():
    reply = "- Chilli up 300% to RM35.00 [#1]."
    text, problems = llm_brief(FLAGS, "2026-10-19", 0.0001, 100_000, client=FakeLLM(reply))
    assert problems and "35" in problems[0] and "## Strongest" in text


def test_missing_row_reference_is_caught():
    rows = top_rows(FLAGS)
    assert check_grounded("See [#9].", rows, set())[0].startswith("cites rows that do not exist")
