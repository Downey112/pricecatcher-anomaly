"""Weekly brief for inspectors: a short Markdown summary of the week's flagged prices.

`template_brief` needs no API. `llm_brief` asks an LLM for a more readable summary, then checks
it: every number in the text must appear in the flagged rows, and every row reference [#n] must
exist. If the check fails, the template brief is used instead. The LLM never decides what is
flagged; it only describes rows the statistics already flagged.
"""
from __future__ import annotations

import os
import re

import pandas as pd

TOP_N = 15
PROMPT = """You write a weekly brief for Malaysian price inspectors. Below are prices flagged as \
unusually high this week (numbered rows). Write at most 8 bullet points: group by state or item \
where it helps, name the premise and item, and cite rows as [#n]. Use ONLY numbers that appear in \
the rows (prices in RM, ratios, counts). Do not explain causes beyond the 'likely_cause' column. \
A flag is a reason to check, not proof of profiteering; say so once."""


def top_rows(flags: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """The n strongest flags (lowest p-value, then highest score), numbered from 1."""
    rows = flags.sort_values(["p_value", "score"], ascending=[True, False]).head(n).reset_index(drop=True)
    rows.index = rows.index + 1
    return rows


def rows_table(rows: pd.DataFrame) -> str:
    lines = ["| # | State | District | Premise | Item | Price (RM) | Reference (RM) | x ref | p-value | Likely cause |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in rows.iterrows():
        lines.append(f"| {i} | {r['state']} | {r['district']} | {r['premise']} | {r['item']} ({r['unit']}) | "
                     f"{r['price']:.2f} | {r['ref_price']:.2f} | {r['ratio_to_ref']:.2f} | {r['p_value']:.1e} | "
                     f"{r['likely_cause']} |")
    return "\n".join(lines)


def template_brief(flags: pd.DataFrame, week: str, alpha: float, n_rows: int) -> str:
    rows = top_rows(flags)
    by_state = flags["state"].value_counts().head(5)
    states = ", ".join(f"{s} ({c})" for s, c in by_state.items())
    return "\n".join([
        f"# Price-check brief: week of {week}",
        "",
        f"{len(flags)} of {n_rows:,} prices were flagged at conformal p <= {alpha} "
        f"(a normal price is flagged at most {alpha:.2%} of the time). Most flags: {states}.",
        "A flag is a reason to check a price, not proof of profiteering. Pack or unit mismatches are common.",
        "",
        f"## Strongest {len(rows)} flags",
        "",
        rows_table(rows),
        "",
    ])


def numbers_in(text: str) -> set[str]:
    """Numbers as written, normalised: '1,234.50' -> '1234.5', '29.90' -> '29.9'."""
    out = set()
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        value = token.replace(",", "")
        out.add(value.rstrip("0").rstrip(".") if "." in value else value)
    return out


def check_grounded(text: str, rows: pd.DataFrame, allowed_extra: set[str]) -> list[str]:
    """Problems with an LLM brief: numbers not found in the rows, or row references that don't exist."""
    problems = []
    refs = {int(n) for n in re.findall(r"\[#(\d+)\]", text)}
    if bad := sorted(refs - set(rows.index)):
        problems.append(f"cites rows that do not exist: {bad}")
    source = rows_table(rows) + " " + " ".join(sorted(allowed_extra))
    known = numbers_in(source) | {str(i) for i in rows.index}
    text_wo_refs = re.sub(r"\[#\d+\]", "", text)
    if unknown := sorted(numbers_in(text_wo_refs) - known):
        problems.append(f"numbers not in the flagged rows: {unknown[:10]}")
    return problems


def llm_brief(flags: pd.DataFrame, week: str, alpha: float, n_rows: int, client=None,
              model: str = "openai/gpt-oss-120b") -> tuple[str, list[str]]:
    """(brief, problems). Falls back to the template brief when the check finds problems."""
    rows = top_rows(flags)
    if client is None:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
    header = f"Week of {week}. {len(flags)} of {n_rows} prices flagged at p <= {alpha}."
    response = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": header + "\n\n" + rows_table(rows)}],
    )
    text = (response.choices[0].message.content or "").strip()
    problems = check_grounded(text, rows, allowed_extra=numbers_in(header) | {week})
    if problems or not text:
        return template_brief(flags, week, alpha, n_rows), problems or ["empty reply"]
    return f"# Price-check brief: week of {week}\n\n{text}\n\n## Flagged rows\n\n{rows_table(rows)}\n", []
