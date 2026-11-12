# Design decisions (be ready to explain these)

## 1. Why the MAD baseline is still the scorer

The v1 study found the robust z-score (distance from the same-week peer median, in units of the
median absolute deviation) beat Isolation Forest and ECOD on planted +20% markups held over
several weeks (0.21 vs 0.09–0.11 recall). It is also explainable to an inspector in one sentence.
v2 keeps it and changes only how the cut-off is chosen.

## 2. Why conformal p-values instead of "top 20 per district"

A fixed budget flags 20 prices per district even in a quiet week, and says nothing about how
unusual they are. A conformal p-value compares each score with a calibration sample:

    p = (1 + number of calibration scores >= this score) / (n + 1)

If the new price behaves like the calibration prices, P(p ≤ α) ≤ α. So α is a stated false-alarm
rate, and the number of flags rises and falls with how unusual the week really is.

**Checked, not assumed:** on validation and test weeks the flag rate was 0.0011 at α = 0.001 and
0.0001 at α = 0.0001 (results/conformal.json).

**Limits:**
- The training weeks include some real anomalies. That makes p-values slightly conservative
  (fewer flags than α would allow), which is the safe direction.
- Exchangeability can break if prices shift (festive seasons, subsidy changes). The flag rate is
  logged every week in MLflow, so drift shows up as a rising rate.
- α controls false alarms per price. With ~200,000 prices a week, α = 0.0001 still allows about
  20 false flags. `benjamini_hochberg` is available for controlling the share of flags that are
  false instead.

## 3. Why the calibration sample is committed

The weekly job only downloads two months, so it cannot see the training weeks. The sorted sample
(200,000 float32 values, 800 KB) is saved once by `04_calibrate.py` and loaded each week.

## 4. Why the LLM only writes the brief, and how it is checked

The statistics decide what is flagged. The LLM only describes rows already flagged. Its text is
checked in code: every number must appear in the flagged rows (prices, references, ratios,
counts), and every `[#n]` must be a real row. A brief that fails is replaced by the plain table,
and the reason is printed. This catches the most damaging error: an invented price.

## 5. MLflow

Each weekly run logs parameters (week, α, scorer, calibration size), metrics (rows scored, flags,
flag rate, districts, share of likely pack/unit mismatches, runtime) and the brief and CSV as
artifacts. MLflow 3 no longer accepts a plain folder store by default, so the default is a local
SQLite file; setting `MLFLOW_TRACKING_URI` sends runs to a server such as DagsHub instead.

## 6. What I would add next

Mondrian (per item group) calibration, because fresh produce varies more than packaged goods;
hand-labelling a sample of flags to measure precision; alerting when the weekly flag rate drifts
far from α.
