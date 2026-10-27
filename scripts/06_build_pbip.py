"""Generate the Power BI project (PBIP: TMDL semantic model + PBIR report) under powerbi/.

The report reads the CSVs written by 05_export_bi.py from the folder in the DataFolder parameter.
Re-run this script after changing the layout; edits made in Power BI Desktop are overwritten.
Lineage tags and logical ids are uuid5 values, so regenerating gives identical files.
Helpers are shared with the hr-attrition-decision report generator.
"""
import json
import shutil
import uuid
from pathlib import Path

NAME = "PriceCatcher"
ROOT = Path(__file__).resolve().parents[1]
PBI = ROOT / "powerbi"
DATA = PBI / "data"
NS = uuid.UUID("3d8b2e5a-19fa-4c8d-b3e1-7a4c9f0e2b6d")
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"
BLUE, INK, MUTED = "#1D4ED8", "#0F172A", "#64748B"
# order matches likely_cause sort order: decimal slip, first observation, pack/unit,
# persistent premium, spike, above peers, so the stacked chart colours carry meaning
SERIES = ["#64748B", "#94A3B8", "#CBD5E1", "#1D4ED8", "#DC2626", "#93C5FD", "#0284C7", "#334155"]


# ---------------------------------------------------------------- semantic model (TMDL)

def tag(*parts):
    return str(uuid.uuid5(NS, "/".join(parts)))


# ---------------------------------------------------------------- semantic model (TMDL)

def csv_table(name, cols, measures=(), calc_cols=(), sort_by=None):
    """cols: [(column, tmdl type, M type, format or None)]; sort_by: {column: column it sorts by}"""
    sort_by = sort_by or {}
    out = [f"table {name}", f"\tlineageTag: {tag(name)}", ""]
    for m in measures:
        out += measure_lines(name, *m)
    for c, typ, _, fmt in cols:
        out += [f"\tcolumn {q(c)}", f"\t\tdataType: {typ}"]
        if fmt:
            out.append(f"\t\tformatString: {fmt}")
        out += [f"\t\tlineageTag: {tag(name, c)}", "\t\tsummarizeBy: none", f"\t\tsourceColumn: {c}"]
        if c in sort_by:
            out.append(f"\t\tsortByColumn: {sort_by[c]}")
        out.append("")
    for c, typ, expr, fmt in calc_cols:
        out += [f"\tcolumn {q(c)} = {expr}", f"\t\tdataType: {typ}"]
        if fmt:
            out.append(f"\t\tformatString: {fmt}")
        out += [f"\t\tlineageTag: {tag(name, c)}", "\t\tsummarizeBy: none", ""]
    types = ",\n".join(f'\t\t\t\t        {{"{c}", {mt}}}' for c, _, mt, _ in cols)
    out += [f"\tpartition {name} = m", "\t\tmode: import", "\t\tsource =", "\t\t\t\tlet",
            f'\t\t\t\t    Source = Csv.Document(File.Contents(DataFolder & "\\{name}.csv"), '
            '[Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),',
            "\t\t\t\t    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),",
            "\t\t\t\t    Typed = Table.TransformColumnTypes(Promoted, {", types, '\t\t\t\t    }, "en-US")',
            "\t\t\t\tin", "\t\t\t\t    Typed", ""]
    return "\n".join(out)


def measure_lines(table, name, expr, fmt=None, folder=None, hidden=False):
    lines = expr.strip().splitlines()
    if len(lines) == 1:
        out = [f"\tmeasure {q(name)} = {lines[0]}"]
    else:
        out = [f"\tmeasure {q(name)} ="] + [f"\t\t\t{ln}" for ln in lines]
    if fmt:
        out.append(f"\t\tformatString: {fmt}")
    if hidden:
        out.append("\t\tisHidden")
    if folder:
        out.append(f"\t\tdisplayFolder: {folder}")
    out += [f"\t\tlineageTag: {tag(table, 'm', name)}", ""]
    return out


def q(name):
    return f"'{name}'" if any(ch in name for ch in " -()/%") else name


def metric(name):
    return f'CALCULATE ( MAX ( metrics[value] ), metrics[metric] = "{name}" )'


SHORTLIST_MEASURES = [
    ("Flagged Prices", "COUNTROWS ( shortlist )", "#,0", "KPI"),
    ("Districts Covered", "DISTINCTCOUNT ( shortlist[district_key] )", "#,0", "KPI"),
    ("Median Ratio", "MEDIAN ( shortlist[ratio_to_ref] )", '0.00"×"', "KPI"),
    ("Persistent Share",
     'DIVIDE (\n    CALCULATE ( COUNTROWS ( shortlist ), KEEPFILTERS ( shortlist[likely_cause] = "persistent premium" ) ),\n'
     '    COUNTROWS ( shortlist )\n)', "0%", "KPI"),
    ("Data Quality Flags",
     'CALCULATE (\n    COUNTROWS ( shortlist ),\n    KEEPFILTERS ( shortlist[likely_cause] IN '
     '{ "first observation", "pack/unit mismatch?", "decimal slip?" } )\n) + 0', "#,0", "KPI"),
    ("Label Flagged", '"top 20 per district · " & FORMAT ( [Districts Covered], "#,0" ) & " districts"', None, "Labels"),
    ("Label Districts", '"with at least 5 comparable shops"', None, "Labels"),
    ("Label Median", '"flagged price ÷ peer median"', None, "Labels"),
    ("Label Persistent", '"shops usually 15%+ above peers"', None, "Labels"),
    ("Label Quality", '"first price, pack size or decimal slip?"', None, "Labels"),
    ("Title Shortlist",
     '"Inspection shortlist · week of " & FORMAT ( MAX ( run_info[latest_week] ), "d mmm yyyy" ) & " · highest score first"',
     None, "Titles"),
]

METRIC_MEASURES = [
    ("Test Prices", metric("test_rows"), "#,0", "KPI"),
    ("Planted Prices", metric("test_planted"), "#,0", "KPI"),
    ("Test District Weeks", metric("test_district_weeks"), "#,0", "KPI"),
    ("Weeks Scored", metric("weeks"), "#,0", "KPI"),
    ("Budget K", metric("budget_k"), "0", "KPI"),
    ("Label Test Prices", '"7 Sep – 24 Oct, unseen in fitting"', None, "Labels"),
    ("Label Planted", '"one-week, held and ×10 rises"', None, "Labels"),
    ("Label District Weeks", '"each gets its own top-20 list"', None, "Labels"),
    ("Label Budget", '"per district per week, about 1.7%"', None, "Labels"),
]


def recall_measure(label):
    return (f'CALCULATE ( AVERAGE ( recall[recall] ), recall[split] = "test", '
            f'recall[scorer_label] = "{label}" )')


RECALL_MEASURES = [
    ("Recall MAD", recall_measure("MAD baseline"), "0%", "Recall"),
    ("Recall IForest", recall_measure("Isolation Forest"), "0%", "Recall"),
    ("Recall ECOD", recall_measure("ECOD"), "0%", "Recall"),
]

RUN_INFO_MEASURES = [
    ("Chosen Scorer", "SELECTEDVALUE ( run_info[chosen_scorer] )", None, "KPI"),
    ("Label Chosen", '"best on validation, rule set first"', None, "Labels"),
]

TABLES = ["shortlist", "flags_weekly", "items_flagged", "recall", "paired", "funnel", "metrics", "run_info"]


def model_files():
    files = {}
    files["definition/database.tmdl"] = "database\n\tcompatibilityLevel: 1567\n"
    files["definition/model.tmdl"] = "\n".join([
        "model Model", "\tculture: en-US", "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tsourceQueryCulture: en-US", "\tdataAccessOptions", "\t\tlegacyRedirects", "\t\treturnErrorValuesAsNull", "",
        "annotation __PBI_TimeIntelligenceEnabled = 0", "",
        f"annotation PBI_QueryOrder = {json.dumps(['DataFolder'] + TABLES)}", "",
        'annotation PBI_ProTooling = ["DevMode"]', "", *[f"ref table {q(t)}" for t in TABLES], ""])
    files["definition/expressions.tmdl"] = (
        f'expression DataFolder = "{DATA}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
        f"\tlineageTag: {tag('DataFolder')}\n\n\tannotation PBI_ResultType = Text\n")

    txt, i64, dbl = ("string", "type text", None), ("int64", "Int64.Type", "#,0"), ("double", "type number", None)
    dt = ("dateTime", "type date", "d mmm yyyy")
    rm, pct_diff = '"RM"#,0.00', "+0.000;-0.000;0.000"

    def cols(*spec):
        return [(n, t[0], t[1], fmt if fmt is not None else t[2]) for n, t, fmt in spec]

    T = {}
    T["shortlist"] = csv_table("shortlist", cols(
        ("state", txt, None), ("district", txt, None), ("district_key", txt, None), ("premise", txt, None),
        ("premise_type", txt, None), ("item_category", txt, None), ("item", txt, None), ("unit", txt, None),
        ("price", dbl, rm), ("ref_price", dbl, rm), ("ratio_to_ref", dbl, '0.00"×"'), ("peer_count", i64, None),
        ("own_premium", dbl, "+0%;-0%;0%"), ("score", dbl, "0.0"), ("rank_in_district", i64, "0"),
        ("likely_cause", txt, None), ("cause_order", i64, "0")),
        SHORTLIST_MEASURES, sort_by={"likely_cause": "cause_order"})
    T["flags_weekly"] = csv_table("flags_weekly", cols(
        ("week", dt, None), ("state", txt, None), ("likely_cause", txt, None), ("flags", i64, None),
        ("mean_ratio", dbl, "0.00"), ("cause_order", i64, "0")),
        [("Weekly Flags", "SUM ( flags_weekly[flags] )", "#,0", "Base")], sort_by={"likely_cause": "cause_order"})
    T["items_flagged"] = csv_table("items_flagged", cols(
        ("item_category", txt, None), ("item", txt, None), ("unit", txt, None), ("flags", i64, None),
        ("prices_checked", i64, None), ("flag_rate", dbl, "0.0%")),
        [("Item Flag Rate", "DIVIDE ( SUM ( items_flagged[flags] ), SUM ( items_flagged[prices_checked] ) )",
          "0.0%", "Base")])
    T["recall"] = csv_table("recall", cols(
        ("split", txt, None), ("scorer_label", txt, None), ("scorer_order", i64, "0"), ("planted", txt, None),
        ("planted_order", i64, "0"), ("recall", dbl, "0.0%"), ("precision_at_k", dbl, "0.0%"),
        ("average_precision", dbl, "0.000")),
        RECALL_MEASURES, sort_by={"planted": "planted_order", "scorer_label": "scorer_order"})
    T["paired"] = csv_table("paired", cols(
        ("comparison", txt, None), ("planted", txt, None), ("planted_order", i64, "0"), ("diff", dbl, pct_diff),
        ("ci_lo", dbl, pct_diff), ("ci_hi", dbl, pct_diff), ("verdict", txt, None)),
        sort_by={"planted": "planted_order"})
    T["funnel"] = csv_table("funnel", cols(("step", txt, None), ("rows", i64, None), ("step_order", i64, "0")),
                            [("Funnel Rows", "SUM ( funnel[rows] )", "#,0", "Base")], sort_by={"step": "step_order"})
    T["metrics"] = csv_table("metrics", cols(("metric", txt, None), ("value", dbl, "0")), METRIC_MEASURES)
    T["run_info"] = csv_table("run_info", cols(("chosen_scorer", txt, None), ("latest_week", dt, None)),
                              RUN_INFO_MEASURES)
    for name, body in T.items():
        files[f"definition/tables/{name}.tmdl"] = body
    files[".platform"] = platform("SemanticModel")
    files["definition.pbism"] = json.dumps({
        "$schema": f"{SCHEMA}/item/semanticModel/definitionProperties/1.0.0/schema.json",
        "version": "4.2", "settings": {}}, indent=2)
    return files


def platform(kind):
    return json.dumps({
        "$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": kind, "displayName": NAME},
        "config": {"version": "2.0", "logicalId": tag("platform", kind)}}, indent=2)


# ---------------------------------------------------------------- report (PBIR)

def lit(v):
    return {"expr": {"Literal": {"Value": v}}}


def s(text):
    return lit("'" + text.replace("'", "''") + "'")


def color(hexv):
    return {"solid": {"color": lit(f"'{hexv}'")}}


def measure_ref(entity, prop):
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def col_ref(entity, prop):
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def proj(entity, prop, display=None, measure=False, active=None):
    p = {"field": measure_ref(entity, prop) if measure else col_ref(entity, prop),
         "queryRef": f"{entity}.{prop}", "nativeQueryRef": prop}
    if display:
        p["displayName"] = display
    if active:
        p["active"] = True
    return p


def M(entity, prop, display=None):
    return proj(entity, prop, display, measure=True)


def C(entity, prop, display=None, active=None):
    return proj(entity, prop, display, active=active)


def padding(t, b, l, r):
    return [{"properties": {"top": lit(f"{t}D"), "bottom": lit(f"{b}D"), "left": lit(f"{l}D"), "right": lit(f"{r}D")}}]


class Page:
    def __init__(self, name, display):
        self.name, self.display, self.visuals, self.z, self.interactions = name, display, {}, 1000, []

    def add(self, name, pos, visual, title=None, subtitle=None, alt=None, background=True, pad=None):
        x, y, w, h = pos
        self.z += 1000
        vco = {"title": [{"properties": {"show": lit("true" if title else "false")}}],
               "background": [{"properties": {"show": lit("true" if background else "false")}}]}
        if title:
            t = vco["title"][0]["properties"]
            t["text"] = title if isinstance(title, dict) else s(title)
        if subtitle:
            vco["subTitle"] = [{"properties": {"show": lit("true"), "text": s(subtitle)}}]
        if alt:
            vco["general"] = [{"properties": {"altText": s(alt)}}]
        if pad:
            vco["padding"] = padding(*pad)
        self.visuals[name] = {
            "$schema": f"{SCHEMA}/item/report/definition/visualContainer/2.3.0/schema.json",
            "name": name, "position": {"x": x, "y": y, "z": self.z, "height": h, "width": w, "tabOrder": self.z},
            "visual": {**visual, "visualContainerObjects": vco, "drillFilterOtherVisuals": True}}

    def text(self, name, pos, paragraphs, background=False):
        runs = [{"textRuns": [{"value": v, "textStyle": {"fontFamily": f, "fontSize": sz, "color": c}}]}
                for v, f, sz, c in paragraphs]
        self.add(name, pos, {"visualType": "textbox", "objects": {"general": [{"properties": {"paragraphs": runs}}]}},
                 background=background, pad=(0, 0, 0, 0) if not background else (12, 12, 16, 16))

    def header(self, pos, title, subtitle):
        self.text("header", pos, [(title, "Segoe UI Semibold", "18pt", INK), (subtitle, "Segoe UI", "9pt", MUTED)])

    def kpi(self, key, pos, entity, value, label, title, alt):
        x, y, w, h = pos
        self.add(f"kpi_{key}_tile", pos, {"visualType": "textbox",
                                          "objects": {"general": [{"properties": {"paragraphs": []}}]}},
                 pad=(0, 0, 0, 0))
        self.add(f"kpi_{key}_value", (x, y, w, 76),
                 {"visualType": "card", "query": {"queryState": {"Values": {"projections": [M(entity, value)]}}},
                  "objects": {"labels": [{"properties": {"fontSize": lit("24D"), "labelDisplayUnits": lit("1D")}}]}},
                 title=title, alt=alt, background=False, pad=(8, 0, 16, 16))
        v = self.visuals[f"kpi_{key}_value"]["visual"]["visualContainerObjects"]["title"][0]["properties"]
        v.update({"fontSize": lit("10D"), "fontColor": color(MUTED), "alignment": s("left")})
        self.add(f"kpi_{key}_label", (x, y + 76, w, h - 76),
                 {"visualType": "card", "query": {"queryState": {"Values": {"projections": [M(entity, label)]}}},
                  "objects": {"labels": [{"properties": {"fontSize": lit("9D"), "color": color(MUTED)}}]}},
                 background=False, pad=(0, 0, 16, 16))

    def slicer(self, name, pos, entity, column, display, default=None, alt=None, single=True):
        objects = {"data": [{"properties": {"mode": s("Dropdown")}}],
                   "header": [{"properties": {"show": lit("true"), "text": s(display)}}]}
        if single:
            objects["selection"] = [{"properties": {"singleSelect": lit("true")}}]
        if default is not None:
            objects["general"] = [{"properties": {"filter": {"filter": {
                "Version": 2, "From": [{"Name": "p", "Entity": entity, "Type": 0}],
                "Where": [{"Condition": {"In": {
                    "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "p"}}, "Property": column}}],
                    "Values": [[{"Literal": {"Value": default}}]]}}}]}}}}]
        self.add(name, pos, {"visualType": "slicer",
                             "query": {"queryState": {"Values": {"projections": [C(entity, column, display, True)]}}},
                             "objects": objects}, alt=alt, pad=(4, 4, 12, 12))

    def chart(self, name, pos, vtype, query, title, subtitle=None, alt=None, objects=None, sort=None, pad=None):
        v = {"visualType": vtype, "query": {"queryState": query}, "objects": objects or {}}
        if sort:
            v["query"]["sortDefinition"] = {"sort": [{"field": sort[0], "direction": sort[1]}]}
        self.add(name, pos, v, title=title, subtitle=subtitle, alt=alt, pad=pad)

    def table(self, name, pos, projections, title, subtitle=None, alt=None, sort=None, widths=None):
        objects = {"total": [{"properties": {"totals": lit("false")}}],
                   "columnHeaders": [{"properties": {"wordWrap": lit("true")}}]}
        if widths:  # one width per projection, in order
            objects["columnWidth"] = [{"properties": {"value": lit(f"{w}D")},
                                       "selector": {"metadata": pr["queryRef"]}}
                                      for pr, w in zip(projections, widths)]
        self.chart(name, pos, "tableEx", {"Values": {"projections": projections}}, title, subtitle, alt,
                   objects=objects, sort=sort, pad=(8, 8, 16, 16))

    def files(self):
        bg = {"color": color("#F4F6F9"), "transparency": lit("0D")}
        out = {f"definition/pages/{self.name}/page.json": {
            "$schema": f"{SCHEMA}/item/report/definition/page/2.0.0/schema.json",
            "name": self.name, "displayName": self.display, "displayOption": "FitToPage", "height": 720, "width": 1280,
            "objects": {"background": [{"properties": bg}], "outspace": [{"properties": bg}]},
            **({"visualInteractions": self.interactions} if self.interactions else {})}}
        for vn, v in self.visuals.items():
            out[f"definition/pages/{self.name}/visuals/{vn}/visual.json"] = v
        return out


KPI_X = [24, 272, 520, 768, 1016]  # 5 x 240 + 4 x 8 gutter + 2 x 24 margin = 1280


def bar(p, name, pos, entity, category, measure, title, subtitle, alt, sort, labels=True):
    objects = {"dataPoint": [{"properties": {"fill": color(BLUE)}}],
               "categoryAxis": [{"properties": {"maxMarginFactor": lit("40L")}}]}
    if labels:
        objects["labels"] = [{"properties": {"show": lit("true"), "labelDisplayUnits": lit("1D")}}]
    p.chart(name, pos, "clusteredBarChart",
            {"Category": {"projections": [C(entity, category, active=True)]},
             "Y": {"projections": [M(entity, measure)]}},
            title, subtitle, alt, objects=objects, sort=sort)


def page_shortlist():
    p = Page("shortlist", "This week")
    p.header((24, 24, 736, 56), "Which shop prices should inspectors check this week?",
             "KPDN PriceCatcher, data.gov.my · each price vs the same item at similar shops nearby that week · "
             "a flag is a lead, not proof")
    p.slicer("slicer_state", (768, 24, 240, 56), "shortlist", "state", "State", single=False,
             alt="Filter the shortlist by state.")
    p.slicer("slicer_cause", (1016, 24, 240, 56), "shortlist", "likely_cause", "Likely cause", single=False,
             alt="Filter the shortlist by likely cause.")
    kp = [("flagged", "Flagged Prices", "Label Flagged", "Flagged prices"),
          ("districts", "Districts Covered", "Label Districts", "Districts covered"),
          ("median", "Median Ratio", "Label Median", "Typical flagged price"),
          ("persistent", "Persistent Share", "Label Persistent", "Persistent premium"),
          ("quality", "Data Quality Flags", "Label Quality", "Possible data issues")]
    for x, (key, val, lab, title) in zip(KPI_X, kp):
        p.kpi(key, (x, 88, 240, 104), "shortlist", val, lab, title, f"{title}.")
    bar(p, "chart_state", (24, 200, 400, 256), "shortlist", "state", "Flagged Prices",
        "Flagged prices by state", "20 per district, so states with more districts get more",
        "Bar chart of flagged prices by state.", (measure_ref("shortlist", "Flagged Prices"), "Descending"))
    bar(p, "chart_cause", (432, 200, 400, 256), "shortlist", "likely_cause", "Flagged Prices",
        "Why each price may stand out", "Rule-based hints, checked in this order",
        "Bar chart of flagged prices by likely cause.", (col_ref("shortlist", "likely_cause"), "Ascending"))
    bar(p, "chart_category", (840, 200, 416, 256), "shortlist", "item_category", "Flagged Prices",
        "Item categories flagged", "Categories as in KPDN's item list",
        "Bar chart of flagged prices by item category.", (measure_ref("shortlist", "Flagged Prices"), "Descending"),
        labels=False)
    p.table("table_shortlist", (24, 464, 1232, 232),
            [C("shortlist", "premise", "Premise"), C("shortlist", "district_key", "District"),
             C("shortlist", "premise_type", "Type"), C("shortlist", "item", "Item"), C("shortlist", "unit", "Unit"),
             C("shortlist", "price", "Price"), C("shortlist", "ref_price", "Peer median"),
             C("shortlist", "ratio_to_ref", "× peers"), C("shortlist", "likely_cause", "Likely cause"),
             C("shortlist", "score", "Score")],
            {"expr": measure_ref("shortlist", "Title Shortlist")},
            "Score = how far above peers, in units of the peers' normal spread · peer median = same item, same week, "
            "same shop type nearby",
            "Table of flagged prices with premise, item, price, peer median and likely cause, highest score first.",
            sort=(col_ref("shortlist", "score"), "Descending"),
            widths=[210, 180, 100, 240, 55, 70, 80, 60, 125, 60])
    return p


def page_over_time():
    p = Page("over_time", "Over time")
    p.header((24, 24, 984, 56), "Is the pattern stable week to week?",
             "Same rule applied to all 40 weeks, Jan–Oct 2026 · the first week is mostly 'first observation' "
             "because shops have no price history yet")
    p.slicer("slicer_state", (1016, 24, 240, 56), "flags_weekly", "state", "State", single=False,
             alt="Filter the weekly chart by state.")
    p.chart("chart_weekly", (24, 88, 1232, 296), "columnChart",
            {"Category": {"projections": [C("flags_weekly", "week", active=True)]},
             "Series": {"projections": [C("flags_weekly", "likely_cause")]},
             "Y": {"projections": [M("flags_weekly", "Weekly Flags")]}},
            "Flagged prices per week, by likely cause",
            "About 3,300 a week: the budget is fixed, so the mix matters more than the total",
            "Stacked column chart of flagged prices per week, split by likely cause.",
            objects={"legend": [{"properties": {"show": lit("true"), "position": s("Top")}}]},
            sort=(col_ref("flags_weekly", "week"), "Ascending"))
    p.table("table_items", (24, 392, 800, 304),
            [C("items_flagged", "item_category", "Category"), C("items_flagged", "item", "Item"),
             C("items_flagged", "unit", "Unit"), C("items_flagged", "flags", "Flags"),
             C("items_flagged", "prices_checked", "Prices checked"), C("items_flagged", "flag_rate", "Flag rate")],
            "Items flagged most often (at least 1,000 prices checked)",
            "Flag rate = share of this item's prices that made a district's top 20",
            "Table of items with the highest flag rates.", sort=(col_ref("items_flagged", "flag_rate"), "Descending"),
            widths=[140, 300, 90, 70, 90, 70])
    p.text("notes_over_time", (832, 392, 424, 304), [
        ("Reading the trend", "Segoe UI Semibold", "12pt", INK),
        ("The budget is fixed at 20 prices per district each week, so the weekly total barely moves. "
         "What changes is why prices stand out.", "Segoe UI", "10pt", INK),
        ("About half the flags each week are shops that are always pricier than their peers (persistent premium). "
         "They are worth a look once, then they mostly use up budget.", "Segoe UI", "10pt", INK),
        ("Infant formula and some packaged goods top the flag rate: same item code, very different prices across "
         "shop types.", "Segoe UI", "10pt", INK),
        ("Hints are rules of thumb, not verified causes.", "Segoe UI", "9pt", MUTED),
    ], background=True)
    return p


def page_model_check():
    p = Page("model_check", "Does it work?")
    p.header((24, 24, 1232, 56), "Does the detector catch overpricing?",
             "No real labels exist, so known price rises were planted in the test weeks and we counted how many "
             "landed in their district's top 20 · scorer fixed on Jul–Aug validation before testing")
    kp = [("test_prices", "metrics", "Test Prices", "Label Test Prices", "Test prices"),
          ("planted", "metrics", "Planted Prices", "Label Planted", "Planted rises"),
          ("dist_weeks", "metrics", "Test District Weeks", "Label District Weeks", "District-weeks"),
          ("scorer", "run_info", "Chosen Scorer", "Label Chosen", "Scorer used"),
          ("budget", "metrics", "Budget K", "Label Budget", "Inspection budget")]
    for x, (key, ent, val, lab, title) in zip(KPI_X, kp):
        p.kpi(key, (x, 88, 240, 104), ent, val, lab, title, f"{title}.")
    p.chart("chart_recall", (24, 200, 736, 256), "clusteredBarChart",
            {"Category": {"projections": [C("recall", "planted", active=True)]},
             "Y": {"projections": [M("recall", "Recall MAD", "MAD baseline"),
                                   M("recall", "Recall IForest", "Isolation Forest"),
                                   M("recall", "Recall ECOD", "ECOD")]}},
            "Share of planted rises caught, test weeks",
            "Random picking ≈ 1.7% · baseline wins on modest and held rises, Isolation Forest on big one-week jumps",
            "Clustered bar chart of the share of planted price rises caught by each scorer.",
            objects={"dataPoint": [{"properties": {"fill": color(BLUE)}, "selector": {"metadata": "recall.Recall MAD"}},
                                   {"properties": {"fill": color("#94A3B8")},
                                    "selector": {"metadata": "recall.Recall IForest"}},
                                   {"properties": {"fill": color("#CBD5E1")},
                                    "selector": {"metadata": "recall.Recall ECOD"}}],
                     "labels": [{"properties": {"show": lit("true")}}],
                     "legend": [{"properties": {"show": lit("true"), "position": s("Top")}}]},
            sort=(col_ref("recall", "planted"), "Ascending"))
    p.table("table_paired", (768, 200, 488, 256),
            [C("paired", "comparison", "MAD baseline"), C("paired", "planted", "Planted rise"),
             C("paired", "diff", "Difference"), C("paired", "ci_lo", "95% CI low"),
             C("paired", "ci_hi", "95% CI high"), C("paired", "verdict", "Better")],
            "Is the difference real?", "Paired bootstrap over 993 district-weeks · positive = baseline catches more",
            "Table of paired differences in recall with 95% intervals.", sort=(col_ref("paired", "planted"), "Ascending"),
            widths=[120, 86, 72, 56, 56, 62])
    bar(p, "chart_funnel", (24, 464, 600, 232), "funnel", "step", "Funnel Rows",
        "From raw records to scored prices", "1 Jan – 24 Oct 2026",
        "Bar chart of row counts at each data-preparation step.", (col_ref("funnel", "step"), "Ascending"))
    p.text("notes_limits", (632, 464, 624, 232), [
        ("Limits", "Segoe UI Semibold", "12pt", INK),
        ("Planted rises are random; a coordinated district-wide rise would not stand out from peers.",
         "Segoe UI", "10pt", INK),
        ("Many flags are premium shops or pack-size mismatches; the data has no pack-size field.",
         "Segoe UI", "10pt", INK),
        ("Festive price caps (SHMMP) would give real labels but are gazetted per district and were not used.",
         "Segoe UI", "9pt", MUTED),
    ], background=True)
    return p


def report_files():
    pages = [page_shortlist(), page_over_time(), page_model_check()]
    files = {
        ".platform": platform("Report"),
        "definition.pbir": {"$schema": f"{SCHEMA}/item/report/definitionProperties/2.0.0/schema.json",
                            "version": "4.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}},
        "definition/version.json": {"$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json",
                                    "version": "2.0.0"},
        "definition/report.json": {
            "$schema": f"{SCHEMA}/item/report/definition/report/3.0.0/schema.json",
            "themeCollection": {"customTheme": {"name": f"{NAME}.json", "reportVersionAtImport": {
                "visual": "2.3.0", "report": "3.0.0", "page": "2.0.0"}, "type": "RegisteredResources"}},
            "objects": {"outspacePane": [{"properties": {"expanded": lit("false")}}]},
            "resourcePackages": [{"name": "RegisteredResources", "type": "RegisteredResources", "items": [
                {"name": f"{NAME}.json", "path": f"{NAME}.json", "type": "CustomTheme"}]}],
            "settings": {"useEnhancedTooltips": True}},
        "definition/pages/pages.json": {"$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
                                        "pageOrder": [p.name for p in pages], "activePageName": pages[0].name},
        f"StaticResources/RegisteredResources/{NAME}.json": theme(),
    }
    for p in pages:
        files.update(p.files())
    return files


def theme():
    axis = {"labelColor": {"solid": {"color": MUTED}}, "fontSize": 9, "fontFamily": "Segoe UI", "showAxisTitle": False}
    return {
        "name": NAME, "dataColors": SERIES, "background": "#FFFFFF", "foreground": INK, "tableAccent": BLUE,
        "good": "#16A34A", "neutral": MUTED, "bad": "#DC2626",
        "textClasses": {"callout": {"fontSize": 26, "fontFace": "Segoe UI Semibold", "color": INK},
                        "title": {"fontSize": 12, "fontFace": "Segoe UI Semibold", "color": INK},
                        "header": {"fontSize": 10, "fontFace": "Segoe UI Semibold", "color": INK},
                        "label": {"fontSize": 9, "fontFace": "Segoe UI", "color": MUTED}},
        "visualStyles": {
            "page": {"*": {"background": [{"color": {"solid": {"color": "#F4F6F9"}}, "transparency": 0}],
                           "outspace": [{"color": {"solid": {"color": "#F4F6F9"}}, "transparency": 0}]}},
            "*": {"*": {"background": [{"show": True, "color": {"solid": {"color": "#FFFFFF"}}, "transparency": 0}],
                        "border": [{"show": False, "radius": 6}], "dropShadow": [{"show": False}],
                        "padding": [{"top": 12, "bottom": 12, "left": 16, "right": 16}],
                        "title": [{"show": True, "fontColor": {"solid": {"color": INK}}, "fontSize": 12,
                                   "fontFamily": "Segoe UI Semibold", "alignment": "left"}],
                        "subTitle": [{"fontColor": {"solid": {"color": MUTED}}, "fontSize": 9}]}},
            "card": {"*": {"labels": [{"color": {"solid": {"color": INK}}, "fontSize": 26,
                                       "fontFamily": "Segoe UI Semibold"}], "categoryLabels": [{"show": False}]}},
            "lineChart": {"*": {"lineStyles": [{"strokeWidth": 2}], "categoryAxis": [{**axis, "gridlineShow": False}],
                                "valueAxis": [{**axis, "gridlineShow": True,
                                               "gridlineColor": {"solid": {"color": "#E2E8F0"}}}],
                                "legend": [{"show": True, "position": "Top", "fontSize": 9, "showTitle": False,
                                            "labelColor": {"solid": {"color": MUTED}}}]}},
            "clusteredBarChart": {"*": {"categoryAxis": [axis], "valueAxis": [{**axis, "show": False}]}},
            "columnChart": {"*": {"categoryAxis": [{**axis, "gridlineShow": False}], "valueAxis": [{**axis, "gridlineShow": True,
                            "gridlineColor": {"solid": {"color": "#E2E8F0"}}}],
                            "legend": [{"show": True, "position": "Top", "fontSize": 9, "showTitle": False}]}},
            "clusteredColumnChart": {"*": {"categoryAxis": [axis], "valueAxis": [{**axis, "gridlineShow": True,
                                     "gridlineColor": {"solid": {"color": "#E2E8F0"}}}],
                                     "legend": [{"show": True, "position": "Top", "fontSize": 9, "showTitle": False}]}},
            "tableEx": {"*": {"grid": [{"gridHorizontal": True, "gridHorizontalColor": {"solid": {"color": "#E2E8F0"}},
                                        "gridVertical": False, "rowPadding": 2}],
                              "columnHeaders": [{"fontColor": {"solid": {"color": MUTED}}, "fontSize": 9,
                                                 "backColor": {"solid": {"color": "#FFFFFF"}}}],
                              "values": [{"fontSize": 9, "fontColorPrimary": {"solid": {"color": INK}},
                                          "backColorPrimary": {"solid": {"color": "#FFFFFF"}},
                                          "backColorSecondary": {"solid": {"color": "#FFFFFF"}}}]}},
            "slicer": {"*": {"header": [{"fontColor": {"solid": {"color": MUTED}}, "fontSize": 9}],
                             "items": [{"fontColor": {"solid": {"color": INK}}, "fontSize": 10}]}},
        }}


def write(base: Path, files: dict):
    for rel, content in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)
        path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def main():
    for part in ["Report", "SemanticModel"]:
        d = PBI / f"{NAME}.{part}"
        if d.exists():
            for sub in d.iterdir():  # keep Desktop's .pbi cache folder, replace the definition
                if sub.name != ".pbi":
                    shutil.rmtree(sub) if sub.is_dir() else sub.unlink()
    write(PBI / f"{NAME}.SemanticModel", model_files())
    write(PBI / f"{NAME}.Report", report_files())
    write(PBI, {f"{NAME}.pbip": {
        "$schema": f"{SCHEMA}/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
        "artifacts": [{"report": {"path": f"{NAME}.Report"}}], "settings": {"enableAutoRecovery": True}},
        ".gitignore": "**/.pbi/localSettings.json\n**/.pbi/cache.abf\n"})
    n = sum(1 for _ in (PBI / f"{NAME}.Report").rglob("visual.json"))
    print(f"wrote {PBI / (NAME + '.pbip')} ({n} visuals)")



if __name__ == "__main__":
    main()
