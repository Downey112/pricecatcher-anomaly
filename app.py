"""Streamlit view of the latest inspection shortlist."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SHORTLIST = ROOT / "results" / "shortlist_latest.csv"
FEATURES = ROOT / "data" / "processed" / "features"

st.set_page_config(page_title="PriceCatcher anomaly triage", layout="wide")
st.title("PriceCatcher anomaly triage")
st.caption("Prices that stand out from same-week peers. A flag means 'look at this', "
           "not proof of profiteering. Data: KPDN PriceCatcher via data.gov.my (CC BY 4.0).")

if not SHORTLIST.exists():
    st.error("Run scripts/01_build.py to 03_shortlist.py first.")
    st.stop()

sl = pd.read_csv(SHORTLIST)
c1, c2, c3 = st.columns(3)
state = c1.selectbox("State", sorted(sl["state"].unique()))
districts = sorted(sl.loc[sl["state"] == state, "district"].unique())
district = c2.selectbox("District", districts)
causes = c3.multiselect("Likely cause", sorted(sl["likely_cause"].unique()),
                        default=sorted(sl["likely_cause"].unique()))

view = sl[(sl["state"] == state) & (sl["district"] == district) & sl["likely_cause"].isin(causes)]
st.subheader(f"Week of {sl['week'].iloc[0]}: {len(view)} flagged prices in {district}")
st.dataframe(view.drop(columns=["week", "state", "district"]), hide_index=True, width="stretch")

if FEATURES.exists() and len(view):
    item = st.selectbox("Compare an item with its peers", view["item"].unique())
    week = pd.Timestamp(sl["week"].iloc[0])
    slug = state.replace(" ", "_").replace(".", "")
    feat = pd.read_parquet(FEATURES / f"{slug}.parquet",
                           columns=["week", "district", "premise_code", "item_code",
                                    "price", "ref_price"],
                           filters=[("week", "==", week)])
    feat["district"] = feat["district"].astype(str)
    code = view.loc[view["item"] == item, "item_code"].iloc[0]
    peers = feat[(feat["item_code"] == code) & (feat["district"] == district)]
    flagged = set(view.loc[view["item"] == item, "premise_code"])
    peers = peers.assign(flagged=peers["premise_code"].isin(flagged).map(
        {True: "flagged", False: "peer"}))
    st.scatter_chart(peers, x="premise_code", y="price", color="flagged")
    st.caption(f"Reference (peer median) price: RM{peers['ref_price'].median():.2f}")
