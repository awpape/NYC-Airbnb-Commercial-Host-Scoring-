"""
NYC Airbnb Commercial Activity Score — Enforcement Priority Dashboard

An interactive Streamlit dashboard that helps city enforcement agencies
identify and prioritize suspected commercial hotel operators in NYC's
short-term rental market.

The dashboard scores listings using five behavioral signals and produces
a ranked priority queue — no access to internal Airbnb data required.

Run with:
    streamlit run app.py

Author: Alain William Pape
"""

import pandas as pd
import streamlit as st
import plotly.express as px

from scoring import score_listings, get_tier_summary
from generate_synthetic_listings import generate_synthetic_listings

st.set_page_config(
    page_title="NYC Airbnb Commercial Activity Score",
    page_icon="🏙️",
    layout="wide"
)

# ── Sidebar ──
st.sidebar.title("Enforcement Console")
st.sidebar.markdown(
    "Score Airbnb listings by commercial activity likelihood using "
    "five behavioral signals from public listing data alone."
)

mode = st.sidebar.radio(
    "Data source",
    ["Use synthetic demo listings", "Upload your own CSV"],
    help="Upload a CSV with standard Inside Airbnb columns to score real listings."
)

n_listings = st.sidebar.slider("Demo cohort size", 20, 200, 50, 10)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
cutoff = st.sidebar.slider(
    "High-tier cutoff (score threshold)",
    min_value=40.0, max_value=80.0, value=60.0, step=5.0,
    help="Listings scoring at or above this threshold are flagged as High tier."
)
regenerate = st.sidebar.button("Regenerate demo listings")

# ── Load and score data ──
@st.cache_data
def get_demo_data(n, seed, cutoff):
    listings = generate_synthetic_listings(n=n, seed=seed)
    return score_listings(listings, cutoff=cutoff)

if regenerate:
    get_demo_data.clear()

if mode == "Use synthetic demo listings":
    scored = get_demo_data(n_listings, seed, cutoff)
    st.sidebar.caption("Synthetic demo data — no real listings are displayed.")
else:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV (Inside Airbnb format)",
        type=["csv"],
        help="Download listings.csv from insideairbnb.com/get-the-data"
    )
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to begin scoring real listings.")
        st.stop()
    raw = pd.read_csv(uploaded)
    scored = score_listings(raw, cutoff=cutoff)

# ── Header ──
st.title("🏙️ NYC Airbnb Commercial Activity Score")
st.caption(
    "Behavioral scoring model using five public signals: portfolio size, "
    "availability, review cadence, minimum nights, and room type. "
    "Score range: 0–100. High tier (≥ 60) = suspected commercial operator."
)

# ── KPI Metrics ──
summary = get_tier_summary(scored)
col1, col2, col3, col4, col5 = st.columns(5)
total = len(scored)

for col, tier, color in zip(
    [col1, col2, col3, col4],
    ["High", "Moderate", "Low", "Dormant"],
    ["🔴", "🟡", "🟢", "⚫"]
):
    row = summary[summary["Tier"] == tier]
    n = int(row["Count"].values[0]) if len(row) else 0
    pct = float(row["Pct"].values[0]) if len(row) else 0.0
    col.metric(f"{color} {tier}", f"{n:,}", f"{pct:.1f}% of listings")

col5.metric("Total listings scored", f"{total:,}")

# ── Score Distribution Chart ──
st.subheader("Score Distribution")
fig = px.histogram(
    scored, x="score", nbins=20,
    color_discrete_sequence=["#1F3864"],
    labels={"score": "Commercial Activity Score", "count": "Listings"},
)
fig.add_vline(x=cutoff, line_dash="dash", line_color="red",
              annotation_text=f"High-tier cutoff ({cutoff:.0f})")
fig.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(gridcolor="#EEEEEE"),
    yaxis=dict(gridcolor="#EEEEEE"),
    showlegend=False, height=300, margin=dict(t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ── Score by Borough ──
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Average Score by Borough")
    borough_avg = (
        scored.groupby("neighbourhood_group")["score"]
        .mean().sort_values(ascending=False).reset_index()
    )
    fig2 = px.bar(
        borough_avg, x="neighbourhood_group", y="score",
        color_discrete_sequence=["#1F3864"],
        labels={"neighbourhood_group": "Borough", "score": "Avg Score"},
    )
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#EEEEEE"),
        yaxis=dict(gridcolor="#EEEEEE"),
        showlegend=False, height=280, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Tier Breakdown by Borough")
    tier_borough = (
        scored.groupby(["neighbourhood_group", "tier"])
        .size().reset_index(name="count")
    )
    fig3 = px.bar(
        tier_borough, x="neighbourhood_group", y="count", color="tier",
        color_discrete_map={
            "High": "#C0392B", "Moderate": "#F39C12",
            "Low": "#27AE60", "Dormant": "#95A5A6"
        },
        labels={"neighbourhood_group": "Borough", "count": "Listings", "tier": "Tier"},
        barmode="stack"
    )
    fig3.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=280, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Priority Queue Table ──
st.subheader("Enforcement Priority Queue (ranked highest score to lowest)")

tier_filter = st.multiselect(
    "Filter by tier",
    ["High", "Moderate", "Low", "Dormant"],
    default=["High", "Moderate"]
)

display_cols = [
    "commercial_rank", "host_name", "name", "neighbourhood_group",
    "score", "tier", "calculated_host_listings_count",
    "availability_365", "minimum_nights", "reviews_per_month", "room_type"
]

filtered = scored[scored["tier"].isin(tier_filter)][display_cols]

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    column_config={
        "score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.1f"
        ),
        "commercial_rank": st.column_config.NumberColumn("Rank"),
        "tier": st.column_config.TextColumn("Tier"),
        "calculated_host_listings_count": st.column_config.NumberColumn("Portfolio Size"),
        "availability_365": st.column_config.NumberColumn("Availability (days/yr)"),
    }
)

# ── Listing Detail ──
st.subheader("Listing Detail and Signal Breakdown")

options = scored["name"] + " — Score: " + scored["score"].astype(str)
selected = st.selectbox("Select a listing", options)
selected_idx = options[options == selected].index[0]
listing = scored.loc[selected_idx]

left, right = st.columns([1, 1])

with left:
    st.markdown(f"### {listing['name']}")
    st.markdown(
        f"**Host:** {listing['host_name']}  \n"
        f"**Borough:** {listing['neighbourhood_group']}  \n"
        f"**Room type:** {listing['room_type']}  \n"
        f"**Price:** ${listing['price']}/night  \n"
        f"**Minimum nights:** {listing['minimum_nights']}  \n"
        f"**Availability:** {listing['availability_365']} days/year  \n"
        f"**Host portfolio:** {listing['calculated_host_listings_count']} listings  \n"
        f"**Reviews/month:** {listing['reviews_per_month']}"
    )

with right:
    st.markdown(f"### Commercial Score: {listing['score']:.1f} — **{listing['tier']}**")
    signal_data = pd.DataFrame({
        "Signal": ["Portfolio Size", "Availability", "Review Cadence",
                   "Minimum Nights", "Room Type"],
        "Weight": [30, 30, 20, 10, 10],
        "Raw Signal": [
            listing["signal_portfolio"],
            listing["signal_availability"],
            listing["signal_reviews"],
            listing["signal_min_nights"],
            listing["signal_room"],
        ]
    })
    signal_data["Contribution"] = (
        signal_data["Weight"] * signal_data["Raw Signal"]
    ).round(1)

    fig4 = px.bar(
        signal_data, x="Signal", y="Contribution",
        color_discrete_sequence=["#1F3864"],
        labels={"Contribution": "Score Contribution (out of weight)"},
        range_y=[0, 35]
    )
    fig4.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=280, margin=dict(t=10, b=10), showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.caption(
        "Each bar shows how much this listing contributed to its total score "
        "from each signal. A full bar means maximum credit on that signal."
    )

# ── Download ──
st.divider()
csv = scored.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download scored listings as CSV",
    data=csv,
    file_name="scored_listings.csv",
    mime="text/csv"
)
