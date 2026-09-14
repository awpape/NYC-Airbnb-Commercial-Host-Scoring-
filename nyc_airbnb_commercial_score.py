"""
NYC Airbnb Commercial Activity Score — Full Analysis Pipeline

Detects de facto commercial hotel operators in NYC's short-term rental
market using behavioral signals from public listing data alone, without
requiring access to internal platform data or confirmed licensing records.

This script reproduces every headline number in the analysis, start to
finish, from raw Inside Airbnb files. It runs as a single pipeline with
printed section headers so every result can be traced back to its source.

Sections
  1.  The Commercial Activity Score: five signals, tier distribution
  2.  Validation check 1 — known operators (face validity)
  3.  Validation check 2 — the price test
  4.  Validation check 3 — a blind algorithm (latent class analysis)
  5.  Validation check 4 — a synthetic city with known ground truth
  6.  Confusion matrix vs simple rules a city might use instead
  7.  Weight and threshold sensitivity
  8.  Scorecard re-derivation vs the hand-set weights
  9.  Six years later — the 2019 to 2025 panel
  10. Does listing-name language add anything?
  11. Missing-value sensitivity
  12. Disparate impact by borough
  13. Model comparison with bootstrap confidence intervals
  14. Consolidated summary of every headline number

Requirements: pandas, numpy, scikit-learn. No internet access needed
once the two CSVs below are in place.

Data: Point D19 / D25 at your own copies of the Inside Airbnb NYC snapshots.
  D19 = December 2019 snapshot (pre Local Law 18)
  D25 = November 2025 snapshot (post Local Law 18, has a 'license' column)
Both need the standard Inside Airbnb columns: id, host_id, host_name, name,
neighbourhood_group, room_type, price, minimum_nights, number_of_reviews,
reviews_per_month, calculated_host_listings_count, availability_365.

Dataset: Inside Airbnb NYC (insideairbnb.com) — public data
Author: Alain William Pape
"""
"""
import re, json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)

# ----------------------------------------------------------------- CONFIG ---
DATA_DIR = "/home/claude/data"
D19 = f"{DATA_DIR}/nyc_listings_2019-12.csv"      # 2019 snapshot, required
D25 = f"{DATA_DIR}/nyc_listings_2025-11.csv"      # 2025 snapshot, optional
OUT_JSON = f"{DATA_DIR}/full_analysis_results.json"

W0 = dict(portfolio=30, availability=30, reviews=20, min_nights=10, room=10)
CUT = 60.0
RNG_SEED = 42
RESULTS = {}

CORP = re.compile(
    r"\b(sonder|blueground|corporate|inc\.?|llc|ltd|group|properties|property|"
    r"realty|management|mgmt|rentals?|suites?|hospitality|stayz?|apartments?|"
    r"lodging|hotel|hostel|residences?|vacation)\b", re.I)


def hdr(n, title):
    print("\n" + "=" * 78)
    print(f"SECTION {n} — {title}")
    print("=" * 78)


def sp(c): return 1.0 if c >= 11 else .75 if c >= 4 else .45 if c >= 2 else 0.
def sa(a): return 1.0 if a >= 271 else .7 if a > 180 else .35 if a > 90 else 0.
def sm(m): return 1.0 if m <= 3 else .4 if m < 30 else 0.


def score_signals(df, q50, q75):
    """The five behavioral signals, 0/partial/full credit each. Verbatim
    across every script in this project — this is the one place it's defined."""
    def sr(r): return 1.0 if r >= q75 else .5 if r >= q50 else 0.
    return pd.DataFrame({
        "portfolio": df["calculated_host_listings_count"].map(sp),
        "availability": df["availability_365"].map(sa),
        "reviews": df["reviews_per_month"].fillna(0).map(sr),
        "min_nights": df["minimum_nights"].map(sm),
        "room": (df["room_type"] == "Entire home/apt").astype(float)})


def apply_score(df, q50, q75, weights=W0, cut=CUT):
    S = score_signals(df, q50, q75)
    score = sum(weights[k] * S[k] for k in weights)
    dormant = (df["availability_365"] == 0) & (df["number_of_reviews"] == 0)
    tier = pd.Series("Low", index=df.index)
    tier[score >= 40] = "Moderate"
    tier[score >= cut] = "High"
    tier[dormant] = "Dormant"
    return score, tier, dormant, S


# =============================================================================
hdr(1, "The Commercial Activity Score")
# =============================================================================
d = pd.read_csv(D19, low_memory=False)
d["reviews_per_month"] = d["reviews_per_month"].fillna(0)
pos = d.loc[d["reviews_per_month"] > 0, "reviews_per_month"]
Q50, Q75 = pos.quantile(.50), pos.quantile(.75)
print(f"Frozen review-cadence cut points (2019 median / 75th pct of positive rpm): "
      f"{Q50:.3f} / {Q75:.3f}")

d["score"], d["tier"], d["dormant"], SIG = apply_score(d, Q50, Q75)
vc = d["tier"].value_counts()
order = ["High", "Moderate", "Low", "Dormant"]
print(f"\n{'Tier':<10}{'n':>8}{'pct':>8}")
for t in order:
    print(f"{t:<10}{vc.get(t, 0):>8}{100*vc.get(t, 0)/len(d):>7.1f}%")

high = d[d["tier"] == "High"]
n_hosts_high = high["host_id"].nunique()
n_single_high = int((high["calculated_host_listings_count"] == 1).sum())
print(f"\nListings:            {len(d):,}")
print(f"High-tier listings:  {len(high):,}")
print(f"Distinct hosts, High:{n_hosts_high:,}")
print(f"Single-listing High: {n_single_high:,}  "
      f"({100*n_single_high/len(high):.1f}% of the High tier)")
print(f"Median score:        {d['score'].median():.1f}")
print(f"Exactly at cutoff 60:{int((d['score'] == 60).sum()):,}")

RESULTS["section1_tiers"] = {
    "n_total": len(d),
    "tiers": {t: {"n": int(vc.get(t, 0)), "pct": round(100*vc.get(t, 0)/len(d), 1)} for t in order},
    "high_hosts_n": int(n_hosts_high),
    "high_single_host_n": n_single_high,
    "high_single_host_pct": round(100*n_single_high/len(high), 1),
    "median_score": float(d["score"].median()),
    "exactly_at_60": int((d["score"] == 60).sum()),
    "rpm_q50": round(float(Q50), 3), "rpm_q75": round(float(Q75), 3)}


# =============================================================================
hdr(2, "Validation check 1 — known operators (face validity)")
# =============================================================================
KNOWN = {"Sonder": "sonder", "Blueground": "blueground"}
RESULTS["section2_face_validity"] = {}
for label, needle in KNOWN.items():
    sub = d[d["host_name"].astype(str).str.contains(needle, case=False, na=False)]
    if len(sub) == 0:
        print(f"{label}: not found in this snapshot (host-name spelling may differ by year)")
        continue
    pct_high = 100 * (sub["tier"] == "High").mean()
    print(f"{label:<12} n={len(sub):>5}   {pct_high:5.1f}% score High tier")
    RESULTS["section2_face_validity"][label] = {"n": len(sub), "pct_high": round(pct_high, 1)}

hosts_11plus = d.groupby("host_id")["id"].count()
hosts_11plus = hosts_11plus[hosts_11plus >= 11]
print(f"\nHosts running 11+ listings: {len(hosts_11plus):,} "
      f"({100*hosts_11plus.sum()/len(d):.1f}% of inventory)")
RESULTS["section2_face_validity"]["hosts_11plus_n"] = int(len(hosts_11plus))


# =============================================================================
hdr(3, "Validation check 2 — the price test")
# =============================================================================
# Does the score predict price after controlling for room type and borough,
# even though price is never one of the five scoring signals?
dp = d[(d["price"] > 0) & (d["price"] < 2000)].copy()
dp["log_price"] = np.log1p(dp["price"])
tier_dummies = pd.get_dummies(dp["tier"], prefix="tier", drop_first=True)
room_dummies = pd.get_dummies(dp["room_type"], prefix="room", drop_first=True)
boro_dummies = pd.get_dummies(dp["neighbourhood_group"], prefix="boro", drop_first=True)
X = pd.concat([tier_dummies, room_dummies, boro_dummies], axis=1).astype(float)
X = np.column_stack([np.ones(len(X)), X.values])
y = dp["log_price"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
r2 = 1 - resid.var() / y.var()
cols = ["intercept"] + list(tier_dummies.columns) + list(room_dummies.columns) + list(boro_dummies.columns)
premiums = {c: round(100*(np.exp(b)-1), 1) for c, b in zip(cols, beta) if c.startswith("tier_")}
print(f"R-squared: {r2:.3f}")
print("Price premium vs the omitted tier (Dormant), same room type & borough:")
for c, v in premiums.items():
    print(f"  {c:<16} {v:+.1f}%")
RESULTS["section3_price_test"] = {"r2": round(float(r2), 3), "premiums_pct": premiums}


# =============================================================================
hdr(4, "Validation check 3 — a blind algorithm (latent class analysis)")
# =============================================================================
# EM on categorical bins of the five raw variables. Sees no score, no weights,
# no cut points, no tiers. If it independently finds the same two-group split,
# that's structure in the market, not an artefact of our weighting.
d["y_corp"] = d["host_name"].astype(str).str.contains(CORP, na=False, regex=True).astype(int)

BINS = {
    "portfolio":    ("calculated_host_listings_count", [-np.inf, 1, 3, 10, np.inf]),
    "availability": ("availability_365",               [-np.inf, 90, 180, 270, np.inf]),
    "reviews":      ("reviews_per_month",               [-np.inf, 0.001, Q50, Q75, np.inf]),
    "min_nights":   ("minimum_nights",                  [-np.inf, 3, 29, 30, np.inf]),
    "room":         (None, None)}

def binned(df):
    B = {}
    for k, (col, edges) in BINS.items():
        B[k] = ((df["room_type"] == "Entire home/apt").map({True: "entire", False: "other"})
                if col is None else pd.cut(df[col], edges)).astype(str)
    return pd.DataFrame(B, index=df.index)

B19 = binned(d)

def lca(Bf, K, seed=7, iters=300, tol=1e-7):
    rng = np.random.default_rng(seed)
    n = len(Bf)
    cats = {c: sorted(Bf[c].unique()) for c in Bf.columns}
    code = {c: Bf[c].map({v: i for i, v in enumerate(cats[c])}).values for c in Bf.columns}
    pi = rng.dirichlet(np.ones(K))
    th = {c: rng.dirichlet(np.ones(len(cats[c])), size=K) for c in Bf.columns}
    ll_old = -np.inf
    for _ in range(iters):
        lp = np.log(pi)[None, :].repeat(n, 0)
        for c in Bf.columns:
            lp += np.log(th[c][:, code[c]].T + 1e-12)
        m = lp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lp - m).sum(1))
        r = np.exp(lp - lse[:, None])
        pi = r.mean(0)
        for c in Bf.columns:
            for k in range(K):
                cnt = np.bincount(code[c], weights=r[:, k], minlength=len(cats[c]))
                th[c][k] = (cnt + 1e-6) / (cnt.sum() + 1e-6 * len(cats[c]))
        ll = lse.sum()
        if abs(ll - ll_old) < tol * abs(ll):
            break
        ll_old = ll
    npar = (K - 1) + K * sum(len(v) - 1 for v in cats.values())
    return dict(pi=pi, theta=th, cats=cats, resp=r, ll=ll, bic=-2*ll + npar*np.log(n))

print("Fitting K = 2, 3, 4 class models (EM)...")
fits = {K: lca(B19, K, seed=7) for K in (2, 3, 4)}
print(f"{'K':>3}{'log-lik':>14}{'BIC':>16}")
for K, f in fits.items():
    print(f"{K:>3}{f['ll']:>14,.1f}{f['bic']:>16,.1f}")
RESULTS["section4_lca_model_selection"] = {
    str(K): {"loglik": round(f["ll"], 1), "BIC": round(f["bic"], 1)} for K, f in fits.items()}

f2 = fits[2]
def prob_of(f, var, level_pred):
    idx = [i for i, v in enumerate(f["cats"][var]) if level_pred(v)]
    return f["theta"][var][:, idx].sum(1)
comm = int(np.argmax(prob_of(f2, "portfolio", lambda v: "10.0, inf" in v)
                   + prob_of(f2, "availability", lambda v: "270.0, inf" in v)))
p_comm = f2["resp"][:, comm]
d["p_commercial"] = p_comm

print(f"\nCommercial class (K=2), independently estimated:")
print(f"  Class prevalence:            {100*f2['pi'][comm]:.1f}% of market")
print(f"  Agrees with our High tier:   ", end="")
hi = d["tier"] == "High"
overlap = 100 * ((p_comm > .5) & hi).sum() / hi.sum()
print(f"{overlap:.1f}% of our High tier also assigned >50% commercial")
print(f"  AUC of p(commercial) vs the external corporate-name label: "
      f"{roc_auc_score(d['y_corp'], p_comm):.3f}")
RESULTS["section4_lca_2class"] = {
    "class_prevalence_pct": round(100*f2["pi"][comm], 1),
    "overlap_with_high_tier_pct": round(overlap, 1),
    "auc_vs_corporate_label": round(roc_auc_score(d["y_corp"], p_comm), 3)}


# =============================================================================
hdr(5, "Validation check 4 — a synthetic city with known ground truth")
# =============================================================================
# Every check above is indirect, because real listing data has no field saying
# "this is commercial." A simulated city has that label by construction. The
# generator is calibrated from the EXTERNAL corporate-name label and the raw
# behavioural distributions it implies — it never sees the score, its weights,
# or its cut points.
srng = np.random.default_rng(20260804)

def make_city(n_hosts=12000, p_corp=0.006, p_solo=0.055, mimicry=0.0, seed=None):
    r = np.random.default_rng(seed) if seed is not None else srng
    rows = []
    kinds = r.choice(["corporate", "solo_comm", "casual"], size=n_hosts,
                      p=[p_corp, p_solo, 1 - p_corp - p_solo])
    for h, kind in enumerate(kinds):
        if kind == "corporate":
            n = int(np.clip(r.negative_binomial(2.0, 0.06), 4, 400))
            avail = r.beta(6.0, 1.6, n) * 365
            minn = r.choice([1, 2, 3, 5, 30], n, p=[.30, .30, .20, .15, .05])
            rpm = np.abs(r.gamma(1.6, 0.9, n)); entire = r.random(n) < 0.87
        elif kind == "solo_comm":
            n = 1
            avail = r.beta(5.0, 1.5, n) * 365
            minn = r.choice([1, 2, 3, 4, 30], n, p=[.34, .30, .20, .13, .03])
            rpm = np.abs(r.gamma(2.0, 0.9, n)); entire = r.random(n) < 0.80
        else:
            n = int(np.clip(r.geometric(0.80), 1, 6))
            avail = r.beta(0.55, 2.2, n) * 365
            minn = r.choice([1, 2, 3, 5, 14, 30, 60], n, p=[.20, .22, .18, .18, .10, .09, .03])
            rpm = np.abs(r.gamma(0.85, 0.42, n)); entire = r.random(n) < 0.45
        if kind != "casual" and mimicry > 0:
            avail = avail * (1 - 0.72*mimicry)
            rpm = rpm * (1 - 0.65*mimicry)
            flip = r.random(n) < mimicry * 0.85
            minn = np.where(flip, 30, minn)
        for i in range(n):
            rows.append((h, kind, n, float(np.clip(avail[i], 0, 365)), int(minn[i]),
                         float(rpm[i]), bool(entire[i])))
    city = pd.DataFrame(rows, columns=["host_id", "kind", "host_listings",
                        "availability_365", "minimum_nights", "reviews_per_month", "entire"])
    city["number_of_reviews"] = (city.reviews_per_month * r.uniform(6, 30, len(city))).round().astype(int)
    city["truth"] = (city.kind != "casual").astype(int)
    return city

def score_city(df, q50, q75, cut=60):
    P = np.select([df.host_listings >= 11, df.host_listings >= 4, df.host_listings >= 2], [1., .75, .45], 0.)
    A = np.select([df.availability_365 >= 271, df.availability_365 > 180, df.availability_365 > 90], [1., .7, .35], 0.)
    R = np.select([df.reviews_per_month >= q75, df.reviews_per_month >= q50], [1., .5], 0.)
    M = np.select([df.minimum_nights <= 3, df.minimum_nights < 30], [1., .4], 0.)
    T = df.entire.astype(float).values
    s = 30*P + 30*A + 20*R + 10*M + 10*T
    dorm = (df.availability_365 == 0) & (df.number_of_reviews == 0)
    flag = (~dorm) & (s >= cut)
    return s, flag

def evaluate_city(df, q50, q75, cut=60):
    s, flag = score_city(df, q50, q75, cut)
    p, r, f1, _ = precision_recall_fscore_support(df.truth, flag, average="binary", zero_division=0)
    return dict(precision=round(float(p), 3), recall=round(float(r), 3), f1=round(float(f1), 3),
                auc=round(float(roc_auc_score(df.truth, s)), 3))

city = make_city(seed=1)
cpos = city.loc[city.reviews_per_month > 0, "reviews_per_month"]
CQ50, CQ75 = cpos.quantile(.5), cpos.quantile(.75)
headline = evaluate_city(city, CQ50, CQ75)
print(f"Simulated city: {len(city):,} listings, {int(city.truth.sum()):,} commercial "
      f"({100*city.truth.mean():.1f}% prevalence)")
print(f"Headline: precision {headline['precision']*100:.1f}%  recall {headline['recall']*100:.1f}%  "
      f"AUC {headline['auc']:.3f}")

_, flag = score_city(city, CQ50, CQ75)
city["flag"] = flag
recall_by_type = {k: round(float(city.loc[city.kind == k, "flag"].mean()), 3)
                   for k in ("corporate", "solo_comm")}
fpr_casual = round(float(city.loc[city.kind == "casual", "flag"].mean()), 3)
print(f"Recall, corporate operators: {recall_by_type['corporate']*100:.1f}%")
print(f"Recall, solo commercial:     {recall_by_type['solo_comm']*100:.1f}%")
print(f"False-positive rate, casual: {fpr_casual*100:.1f}%")

print("\nThe incumbent rule (portfolio size alone), same ground truth:")
portfolio_rule = {}
for thresh in (2, 4, 11):
    rule = city.host_listings >= thresh
    p, r, f1, _ = precision_recall_fscore_support(city.truth, rule, average="binary", zero_division=0)
    solo_recall = float(rule[city.kind == "solo_comm"].mean())
    portfolio_rule[f">= {thresh}"] = {"precision": round(float(p), 3), "recall": round(float(r), 3),
                                       "solo_recall": round(solo_recall, 3)}
    print(f"  >= {thresh:>2} listings   precision {p*100:5.1f}%   recall {r*100:5.1f}%   "
          f"solo-operator recall {solo_recall*100:5.1f}%")

RESULTS["section5_synthetic"] = {
    "n": len(city), "commercial_n": int(city.truth.sum()),
    "prevalence_pct": round(100*city.truth.mean(), 1),
    "headline": headline, "recall_by_type": recall_by_type,
    "fpr_casual": fpr_casual, "portfolio_rule": portfolio_rule}


# =============================================================================
hdr(6, "Confusion matrix vs simple rules a city might use instead")
# =============================================================================
# Using the K=2 latent-class posterior (Section 4) as a proxy label, since it
# was estimated completely blind to the score.
proxy = (d["p_commercial"] > 0.5).astype(int)
active = ~d["dormant"]

def confusion(flag, label, mask=active):
    flag, label = flag[mask], label[mask]
    tp = int((flag & (label == 1)).sum()); fp = int((flag & (label == 0)).sum())
    fn = int((~flag & (label == 1)).sum()); tn = int((~flag & (label == 0)).sum())
    prec = tp/(tp+fp) if tp+fp else float("nan")
    rec = tp/(tp+fn) if tp+fn else float("nan")
    fpr = fp/(fp+tn) if fp+tn else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(prec, 3), recall=round(rec, 3), fpr=round(fpr, 3))

rules = {
    "Five-signal score":  d["tier"] == "High",
    "Availability alone": d["availability_365"] >= 271,
    "Portfolio alone":    d["calculated_host_listings_count"] >= 4,
    "EDA two-signal":     (d["calculated_host_listings_count"] >= 4) & (d["availability_365"] >= 271),
}
RESULTS["section6_confusion"] = {}
print(f"{'Rule':<20}{'precision':>11}{'recall':>10}{'FPR':>8}")
for name, flag in rules.items():
    c = confusion(flag.values.astype(bool), proxy.values)
    print(f"{name:<20}{c['precision']*100:>10.1f}%{c['recall']*100:>9.1f}%{c['fpr']*100:>7.1f}%")
    RESULTS["section6_confusion"][name] = c


# =============================================================================
hdr(7, "Weight and threshold sensitivity")
# =============================================================================
rng = np.random.default_rng(11)
S = SIG[["portfolio", "availability", "reviews", "min_nights", "room"]].values
w0 = np.array([30., 30., 20., 10., 10.])
base = S @ w0
dorm = d["dormant"].values
active_m = ~dorm
base_high = active_m & (base >= CUT)

N = 10000
flips = np.zeros(len(d), dtype=np.int32)
still_high = np.zeros(len(d), dtype=np.int32)
for _ in range(N):
    w = w0 * (1 + rng.uniform(-.25, .25, 5)); w = w / w.sum() * 100
    s = S @ w
    h = active_m & (s >= CUT)
    flips += (h != base_high)
    still_high += h
retained = still_high[base_high] / N
print(f"Weight perturbation: +/-25% on each weight, renormalized to 100, {N:,} draws")
print(f"  Mean High-tier retention: {100*retained.mean():.1f}%")
print(f"  5th percentile:           {100*np.percentile(retained, 5):.1f}%")
print(f"  Worst draw:               {100*retained.min():.1f}%")

sc = d.loc[active_m, "score"]
sweep = {c: int((sc >= c).sum()) for c in (55, 58, 60, 61, 65, 70)}
print(f"\nThreshold sweep (High-tier count by cutoff): {sweep}")
print(f"Listings scoring exactly 60: {int((sc == 60).sum()):,} "
      f"({100*(sc==60).sum()/base_high.sum():.1f}% of the High tier)")

cont = (30 * np.clip(np.log1p(d["calculated_host_listings_count"]) / np.log1p(20), 0, 1)
      + 30 * (d["availability_365"] / 365)
      + 20 * np.clip(d["reviews_per_month"] / Q75, 0, 1)
      + 10 * np.clip((30 - d["minimum_nights"]) / 27, 0, 1)
      + 10 * (d["room_type"] == "Entire home/apt"))
spearman = pd.Series(d["score"]).corr(cont, method="spearman")
print(f"Stepped-vs-continuous rank correlation (Spearman): {spearman:.3f}")

RESULTS["section7_sensitivity"] = {
    "weight_perturbation": {"mean_retained_pct": round(100*retained.mean(), 1),
                             "p05_retained_pct": round(100*np.percentile(retained, 5), 1),
                             "min_retained_pct": round(100*retained.min(), 1)},
    "threshold_sweep": sweep, "at_exactly_60": int((sc == 60).sum()),
    "stepped_vs_continuous_spearman": round(float(spearman), 3)}


# =============================================================================
hdr(8, "Scorecard re-derivation vs the hand-set weights")
# =============================================================================
# If we had let the data set the weights instead of using judgment, would they
# look anything like 30/30/20/10/10? Label: the external corporate-name
# evidence, which is not one of the five scoring variables.
def woe_fit(Bf, y):
    tabs, X = {}, pd.DataFrame(index=Bf.index)
    tp, tn = y.sum(), (1 - y).sum()
    for k in Bf.columns:
        g = pd.DataFrame({"b": Bf[k], "y": y}).groupby("b")["y"].agg(["size", "sum"])
        g["pos"] = (g["sum"] + .5) / (tp + .5)
        g["neg"] = (g["size"] - g["sum"] + .5) / (tn + .5)
        g["woe"] = np.log(g["pos"] / g["neg"])
        g["iv"] = (g["pos"] - g["neg"]) * g["woe"]
        tabs[k] = g; X[k] = Bf[k].map(g["woe"])
    return tabs, X

tabs, X = woe_fit(B19, d["y_corp"])
Xtr, Xte, ytr, yte = train_test_split(X, d["y_corp"], test_size=.3, random_state=RNG_SEED, stratify=d["y_corp"])
lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
auc_train, auc_test = roc_auc_score(ytr, lr.predict_proba(Xtr)[:, 1]), roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
spread = {k: float(tabs[k]["woe"].max() - tabs[k]["woe"].min()) for k in BINS}
contrib = {k: abs(lr.coef_[0][i]) * spread[k] for i, k in enumerate(X.columns)}
tot = sum(contrib.values())
derived = {k: round(100*v/tot, 1) for k, v in contrib.items()}
print(f"Scorecard fit on external corporate-name label. Test AUC: {auc_test:.3f}")
print(f"{'Signal':<14}{'hand-set':>10}{'derived':>10}")
for k in W0:
    key = "min_nights" if k == "min_nights" else k
    print(f"{k:<14}{W0[k]:>10}{derived.get(key, 0):>10.1f}")
RESULTS["section8_scorecard"] = {"test_auc": round(float(auc_test), 3),
                                  "hand_set": W0, "derived": derived}

if D25:
    try:
        d25 = pd.read_csv(D25, low_memory=False)
        d25["reviews_per_month"] = d25["reviews_per_month"].fillna(0)
        lic = d25["license"].astype(str)
        d25["y_exempt"] = (lic.str.strip().str.lower() == "exempt").astype(int)
        B25 = binned(d25)
        tabs25, X25 = woe_fit(B25, d25["y_exempt"])
        Xtr2, Xte2, ytr2, yte2 = train_test_split(X25, d25["y_exempt"], test_size=.3,
                                                    random_state=RNG_SEED, stratify=d25["y_exempt"])
        lr2 = LogisticRegression(max_iter=1000).fit(Xtr2, ytr2)
        auc_test2 = roc_auc_score(yte2, lr2.predict_proba(Xte2)[:, 1])
        print(f"\nSame exercise on the 2025 file, label = registration-exempt (Class B/hotel-type). "
              f"Test AUC: {auc_test2:.3f}")
        RESULTS["section8_scorecard"]["2025_exempt_label_test_auc"] = round(float(auc_test2), 3)
    except FileNotFoundError:
        print("\n(2025 file not found — skipping the 2025-label scorecard re-derivation.)")


# =============================================================================
hdr(9, "Six years later — the 2019 to 2025 panel")
# =============================================================================
try:
    d25 = pd.read_csv(D25, low_memory=False)
    # if D25 arrives already scored by an earlier run, drop those columns so our
    # freshly-computed ones (and the merge below) can't collide with stale copies
    d25 = d25.drop(columns=[c for c in ("score", "dormant", "tier", "reg_status")
                             if c in d25.columns])
    d25["score25"], d25["tier25"], d25["dormant25"], _ = apply_score(d25, Q50, Q75)   # frozen 2019 cut points
    lic = d25["license"].astype(str)
    d25["reg_status"] = np.where(lic.str.startswith("OSE-STRREG"), "registered",
                          np.where(lic.str.strip().str.lower() == "exempt", "exempt", "none"))

    vc25 = d25["tier25"].value_counts()
    print(f"{'Tier':<10}{'2019 %':>10}{'2025 %':>10}")
    for t in order:
        print(f"{t:<10}{100*vc.get(t,0)/len(d):>9.1f}%{100*vc25.get(t,0)/len(d25):>9.1f}%")

    panel = d.merge(d25, on="id", suffixes=("_19", "_25"))
    print(f"\nPanel (listings present in both snapshots): {len(panel):,}")

    survival = {}
    for t in order:
        n19 = int((d["tier"] == t).sum())
        surv = panel[panel["tier"] == t]
        if n19:
            survival[t] = {"n_2019": n19, "survived": len(surv),
                            "survival_pct": round(100*len(surv)/n19, 1)}
    print("Survival to 2025 by 2019 tier:")
    for t, v in survival.items():
        print(f"  {t:<10} {v['survival_pct']:5.1f}%  ({v['survived']:,} of {v['n_2019']:,})")

    min30 = {"2019": round(100*(d["minimum_nights"] >= 30).mean(), 1),
              "2025": round(100*(d25["minimum_nights"] >= 30).mean(), 1)}
    print(f"\nShare of listings with a 30+ night minimum: {min30['2019']}% (2019) -> {min30['2025']}% (2025)")

    reg_by_tier = {}
    for t in order:
        sub = d25[d25["tier25"] == t]
        if len(sub):
            reg_by_tier[t] = {k: round(100*(sub["reg_status"] == k).mean(), 1)
                               for k in ("registered", "exempt", "none")}
    print("\nRegistration status by 2025 tier:")
    for t, v in reg_by_tier.items():
        print(f"  {t:<10} registered {v['registered']:5.1f}%   exempt {v['exempt']:5.1f}%   none {v['none']:5.1f}%")

    reg_lift = {"registered_pct_high": round(100*(d25[d25["tier25"]=="High"]["reg_status"]=="registered").mean(), 2),
                "registered_pct_not_high": round(100*(d25[d25["tier25"]!="High"]["reg_status"]=="registered").mean(), 2)}
    print(f"\nRegistration lift: High tier {reg_lift['registered_pct_high']}% registered vs "
          f"{reg_lift['registered_pct_not_high']}% for everyone else")

    RESULTS["section9_panel"] = {
        "panel_n": len(panel), "tiers_2025": {t: round(100*vc25.get(t,0)/len(d25),1) for t in order},
        "survival_by_2019_tier": survival, "min30_share": min30,
        "reg_by_tier": reg_by_tier, "registration_lift": reg_lift}
except FileNotFoundError:
    print("2025 file not found at D25 — skipping the panel analysis. "
          "Set D25 to your November 2025 Inside Airbnb snapshot to run this section.")


# =============================================================================
hdr(10, "Does listing-name language add anything?")
# =============================================================================
d["name"] = d["name"].fillna("").astype(str)
BLACKLIST = set("""sonder blueground corporate inc llc ltd group properties property realty
management mgmt rental rentals suite suites hospitality stay stays stayz apartment apartments
lodging hotel hostel residence residences vacation""".split())
TOKEN = re.compile(r"[a-z]{3,}")
def toks(s): return [t for t in TOKEN.findall(str(s).lower()) if t not in BLACKLIST]
d["tok"] = d["name"].map(toks)
d["clean"] = d["tok"].map(" ".join)

S5 = SIG[["portfolio", "availability", "reviews", "min_nights", "room"]].values
tr, te = train_test_split(d.index, test_size=.3, random_state=RNG_SEED, stratify=d["y_corp"])
tfidf = TfidfVectorizer(min_df=25, ngram_range=(1, 2), sublinear_tf=True)
Xtr_t = tfidf.fit_transform(d.loc[tr, "clean"]); Xte_t = tfidf.transform(d.loc[te, "clean"])
ytr, yte = d.loc[tr, "y_corp"], d.loc[te, "y_corp"]

m_beh = LogisticRegression(max_iter=2000).fit(S5[tr], ytr)
p_beh = m_beh.predict_proba(S5[te])[:, 1]
m_text = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_t, ytr)
p_text = m_text.predict_proba(Xte_t)[:, 1]
Xtr_b = np.column_stack([S5[tr], m_text.predict_proba(Xtr_t)[:, 1]])
Xte_b = np.column_stack([S5[te], p_text])
m_both = LogisticRegression(max_iter=2000).fit(Xtr_b, ytr)
p_both = m_both.predict_proba(Xte_b)[:, 1]

auc_beh, auc_text, auc_both = (roc_auc_score(yte, p_beh), roc_auc_score(yte, p_text), roc_auc_score(yte, p_both))
print(f"AUC vs external corporate label:")
print(f"  Five behavioral signals only: {auc_beh:.3f}")
print(f"  Listing-name text only:       {auc_text:.3f}")
print(f"  Behavior + text combined:     {auc_both:.3f}")

solo = d[(d["calculated_host_listings_count"] == 1) & (~d["dormant"])].copy()
zmap = None
if len(solo):
    corp_counts = Counter(t for ts in d.loc[d["y_corp"] == 1, "tok"] for t in ts)
    rest_counts = Counter(t for ts in d.loc[d["y_corp"] == 0, "tok"] for t in ts)
    alpha = Counter(); [alpha.update(c) for c in (corp_counts, rest_counts)]
    a0 = sum(alpha.values()); n1, n2 = sum(corp_counts.values()), sum(rest_counts.values())
    rows = []
    for w, a in alpha.items():
        if a < 40: continue
        y1, y2 = corp_counts.get(w, 0), rest_counts.get(w, 0)
        o1 = np.log((y1+a)/(n1+a0-y1-a)); o2 = np.log((y2+a)/(n2+a0-y2-a))
        var = 1/(y1+a) + 1/(y2+a)
        rows.append((w, (o1-o2)/np.sqrt(var)))
    lex = pd.DataFrame(rows, columns=["token", "z"]).sort_values("z", ascending=False)
    zmap = dict(zip(lex.token, lex.z))
    d["lang"] = d["tok"].map(lambda ts: float(np.mean([zmap[t] for t in ts if t in zmap])) if any(t in zmap for t in ts) else 0.0)
    solo = d[(d["calculated_host_listings_count"] == 1) & (~d["dormant"])]
    auc_solo = roc_auc_score(solo["tier"] == "High", solo["lang"])
    print(f"\nAmong single-listing hosts with no corporate name evidence — the group a "
          f"portfolio rule can't see — does listing language predict OUR High tier?")
    print(f"  AUC: {auc_solo:.3f}  (near 0.5 means language alone can't find them; behavior can)")
    RESULTS["section10_name_signal"] = {
        "auc_behavior_only": round(float(auc_beh), 3), "auc_text_only": round(float(auc_text), 3),
        "auc_combined": round(float(auc_both), 3), "auc_text_among_solo": round(float(auc_solo), 3),
        "top_commercial_tokens": lex.head(10)["token"].tolist(),
        "top_residential_tokens": lex.tail(10)["token"].tolist()}


# =============================================================================
hdr(11, "Missing-value sensitivity")
# =============================================================================
missing_pct = round(100 * d["reviews_per_month"].isna().mean(), 2)
d_raw = pd.read_csv(D19, low_memory=False)   # reload with NaNs intact

def high_pct_with(fill):
    dd = d_raw.copy()
    dd["reviews_per_month"] = fill(dd["reviews_per_month"])
    score, tier, dormant, _ = apply_score(dd, Q50, Q75)
    return round(100 * (tier == "High").mean(), 2)

variants = {
    "fill_zero": high_pct_with(lambda s: s.fillna(0)),
    "fill_mean": high_pct_with(lambda s: s.fillna(s.mean())),
    "drop_rows": None}
dd = d_raw.dropna(subset=["reviews_per_month"]).copy()
score, tier, dormant, _ = apply_score(dd, Q50, Q75)
variants["drop_rows"] = round(100 * (tier == "High").mean(), 2)

print(f"Missing reviews_per_month: {missing_pct}% of rows")
print("High-tier share under three treatments:")
for k, v in variants.items():
    print(f"  {k:<12} {v}%")
RESULTS["section11_missing_values"] = {"missing_pct": missing_pct, "high_pct_by_treatment": variants}


# =============================================================================
hdr(12, "Disparate impact by borough")
# =============================================================================
boro = d.groupby("neighbourhood_group")["tier"].apply(lambda s: round(100*(s=="High").mean(), 1)).sort_values()
print("High-tier rate by borough:")
for b, v in boro.items():
    print(f"  {b:<15} {v:5.1f}%")
four_fifths = round(boro.min() / boro.max(), 3)
print(f"\nFour-fifths ratio (lowest / highest borough rate): {four_fifths}")
RESULTS["section12_disparate_impact"] = {"by_borough_pct": boro.to_dict(), "four_fifths_ratio": four_fifths}


# =============================================================================
hdr(13, "Model comparison with bootstrap confidence intervals")
# =============================================================================
idx_tr, idx_te = train_test_split(d.index, test_size=.3, random_state=RNG_SEED, stratify=d["y_corp"])
ytr3 = d.loc[idx_tr, "y_corp"]
tabs3, X3 = {}, pd.DataFrame(index=d.index)
tp3, tn3 = ytr3.sum(), (1 - ytr3).sum()
for k in B19.columns:
    g = pd.DataFrame({"b": B19.loc[idx_tr, k], "y": ytr3}).groupby("b")["y"].agg(["size", "sum"])
    g["woe"] = np.log(((g["sum"]+.5)/(tp3+.5)) / ((g["size"]-g["sum"]+.5)/(tn3+.5)))
    X3[k] = B19[k].map(g["woe"]).fillna(0)
lr3 = LogisticRegression(max_iter=1000).fit(X3.loc[idx_tr], ytr3)
d["scorecard"] = lr3.predict_proba(X3)[:, 1]

raw = d[["calculated_host_listings_count", "availability_365", "reviews_per_month",
         "minimum_nights", "number_of_reviews"]].copy()
raw["entire"] = (d["room_type"] == "Entire home/apt").astype(int)
print("Fitting gradient boosting comparison model...")
gb = HistGradientBoostingClassifier(max_iter=300, random_state=RNG_SEED).fit(raw.loc[idx_tr], ytr3)
d["gbm"] = gb.predict_proba(raw)[:, 1]

MODELS = [("Commercial Activity Score", "score"), ("Latent class posterior", "p_commercial"),
          ("Derived scorecard", "scorecard"), ("Gradient boosting", "gbm")]
yte3 = d.loc[idx_te, "y_corp"].values
Sm = {n: d.loc[idx_te, c].values for n, c in MODELS}
brng = np.random.default_rng(1)
NB = 800
boot = {n: np.empty(NB) for n, _ in MODELS}
n_te = len(yte3)
print(f"Bootstrapping {NB} resamples of the {n_te:,}-row test set...")
for b in range(NB):
    ii = brng.integers(0, n_te, n_te)
    if yte3[ii].sum() in (0, len(ii)):
        continue
    for n, _ in MODELS:
        boot[n][b] = roc_auc_score(yte3[ii], Sm[n][ii])

print(f"\n{'Model':<28}{'AUC':>8}{'95% CI':>18}")
boot_summary = {}
for n, _ in MODELS:
    auc = roc_auc_score(yte3, Sm[n])
    lo, hi_ = np.percentile(boot[n], 2.5), np.percentile(boot[n], 97.5)
    print(f"{n:<28}{auc:>8.3f}   [{lo:.3f}, {hi_:.3f}]")
    boot_summary[n] = {"auc": round(float(auc), 3), "ci95_low": round(float(lo), 3), "ci95_high": round(float(hi_), 3)}
RESULTS["section13_bootstrap_auc"] = boot_summary


# =============================================================================
hdr(14, "Consolidated summary of every headline number")
# =============================================================================
print(json.dumps(RESULTS, indent=2, default=str))
with open(OUT_JSON, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nFull results written to {OUT_JSON}")
