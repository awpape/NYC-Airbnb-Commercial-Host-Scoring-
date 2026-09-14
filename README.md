# NYC Airbnb Commercial Activity Score

A behavioral scoring model and interactive enforcement dashboard that
identifies de facto commercial hotel operators in NYC's short-term rental
market using public listing data alone — no access to internal platform
data or confirmed licensing records required.

Built on the Inside Airbnb NYC 2019 dataset (48,895 listings).

---

## The Problem

New York City's Local Law 18 restricts short-term rentals to hosts present
during guest stays. Enforcement depends on identifying commercial operators
from public listing behavior. The challenge: simple portfolio-count rules
catch **0% of single-apartment commercial operators** at every threshold tested.

---

## The Solution

A five-signal weighted composite score (0–100, four tiers):

| Signal | Weight | Logic |
|---|---|---|
| Portfolio size | 30% | Hosts with 4+ listings score higher |
| Availability | 30% | 271+ days/year signals hotel-like operation |
| Review cadence | 20% | High review frequency indicates commercial turnover |
| Minimum nights | 10% | Short minimums (≤ 3 nights) signal guest-facing commercial use |
| Room type | 10% | Entire home/apt listings score higher |

**Tiers:**
- 🔴 **High** (≥ 60) — suspected commercial operator, priority for enforcement
- 🟡 **Moderate** (40–59) — worth monitoring
- 🟢 **Low** (< 40) — likely casual host
- ⚫ **Dormant** — zero availability and zero reviews

---

## Key Results

| Finding | Value |
|---|---|
| Listings analyzed | 48,895 |
| Model recovery of single-apartment commercial operators | **71%** |
| Portfolio-count rule recovery | **0%** at every threshold |
| AUC range across four model families (bootstrap 95% CI) | 0.908–0.988 |

**Temporal holdout:** Applying the frozen 2019 model to a November 2025
post-regulation snapshot, minimum-stay listings of 30+ nights rose from
**9.2% to 85.3%**, confirming operator behavioral adaptation to Local Law 18.

---

## Validation — Six Independent Methods

1. **Face validity** — known commercial operators (Sonder, Blueground) score in the top tier
2. **Price test** — log-price OLS on a variable the score never sees
3. **Latent class analysis** — EM algorithm with BIC model selection across K=2,3,4
4. **Synthetic city** — constructed ground truth with 16,926 listings, frozen score applied
5. **Confusion matrix** — vs three naive baseline rules a city might use instead
6. **Fairness audit** — four-fifths disparate-impact test across NYC boroughs

Plus: 10,000-draw weight perturbation (±25%), threshold sweeps, and
missing-value sensitivity analysis.

---

## Project Structure

```
├── scoring.py                     # Five-signal scoring model
├── generate_synthetic_listings.py # Synthetic listing cohort generator
├── app.py                         # Streamlit enforcement dashboard
├── nyc_airbnb_commercial_score.py # Full analysis pipeline (all 14 sections)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Launch the demo dashboard:**
```bash
streamlit run app.py
```

Opens at `localhost:8501`. Uses synthetic listings by default — no data download required.

**3. Score real listings:**

Download `listings.csv` from [Inside Airbnb](http://insideairbnb.com/get-the-data)
and upload it using the sidebar file uploader in the dashboard.

**4. Run the full analysis pipeline:**

Download the Inside Airbnb NYC snapshots and update the paths in
`nyc_airbnb_commercial_score.py`, then:
```bash
python nyc_airbnb_commercial_score.py
```

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly

---

## Author

Alain William Pape
