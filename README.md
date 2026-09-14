# NYC-Airbnb-Commercial-Host-Scoring

## Overview
A behavioral scoring model built to identify de facto commercial hotel 
operators in NYC's Airbnb market using listing behavior alone, without 
requiring access to internal platform data or confirmed licensing records.
Developed as part of the BANA 5160 Capstone at Cornell University (Team 19).
Presented at Cornell Tech, New York August 2026.

## Business Problem
New York City's Local Law 18 restricts short-term rentals to hosts 
present during guest stays, but enforcement relies on identifying 
commercial operators from public listing data. Simple portfolio-count 
rules catch 0% of single-apartment commercial operators at every 
threshold tested.

## Solution
Built a five-signal weighted composite scoring model (0 to 100, four tiers) 
using availability patterns, review cadence, minimum night requirements, 
portfolio size, and price positioning as behavioral signals.

## Validation
Six independent methods, sharing no common assumptions:
- Known-operator face validity (Sonder, Blueground)
- Log-price OLS on variables the score never sees
- EM latent-class model with BIC selection across K=2,3,4
- Synthetic city ground truth (16,926 listings)
- Confusion matrix vs three naive baseline rules
- Four-fifths disparate-impact fairness audit
- Bootstrap 95% CIs on AUC across four model families (0.908 to 0.988)

## Key Findings
- Model recovers 71% of single-apartment commercial operators vs 0% 
  for portfolio-count rules
- Temporal holdout: minimum-stay listings of 30+ nights rose from 
  9.2% to 85.3% post Local Law 18

## Tools and Methods
- Python, pandas, scikit-learn
- Composite scoring, EM latent-class analysis
- Bootstrap confidence intervals, sensitivity analysis
- React decision dashboard with geospatial rendering

## Dataset
AB_NYC_2019.csv — 48,895 listings, 16 variables
