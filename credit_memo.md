# Credit Portfolio Monitoring — Historical Case Study

## Data and model
Historical source: UCI Default of Credit Card Clients (Taiwan), 30,000 observations.
Model: logistic regression; 75/25 stratified random split (not a time-based test).

## Holdout model results
- ROC-AUC: 0.716
- Average precision (PR-AUC): 0.502
- Brier score: 0.146
- Observed default rate: 22.1%

## Hypothetical monitoring decision
With capacity to review the top 20% of accounts ranked by predicted default probability:
- Accounts selected: 1,500 of 7,500
- Observed defaults among selected accounts: 805 of 1,659
- Share of all observed defaults captured: 48.5%
- Observed default rate among reviewed accounts: 53.7%
- Lift compared with random selection: 2.43x

This is a retrospective prioritization test, not evidence that reviewing accounts prevents defaults.

## Illustrative loss sensitivity
Credit limits (X1) are used as an exposure proxy, **not actual exposure at default**.
The historical source reports credit limits in New Taiwan dollars (NTD/TWD).
LGD of 60% and PD multipliers of 1.25x and 1.60x are hypothetical assumptions.
The scenarios are not validated expected-credit-loss estimates or macroeconomic forecasts.
Refer to `stress_scenarios.csv` for numeric outputs.

## Limitations
The data are historical Taiwanese accounts, not a Canadian lending portfolio.
The random holdout is not out-of-time validation. Model calibration varies across risk bands.
Legal, fairness, population-drift, monitoring, governance and operational review costs
must be separately assessed before any real-world use.
