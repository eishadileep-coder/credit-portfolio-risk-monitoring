"""Historical credit-card default monitoring analysis (illustrative; not for live lending)."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score, RocCurveDisplay,
)

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs'
OUT.mkdir(exist_ok=True)
SQL_PATH = OUT / 'credit_portfolio.sqlite'


def main():
    print('1/5 Loading UCI credit-card data...', flush=True)
    data = fetch_ucirepo(id=350)
    x = data.data.features.copy()
    y = data.data.targets.iloc[:, 0].astype(int).reset_index(drop=True)
    x.columns = [str(c).strip().replace(' ', '_') for c in x.columns]
    x = x.drop(columns=[c for c in x if c.lower() in
                        ['id', 'y', 'default_payment_next_month', 'default.payment.next.month']],
               errors='ignore')

    # The UCI version of this dataset commonly uses X1 for the credit limit (LIMIT_BAL).
    # Explicitly require it rather than silently inventing a unit exposure.
    possible_limits = [c for c in x.columns if c.upper() == 'LIMIT_BAL']
    if not possible_limits and 'X1' in x.columns:
        possible_limits = ['X1']
    if len(possible_limits) != 1:
        raise ValueError(f'Cannot uniquely identify credit limit (LIMIT_BAL or X1). Columns: {list(x.columns)}')
    limit_col = possible_limits[0]
    print(f'    Credit limit field identified: {limit_col}', flush=True)

    x = x.apply(pd.to_numeric, errors='coerce')
    if x[limit_col].isna().any() or (x[limit_col] <= 0).any():
        raise ValueError('Credit limits must be present and positive; inspect the source data.')
    x = x.fillna(x.median(numeric_only=True))
    if x.isna().any().any():
        raise ValueError('Unexpected missing or non-numeric data after preprocessing.')
    x = x.reset_index(drop=True)

    print('2/5 Training logistic regression and evaluating holdout...', flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    actual = y_test.to_numpy()
    exposure = X_test[limit_col].to_numpy(dtype=float)
    metrics = {
        'ROC AUC': roc_auc_score(actual, prob),
        'PR AUC': average_precision_score(actual, prob),
        'Brier score': brier_score_loss(actual, prob),
        'Test defaults rate': float(actual.mean()),
        'Test observations': len(actual),
    }
    pd.Series(metrics, name='value').to_csv(OUT / 'model_metrics.csv')

    # Each record remains an individual historical account, not a real-world lending recommendation.
    decision = pd.DataFrame({
        'predicted_pd': prob,
        'actual_default': actual,
        'credit_limit_exposure_proxy': exposure,
    })
    decision['risk_band'] = pd.cut(
        decision.predicted_pd,
        bins=[-0.001, 0.1, 0.2, 0.4, 1.0],
        labels=['Low', 'Moderate', 'Elevated', 'High'],
    )
    if decision['risk_band'].isna().any():
        raise ValueError('Risk band missing for one or more predictions.')
    decision.to_csv(OUT / 'holdout_risk_scores.csv', index=False)

    print('3/5 Evaluating top-20% review capacity...', flush=True)
    review_count = int(np.ceil(len(decision) * 0.20))
    ranked = decision.sort_values('predicted_pd', ascending=False, kind='mergesort')
    selected = ranked.head(review_count)
    total_defaults = int(decision['actual_default'].sum())
    captured_defaults = int(selected['actual_default'].sum())
    review = pd.DataFrame([{
        'review_share': review_count / len(decision),
        'reviewed_accounts': review_count,
        'total_accounts': len(decision),
        'total_observed_defaults': total_defaults,
        'defaults_in_review_group': captured_defaults,
        'review_group_default_rate': captured_defaults / review_count,
        'default_capture_rate': captured_defaults / total_defaults if total_defaults else np.nan,
        'random_selection_expected_capture_rate': review_count / len(decision),
        'lift_vs_random': (captured_defaults / total_defaults) / (review_count / len(decision))
                          if total_defaults else np.nan,
    }])
    review.to_csv(OUT / 'review_strategy.csv', index=False)

    print('4/5 Creating illustrative loss scenarios and SQL summary...', flush=True)
    lgd = 0.60  # Fictional loss-given-default assumption; no recoveries in source.
    scenarios = pd.DataFrame([{
        'scenario': label,
        'pd_multiplier': multiplier,
        'assumed_LGD': lgd,
        'credit_limit_proxy_total_TWD': float(exposure.sum()),
        'illustrative_expected_loss_TWD': float((np.minimum(prob * multiplier, 1.0) * exposure * lgd).sum()),
    } for label, multiplier in [('Baseline', 1.0), ('Downturn', 1.25), ('Severe', 1.60)]])
    scenarios.to_csv(OUT / 'stress_scenarios.csv', index=False)
    with sqlite3.connect(SQL_PATH) as connection:
        decision.to_sql('holdout_accounts', connection, if_exists='replace', index=False)
        sql_file = BASE / 'portfolio_queries.sql'
        if not sql_file.exists():
            raise FileNotFoundError('Keep the original portfolio_queries.sql beside analyze.py')
        query = sql_file.read_text(encoding='utf-8')
        pd.read_sql_query(query, connection).to_csv(OUT / 'sql_risk_summary.csv', index=False)

    print('5/5 Saving plots and credit memo...', flush=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(actual, prob, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / 'roc_curve.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = pd.qcut(prob, 10, duplicates='drop')
    calibration = pd.DataFrame({'pd': prob, 'actual': actual, 'bin': bins}).groupby(
        'bin', observed=True
    ).agg(mean_pd=('pd', 'mean'), default_rate=('actual', 'mean'))
    ax.plot(calibration.mean_pd, calibration.default_rate, 'o-', label='Observed by decile')
    ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    ax.set(xlabel='Mean predicted probability', ylabel='Observed default rate',
           title='Holdout calibration by predicted-risk decile')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'calibration.png', dpi=160)
    plt.close(fig)

    memo = f'''# Credit Portfolio Monitoring — Historical Case Study

## Data and model
Historical source: UCI Default of Credit Card Clients (Taiwan), 30,000 observations.
Model: logistic regression; 75/25 stratified random split (not a time-based test).

## Holdout model results
- ROC-AUC: {metrics['ROC AUC']:.3f}
- Average precision (PR-AUC): {metrics['PR AUC']:.3f}
- Brier score: {metrics['Brier score']:.3f}
- Observed default rate: {metrics['Test defaults rate']:.1%}

## Hypothetical monitoring decision
With capacity to review the top 20% of accounts ranked by predicted default probability:
- Accounts selected: {review_count:,} of {len(decision):,}
- Observed defaults among selected accounts: {captured_defaults:,} of {total_defaults:,}
- Share of all observed defaults captured: {review.iloc[0]['default_capture_rate']:.1%}
- Observed default rate among reviewed accounts: {review.iloc[0]['review_group_default_rate']:.1%}
- Lift compared with random selection: {review.iloc[0]['lift_vs_random']:.2f}x

This is a retrospective prioritization test, not evidence that reviewing accounts prevents defaults.

## Illustrative loss sensitivity
Credit limits ({limit_col}) are used as an exposure proxy, **not actual exposure at default**.
The historical source reports credit limits in New Taiwan dollars (NTD/TWD).
LGD of 60% and PD multipliers of 1.25x and 1.60x are hypothetical assumptions.
The scenarios are not validated expected-credit-loss estimates or macroeconomic forecasts.
Refer to `stress_scenarios.csv` for numeric outputs.

## Limitations
The data are historical Taiwanese accounts, not a Canadian lending portfolio.
The random holdout is not out-of-time validation. Model calibration varies across risk bands.
Legal, fairness, population-drift, monitoring, governance and operational review costs
must be separately assessed before any real-world use.
'''
    (OUT / 'credit_memo.md').write_text(memo, encoding='utf-8')
    print('Finished:', OUT, flush=True)
    print(pd.Series(metrics).to_string(), flush=True)
    print('\nReview strategy:', flush=True)
    print(review.to_string(index=False), flush=True)
    print('\nCredit limits used as proxy; NOT actual exposure at default.', flush=True)


if __name__ == '__main__':
    main()
