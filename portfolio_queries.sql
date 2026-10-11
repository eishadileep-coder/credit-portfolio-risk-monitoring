SELECT risk_band,
       COUNT(*) AS accounts,
       ROUND(AVG(predicted_pd), 4) AS mean_predicted_default_probability,
       ROUND(AVG(actual_default), 4) AS observed_default_rate,
       ROUND(SUM(credit_limit_exposure_proxy), 2) AS total_limit_proxy
FROM holdout_accounts
GROUP BY risk_band
ORDER BY mean_predicted_default_probability;
