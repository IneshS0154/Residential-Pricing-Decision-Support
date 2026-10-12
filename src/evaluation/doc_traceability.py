"""Read-only check that the numbers quoted in the stage documents equal the values stored in the result tables / artifacts.

    python -m src.evaluation.doc_traceability        # prints the table and writes reports/tables/doc_traceability.csv

Nothing is fitted or recomputed; each row compares a number typed in a document (rounded as written there) with the value in
the file that produced it, using a tolerance of the last printed digit. Business-facing numbers in notebook 10 are covered separately
by the evidence registry (src/evaluation/evidence.py) and its consistency checks.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]


def T(f):
    return pd.read_csv(R / "reports" / "tables" / f"{f}.csv")


def build():
    rows = []

    def chk(doc, claim, quoted, actual, source, tol):
        ok = abs(float(quoted) - float(actual)) <= tol
        rows.append((doc, claim, quoted, round(float(actual), 6), source, "MATCH" if ok else "MISMATCH"))



    bc = T('baseline_comparison').set_index('Model')
    src = 'reports/tables/baseline_comparison.csv'
    for m, q in [('ElasticNet', .1289), ('Lasso', .1295), ('Gradient Boosting', .1311), ('Ridge', .1322), ('OLS', .1341), ('Random Forest', .1423), ('Dummy (mean)', .3992)]:
        chk('baseline_comparison.md', f'{m} Log-RMSE', q, bc.loc[m, 'Log-RMSE Mean'], src, 5e-5)
    chk('baseline_comparison.md', 'Gradient Boosting USD RMSE ~28,248', 28248, bc.loc['Gradient Boosting', 'RMSE USD'], src, 1)

    tv = T('tuning_vs_baseline').set_index('Model')
    src = 'reports/tables/tuning_vs_baseline.csv'
    chk('tuning.md', 'tuned GBR Log-RMSE 0.1260', .1260, tv.loc['Gradient Boosting', 'Tuned Log-RMSE'], src, 5e-5)
    chk('tuning.md', 'tuned GBR MAE 15,535', 15535, tv.loc['Gradient Boosting', 'MAE USD'], src, 1)
    chk('tuning.md', 'tuned GBR RMSE 26,774', 26774, tv.loc['Gradient Boosting', 'RMSE USD'], src, 1)
    chk('tuning.md', 'tuned GBR MAPE 8.76', 8.76, tv.loc['Gradient Boosting', 'MAPE %'], src, 5e-3)
    chk('tuning.md', 'tuned GBR R2 0.884', .884, tv.loc['Gradient Boosting', 'R2 USD'], src, 5e-4)
    chk('tuning.md', 'tuned ElasticNet Log-RMSE 0.1307', .1307, tv.loc['ElasticNet', 'Tuned Log-RMSE'], src, 5e-5)
    chk('tuning.md', 'GBR tuned - baseline = -0.0051', -.0051, tv.loc['Gradient Boosting', 'Difference (tuned - baseline)'], src, 5e-5)
    chk('tuning.md', 'ElasticNet tuned - baseline = +0.0018', .0018, tv.loc['ElasticNet', 'Difference (tuned - baseline)'], src, 5e-5)
    pr = json.loads((R / 'artifacts' / 'tuning' / 'final_params_tuning_v1.json').read_text(encoding='utf-8'))
    chk('tuning.md', 'final GBR learning_rate 0.0242', .0242, pr['gbr']['params']['learning_rate'], 'artifacts/tuning/final_params_tuning_v1.json', 5e-5)
    chk('tuning.md', 'final GBR subsample 0.655', .655, pr['gbr']['params']['subsample'], 'artifacts/tuning/final_params_tuning_v1.json', 5e-4)
    chk('tuning.md', 'final GBR inner score 0.1271', .1271, pr['gbr']['inner_cv_best_logrmse'], 'artifacts/tuning/final_params_tuning_v1.json', 5e-5)

    fa = T('final_eval_accuracy').set_index(T('final_eval_accuracy').columns[0])
    src = 'reports/tables/final_eval_accuracy.csv'
    chk('final_evaluation.md', 'GBR Log-RMSE 0.1260', .1260, fa.loc['Gradient Boosting (tuned)', 'Log-RMSE mean'], src, 5e-5)
    chk('final_evaluation.md', 'ElasticNet Log-RMSE 0.1307', .1307, fa.loc['ElasticNet (tuned)', 'Log-RMSE mean'], src, 5e-5)
    chk('final_evaluation.md', 'Lasso Log-RMSE 0.1295', .1295, fa.loc['Lasso (fixed)', 'Log-RMSE mean'], src, 5e-5)
    chk('final_evaluation.md', 'GBR fold SD 0.0126', .0126, fa.loc['Gradient Boosting (tuned)', 'Log-RMSE SD'], src, 5e-5)

    im = T('intervals_main_comparison').set_index(T('intervals_main_comparison').columns[0])
    src = 'reports/tables/intervals_main_comparison.csv'
    chk('prediction_intervals.md', 'quantile GB coverage 65.1%', 65.1, 100 * im.loc['Quantile GB', 'Empirical coverage'], src, .05)
    chk('prediction_intervals.md', 'CQR coverage 80.7%', 80.7, 100 * im.loc['CQR', 'Empirical coverage'], src, .05)
    chk('prediction_intervals.md', 'quantile GB mean width $43.2k', 43.2, im.loc['Quantile GB', 'Mean width USD'] / 1000, src, .05)
    chk('prediction_intervals.md', 'CQR mean width $56.7k', 56.7, im.loc['CQR', 'Mean width USD'] / 1000, src, .05)
    chk('prediction_intervals.md', 'CQR median relative width 27.0%', 27.0, im.loc['CQR', 'Median relative width %'], src, .05)
    it = T('intervals_temporal').set_index('method')
    chk('prediction_intervals.md', 'temporal CQR 76.7%', 76.7, 100 * it.loc['CQR', 'coverage (mean of 20 splits)'], 'reports/tables/intervals_temporal.csv', .05)
    chk('prediction_intervals.md', 'temporal raw 62.7%', 62.7, 100 * it.loc['Quantile GB', 'coverage (mean of 20 splits)'], 'reports/tables/intervals_temporal.csv', .05)

    rh = T('routing_heldout_policy').set_index('scenario')
    P = rh.loc[[i for i in rh.index if i.startswith('P:')][0]]
    src = 'reports/tables/routing_heldout_policy.csv'
    chk('routing.md', 'AUTO share 45.0%', 45.0, P['AUTO %'], src, .05)
    chk('routing.md', 'AUTO MAE $10.8k', 10.8, P['AUTO MAE'] / 1000, src, .05)
    chk('routing.md', 'AUTO MAPE 6.49%', 6.49, P['AUTO MAPE'], src, .005)
    chk('routing.md', 'AUTO coverage 78.5%', 78.5, 100 * P['AUTO coverage'], src, .05)
    chk('routing.md', 'ESCALATE share 55.0%', 55.0, P['ESCALATE %'], src, .05)
    chk('routing.md', 'ESCALATE MAE $19.4k', 19.4, P['ESCALATE MAE'] / 1000, src, .05)
    chk('routing.md', 'ESCALATE MAPE 10.66%', 10.66, P['ESCALATE MAPE'], src, .005)
    chk('routing.md', 'overall MAPE 8.76%', 8.76, P['overall MAPE'], src, .005)
    rt = T('routing_temporal').set_index('scenario')
    src = 'reports/tables/routing_temporal.csv'
    chk('routing.md', 'temporal tau 0.32', .32, rt.loc['P', 'tau_T'], src, 1e-9)
    chk('routing.md', 'temporal AUTO 64.5%', 64.5, rt.loc['P', 'AUTO % (mean)'], src, .05)
    chk('routing.md', 'temporal AUTO MAPE 8.16%', 8.16, rt.loc['P', 'AUTO MAPE (mean)'], src, .005)
    chk('routing.md', 'temporal AUTO coverage 75.5%', 75.5, 100 * rt.loc['P', 'AUTO coverage (mean)'], src, .05)

    sc = T('segments_temporal_comparison')
    for m, q in (('Gradient Boosting', .1317), ('ElasticNet', .1316)):
        g = sc[(sc.Model == m) & sc['Evaluation Protocol'].str.contains('historical-only')]
        chk('segment_temporal.md', f'temporal {m} Log-RMSE (historical-only tuned) {q}', q, g['Log-RMSE'].iloc[0], 'reports/tables/segments_temporal_comparison.csv', 5e-5)

    fs = T('feature_set_decision')
    src = 'reports/tables/feature_set_decision.csv'
    for (cfg, m), q in {('A', 'Ridge (alpha 1)'): .1321, ('B', 'Ridge (alpha 1)'): .1369, ('A', 'Gradient Boosting (defaults)'): .1311, ('B', 'Gradient Boosting (defaults)'): .1293, ('A', 'Random Forest'): .1424, ('B', 'Random Forest'): .1411}.items():
        a = fs[(fs['Feature configuration'].str[0] == cfg) & (fs.Model == m)]['Mean Log-RMSE'].iloc[0]
        chk('feature_set_decision.md', f'{cfg} {m} Log-RMSE', q, a, src, 5e-5)

    d2 = T('decision_d2_outliers').set_index('model')
    src = 'reports/tables/decision_d2_outliers.csv'
    chk('eda_insight_log.md (D2)', 'Ridge gain 0.00416', .00416, d2.loc['ridge', 'gain (keep-drop)'], src, 5e-6)
    chk('eda_insight_log.md (D2)', 'Ridge gain/SE 4.2', 4.2, d2.loc['ridge', 'gain/SE'], src, .05)
    chk('eda_insight_log.md (D2)', 'GBM gain 0.00147', .00147, d2.loc['gbm', 'gain (keep-drop)'], src, 5e-6)
    chk('eda_insight_log.md (D2)', 'GBM gain/SE 2.5', 2.5, d2.loc['gbm', 'gain/SE'], src, .05)
    pl = T('decision_d2_placebo')['placebo_gain']
    chk('eda_insight_log.md (D2)', 'placebo mean +0.00001', .00001, pl.mean(), 'reports/tables/decision_d2_placebo.csv', 1e-5)
    chk('eda_insight_log.md (D2)', 'placebo max 0.00016', .00016, pl.max(), 'reports/tables/decision_d2_placebo.csv', 5e-5)
    d3 = T('decision_d3_neighborhood_encoding')
    src = 'reports/tables/decision_d3_neighborhood_encoding.csv'
    g3 = lambda m, e: d3[(d3.model == m) & (d3['Neighborhood encoding'] == e)].iloc[0]
    chk('eda_insight_log.md (D3)', 'Ridge onehot 0.1309', .1309, g3('ridge', 'onehot')['RMSE mean'], src, 5e-5)
    chk('eda_insight_log.md (D3)', 'Ridge target 0.1338', .1338, g3('ridge', 'target')['RMSE mean'], src, 5e-5)
    chk('eda_insight_log.md (D3)', 'GBM onehot 0.1290', .1290, g3('gbm', 'onehot')['RMSE mean'], src, 5e-5)
    chk('eda_insight_log.md (D3)', 'GBM target 0.1269', .1269, g3('gbm', 'target')['RMSE mean'], src, 5e-5)
    chk('eda_insight_log.md (D3)', 'GBM both 0.1258', .1258, g3('gbm', 'both')['RMSE mean'], src, 5e-5)
    chk('eda_insight_log.md (D3)', 'Ridge no-Neighborhood cost 0.0063', .0063, -g3('ridge', 'none')['gain vs one-hot'], src, 5e-5)
    chk('eda_insight_log.md (D3)', 'GBM no-Neighborhood cost 0.0019', .0019, -g3('gbm', 'none')['gain vs one-hot'], src, 5e-5)

    ps = T('prep_strategy_comparison').set_index(T('prep_strategy_comparison').columns[0])
    for k, q in [('baseline', .1309), ('neighborhood', .1310), ('neighborhood_indicator', .1310), ('knn', .1309)]:
        chk('preprocessing_log.md', f'strategy {k} RMSE', q, ps.loc[k, 'RMSE_mean'], 'reports/tables/prep_strategy_comparison.csv', 6e-5)

    ab = T('ablation_sale_context')
    src = 'reports/tables/ablation_sale_context.csv (earlier-environment values; see docs/reproducibility_notes.md)'
    chk('eda_insight_log.md (F22)', 'CV Log-RMSE, all predictors 0.1311', .1311, ab.iloc[0, 1], src, 5e-5)
    chk('eda_insight_log.md (F22)', 'drop all four sale-context columns 0.1319', .1319, ab.iloc[1, 1], src, 5e-5)
    chk('eda_insight_log.md (F22)', 'cost of dropping them +0.0008', .0008, ab.iloc[1, 3], src, 5e-5)

    we = T('worked_examples').set_index('Id')
    src = 'reports/tables/worked_examples.csv'
    for Id, price, rel, err in [(129, 147169, .208, 7831), (42, 158087, .2999, 11913), (785, 136283, .311, -8283), (750, 83447, .842, 14553)]:
        chk('worked_examples.md', f'Id {Id} predicted price', price, we.loc[Id, 'predicted_price'], src, 1)
        chk('worked_examples.md', f'Id {Id} relative width', rel, we.loc[Id, 'relative_width'], src, 5e-4)
        chk('worked_examples.md', f'Id {Id} retrospective error', err, we.loc[Id, 'error_retrospective'], src, 1)
    wd = T('worked_examples_decomposition')
    src = 'reports/tables/worked_examples_decomposition.csv'
    for Id, tot in [(129, -33126), (42, -2411), (785, -42915), (750, -83809)]:
        chk('worked_examples.md', f'Id {Id} total decomposition change (USD)', tot, wd[wd.Id == Id].usd_contribution.sum(), src, 1.5)
    from src.evaluation.worked_examples import model_agreement
    ag = model_agreement()
    chk('worked_examples.md', 'GBR vs ElasticNet OOF Pearson 0.983', .983, ag['pearson_log_prediction'], 'artifacts/final_evaluation/oof_predictions.csv.gz', 5e-4)
    chk('worked_examples.md', 'GBR vs ElasticNet OOF RMSE of log difference 0.071', .071, ag['rmse_log_difference'], 'artifacts/final_evaluation/oof_predictions.csv.gz', 5e-4)
    chk('worked_examples.md', 'median absolute price difference 3.9 %', 3.9, ag['median_abs_pct_price_difference'], 'artifacts/final_evaluation/oof_predictions.csv.gz', .05)

    return pd.DataFrame(rows, columns=["Document", "Quoted claim", "Quoted value", "Value in source", "Source", "Result"])


if __name__ == "__main__":
    df = build()
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 300)
    print(df.to_string(index=False))
    print(f"Total {len(df)}  MATCH {(df.Result == 'MATCH').sum()}  MISMATCH {(df.Result == 'MISMATCH').sum()}")
    df.to_csv(R / "reports" / "tables" / "doc_traceability.csv", index=False)
