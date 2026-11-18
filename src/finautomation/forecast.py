"""Chronological revenue forecasting and like-for-like evaluation."""

import math
from collections import defaultdict

from sklearn.linear_model import LinearRegression

from .periods import month_index, shift_months

FORECAST_COLUMNS = (
    "target_period_end", "kind", "history_count", "y_true", "forecast", "naive",
    "naive_period_end", "error", "forecast_growth", "mape_excluded", "mape_exclusion_reason", "status", "reason",
)


def fit_trend(history):
    scale = max(abs(value) for value in history) or 1
    model = LinearRegression().fit([[i] for i in range(len(history))], [value / scale for value in history])
    result = float(model.predict([[len(history)]])[0]) * scale
    return result if math.isfinite(result) else None


def forecast_row(statement, target, history, minimum, *, evaluation, last_observed, actual=None, ambiguous=False):
    naive_period_end, naive = last_observed
    enough_history = len(history) >= minimum
    predicted = fit_trend(history) if enough_history and not ambiguous else None
    if ambiguous:
        reason = "ambiguous_period"
    elif not enough_history:
        reason = "insufficient_contiguous_history"
    elif predicted is None:
        reason = "nonfinite_prediction"
    else:
        reason = ""
    error = predicted - actual if predicted is not None and actual is not None else None
    growth = predicted / naive - 1 if predicted is not None and naive not in (None, 0) else None
    if growth is not None and not math.isfinite(growth):
        growth = None
    if not evaluation:
        exclusion = "future_period"
    elif actual is None:
        exclusion = "missing_actual"
    elif actual <= 0:
        exclusion = "nonpositive_actual"
    else:
        exclusion = reason
    status = "not_evaluable" if reason else "ok"
    return {
        **statement.instrument.columns(), "period_type": statement.period_type, "target_period_end": target,
        "kind": "backtest" if evaluation else "next", "history_count": len(history), "y_true": actual,
        "forecast": predicted, "naive": naive, "naive_period_end": naive_period_end,
        "error": error if error is None or math.isfinite(error) else None,
        "forecast_growth": growth, "mape_excluded": bool(exclusion), "mape_exclusion_reason": exclusion,
        "status": status, "reason": reason,
    }


def forecast_revenue(statements, minimum):
    grouped = defaultdict(list)
    for item in statements:
        grouped[item.instrument, item.period_type].append(item)
    rows = []
    for (_, frequency), series in sorted(grouped.items()):
        step = 3 if frequency == "Q" else 12
        buckets = defaultdict(list)
        for item in series:
            buckets[month_index(item.period_end)].append(item)
        history = []
        previous = None
        latest = None
        last_observed = (None, None)
        for month, candidates in sorted(buckets.items()):
            latest = max(candidates, key=lambda item: item.period_end)
            actual = latest.fields["revenue"].value if len(candidates) == 1 else None
            if previous is None or month != previous + step:
                history.clear()
            rows.append(forecast_row(latest, latest.period_end, history, minimum,
                                     evaluation=True, last_observed=last_observed, actual=actual,
                                     ambiguous=len(candidates) != 1))
            if actual is None:
                history.clear()
            else:
                history.append(actual)
                last_observed = (latest.period_end, actual)
            previous = month
        target = shift_months(latest.period_end, step, preserve_month_end=True)
        rows.append(forecast_row(latest, target, history, minimum, evaluation=False, last_observed=last_observed))
    return rows, score(rows, minimum)


def score(rows, minimum):
    backtests = [row for row in rows if row["kind"] == "backtest"]
    scored = []
    for row in backtests:
        if row["mape_excluded"]:
            continue
        actual = row["y_true"]
        model_ape = abs(row["forecast"] - actual) / actual
        naive_ape = abs(row["naive"] - actual) / actual
        if math.isfinite(model_ape) and math.isfinite(naive_ape):
            scored.append((model_ape, naive_ape))
        else:
            row["mape_excluded"] = True
            row["mape_exclusion_reason"] = "nonfinite_error"
    count = len(scored)
    def mean(errors):
        scale = max(errors)
        return (math.fsum(value / scale for value in errors) / count) * scale if scale else 0.0

    model = mean([item[0] for item in scored]) if count else None
    naive = mean([item[1] for item in scored]) if count else None
    status = "not_evaluable"
    if count:
        status = "pass" if model <= 0.20 or model < naive else "fail"
    return {
        "model": "linear_trend", "minimum_training_periods": minimum, "threshold_mape": 0.20,
        "mape_model": model, "mape_naive": naive, "eligible_points": count,
        "excluded_points": len(backtests) - count, "status": status,
    }
