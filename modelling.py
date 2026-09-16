"""
Regression training, metrics, and coefficient utilities.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def train_ols(X, y, test_size=0.20, seed=42):
    # Remove rows where target is missing.
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const = sm.add_constant(X_test, has_constant="add")

    model = sm.OLS(y_train, X_train_const).fit()
    y_pred_train = model.predict(X_train_const)
    y_pred_test = model.predict(X_test_const)

    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test


def compute_metrics(y_true, y_pred, n_features=1):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

    n = len(y_true)
    denom = n - n_features - 1
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / denom if denom > 0 else np.nan

    nz = y_true != 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else np.nan

    return {"r2": r2, "adj_r2": adj_r2, "mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


def extract_coefficients(model, feature_names=None):
    rows = []
    conf = model.conf_int()
    for name in model.params.index:
        rows.append({
            "Feature": "Intercept" if name == "const" else name,
            "Coefficient": float(model.params[name]),
            "t-Statistic": float(model.tvalues[name]),
            "p-value": float(model.pvalues[name]),
            "CI Lower (95%)": float(conf.loc[name, 0]),
            "CI Upper (95%)": float(conf.loc[name, 1]),
        })
    return pd.DataFrame(rows)


def build_equation(coef_df, target="Price", max_terms=18):
    intercept = coef_df.loc[coef_df["Feature"] == "Intercept", "Coefficient"]
    value = float(intercept.iloc[0]) if len(intercept) else 0.0
    terms = [f"{value:,.2f}"]
    body = coef_df[coef_df["Feature"] != "Intercept"].copy()
    body["abs"] = body["Coefficient"].abs()
    body = body.sort_values("abs", ascending=False).head(max_terms)
    for _, row in body.iterrows():
        sign = "+" if row["Coefficient"] >= 0 else "-"
        terms.append(f"{sign} {abs(row['Coefficient']):.4f}·{row['Feature']}")
    suffix = " + …" if len(coef_df) - 1 > max_terms else ""
    return f"**{target}** = " + " ".join(terms) + suffix
