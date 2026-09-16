"""Leakage-safe preprocessing for the Toyota Corolla regression dashboard."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_transformer(X, standardize=False):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        numeric_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", num_pipe, numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_cols, categorical_cols


def prepare_regression_split(df, feature_cols, target_col, test_size=0.2, standardize=False, seed=42):
    cols = [c for c in feature_cols if c in df.columns and c not in {"Id", target_col}]
    work = df[cols + [target_col]].copy()
    y = pd.to_numeric(work[target_col], errors="coerce")
    valid = y.notna()
    work = work.loc[valid].copy()
    original_indices = work.index.to_numpy()
    work = work.reset_index(drop=True)
    y = y.loc[valid].astype(float).reset_index(drop=True)
    X = work[cols].copy()

    idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=test_size, random_state=seed)
    idx_train = np.sort(idx_train)
    idx_test = np.sort(idx_test)

    X_train_raw = X.iloc[idx_train]
    X_test_raw = X.iloc[idx_test]
    y_train = y.iloc[idx_train]
    y_test = y.iloc[idx_test]

    transformer, numeric_cols, categorical_cols = _make_transformer(X_train_raw, standardize)
    X_train_arr = transformer.fit_transform(X_train_raw)
    X_test_arr = transformer.transform(X_test_raw)
    X_all_arr = transformer.transform(X)
    names = transformer.get_feature_names_out().tolist()

    X_train = pd.DataFrame(X_train_arr, columns=names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=names, index=X_test_raw.index)
    X_all = pd.DataFrame(X_all_arr, columns=names, index=X.index)

    model = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    pred_train = model.predict(sm.add_constant(X_train, has_constant="add"))
    pred_test = model.predict(sm.add_constant(X_test, has_constant="add"))

    # Missing count is reported from the raw selected features, while all fitted
    # imputation parameters come only from training data.
    missing_before = int(X_train_raw.isna().sum().sum() + X_test_raw.isna().sum().sum())
    report = {
        "missing_imputed": missing_before,
        "cat_cols_encoded": categorical_cols,
        "numeric_cols": numeric_cols,
        "n_features_out": len(names),
        "scaled": bool(standardize),
        "fit_scope": "Training data only",
    }

    return {
        "X_all": X_all,
        "y_all": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pred_train": pred_train,
        "row_indices_train": original_indices[idx_train],
        "row_indices_test": original_indices[idx_test],
        "pred_test": pred_test,
        "model": model,
        "transformer": transformer,
        "feature_names": names,
        "report": report,
    }


def transform_new_data(X_new, transformer):
    arr = transformer.transform(X_new)
    names = transformer.get_feature_names_out().tolist()
    return pd.DataFrame(arr, columns=names, index=X_new.index)


def preprocess(df, feature_cols, target_col, standardize=False):
    """Backward-compatible helper for non-model exploratory uses."""
    cols = [c for c in feature_cols if c in df.columns and c not in {"Id", target_col}]
    X_raw = df[cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series(dtype=float)
    transformer, numeric_cols, categorical_cols = _make_transformer(X_raw, standardize)
    X_arr = transformer.fit_transform(X_raw)
    names = transformer.get_feature_names_out().tolist()
    X_enc = pd.DataFrame(X_arr, columns=names, index=X_raw.index)
    report = {"missing_imputed": int(X_raw.isna().sum().sum()), "cat_cols_encoded": categorical_cols,
              "n_features_out": len(names), "scaled": bool(standardize)}
    return X_enc, y, names, report
