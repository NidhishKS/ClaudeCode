"""
inflation_drivers_analysis.py
==============================
Analyzes statistical drivers of core goods CPI inflation using:
  1. Shapley Value Regression (manual R² decomposition)
  2. VAR + Forecast Error Variance Decomposition (FEVD)
  3. Gradient Boosting (XGBoost) + SHAP values
  4. Markov-Switching Regression

Data sourced from FRED. Requires FRED_API_KEY environment variable.
"""

import os
import sys
import warnings
import itertools
import math as _math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from fredapi import Fred
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import statsmodels.api as sm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 0. FRED DATA COLLECTION
# ──────────────────────────────────────────────────────────────

def fetch_data():
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        sys.exit("ERROR: FRED_API_KEY environment variable not set.")

    fred = Fred(api_key=api_key)

    print("Fetching FRED series (full history)...")
    core_goods_raw  = fred.get_series("CUSR0000SACL1E")
    energy_raw      = fred.get_series("CPIENGSL")
    inf_exp         = fred.get_series("MICH")
    wages_raw       = fred.get_series("A576RC1")

    # Convert all to monthly period index for clean alignment
    def to_monthly(s):
        s.index = pd.to_datetime(s.index)
        s = s.resample("MS").last()          # month-start, last obs in month
        return s

    core_goods_raw = to_monthly(core_goods_raw)
    energy_raw     = to_monthly(energy_raw)
    inf_exp        = to_monthly(inf_exp)
    wages_raw      = to_monthly(wages_raw)

    # YoY % changes
    core_goods_yoy = core_goods_raw.pct_change(12) * 100
    energy_yoy     = energy_raw.pct_change(12)     * 100
    wages_yoy      = wages_raw.pct_change(12)      * 100

    df = pd.DataFrame({
        "core_goods_yoy": core_goods_yoy,
        "energy_yoy":     energy_yoy,
        "inf_exp":        inf_exp,
        "wages_yoy":      wages_yoy,
    }).dropna()

    print(f"Aligned dataset: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} observations)")
    return df


# ──────────────────────────────────────────────────────────────
# 1. STATIONARITY CHECKS & FIRST-DIFFERENCING IF NEEDED
# ──────────────────────────────────────────────────────────────

def adf_test(series, name, alpha=0.05):
    result = adfuller(series.dropna(), autolag="AIC")
    pval   = result[1]
    stat   = "STATIONARY" if pval < alpha else "NON-STATIONARY"
    print(f"  ADF {name:20s}: p={pval:.4f}  → {stat}")
    return pval < alpha

def prepare_stationary(df):
    print("\n── Stationarity Tests (ADF, α=0.05) ──")
    stationary_df = pd.DataFrame(index=df.index)
    diff_flags    = {}

    for col in df.columns:
        is_stat = adf_test(df[col], col)
        diff_flags[col] = not is_stat
        if is_stat:
            stationary_df[col] = df[col]
        else:
            # First-difference non-stationary series and re-test
            d1 = df[col].diff()
            is_stat2 = adf_test(d1, f"Δ{col}")
            stationary_df[col] = d1
            # Economic note: if a series needs differencing it means we are
            # modeling the *change* in that variable, not its level
            if not is_stat2:
                print(f"  WARNING: {col} still non-stationary after first diff — check series")

    stationary_df = stationary_df.dropna()
    print(f"\nFinal modelling dataset: {len(stationary_df)} obs after differencing/NA removal")
    return stationary_df, diff_flags


# ──────────────────────────────────────────────────────────────
# 2. METHOD 1 — SHAPLEY VALUE R² DECOMPOSITION
# ──────────────────────────────────────────────────────────────
# Shapley values apportion the model R² to each predictor by
# averaging its marginal R² contribution across ALL possible
# orderings of predictors — giving a fair, order-independent
# decomposition. High Shapley share → strong unique driver.

def shapley_r2(df):
    print("\n══════════════════════════════════════")
    print("METHOD 1 — SHAPLEY VALUE R² DECOMPOSITION")
    print("══════════════════════════════════════")

    y    = df["core_goods_yoy"].values
    Xcols = ["energy_yoy", "inf_exp", "wages_yoy"]
    X    = df[Xcols].values
    n    = len(Xcols)

    def ols_r2(y, X_sub):
        """OLS R² for a subset of columns."""
        if X_sub.shape[1] == 0:
            return 0.0
        Xa = sm.add_constant(X_sub, has_constant="add")
        try:
            res = sm.OLS(y, Xa).fit()
            return max(res.rsquared, 0.0)
        except Exception:
            return 0.0

    # Enumerate all 2^n subsets; compute R² for each
    subset_r2 = {}
    for size in range(n + 1):
        for subset in itertools.combinations(range(n), size):
            key = frozenset(subset)
            if len(subset) == 0:
                subset_r2[key] = 0.0
            else:
                subset_r2[key] = ols_r2(y, X[:, list(subset)])

    # Shapley value for predictor i = weighted average of marginal contributions
    shapley_vals = np.zeros(n)
    for i in range(n):
        sv = 0.0
        other = [j for j in range(n) if j != i]
        for size in range(n):
            for subset in itertools.combinations(other, size):
                s_without = frozenset(subset)
                s_with    = frozenset(subset + (i,))
                weight    = (
                    _math.factorial(size) *
                    _math.factorial(n - size - 1) /
                    _math.factorial(n)
                )
                sv += weight * (subset_r2[s_with] - subset_r2[s_without])
        shapley_vals[i] = sv

    total_r2   = subset_r2[frozenset(range(n))]
    pct_shares = shapley_vals / total_r2 * 100

    result = pd.DataFrame({
        "Variable":       Xcols,
        "Shapley_R2":     shapley_vals,
        "Pct_of_Total":   pct_shares,
    }).sort_values("Shapley_R2", ascending=False).reset_index(drop=True)

    result["Rank_Shapley"] = range(1, n + 1)
    print(f"\nTotal model R² = {total_r2:.4f}")
    print(result.to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("Blues_d", n)
    ax.bar(result["Variable"], result["Pct_of_Total"], color=colors)
    ax.set_ylabel("% of Total Explained Variance")
    ax.set_title("Shapley R² Decomposition\n(% share of total model R²)")
    ax.set_ylim(0, 100)
    for i, (v, p) in enumerate(zip(result["Variable"], result["Pct_of_Total"])):
        ax.text(i, p + 1, f"{p:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/1_shapley_r2.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/1_shapley_r2.png")

    return result


# ──────────────────────────────────────────────────────────────
# 3. METHOD 2 — VAR + FORECAST ERROR VARIANCE DECOMPOSITION
# ──────────────────────────────────────────────────────────────
# FEVD asks: at a given horizon h, what fraction of the forecast
# uncertainty in core goods inflation is attributable to each
# variable's own shocks? A large share at long horizons implies
# structural/persistent influence on inflation dynamics.

def var_fevd(df):
    print("\n══════════════════════════════════════")
    print("METHOD 2 — VAR + FEVD")
    print("══════════════════════════════════════")

    var_cols = ["core_goods_yoy", "energy_yoy", "inf_exp", "wages_yoy"]
    data     = df[var_cols].copy()

    model    = VAR(data)
    lag_sel  = model.select_order(maxlags=12)
    best_lag = int(lag_sel.aic)
    best_lag = max(best_lag, 1)
    print(f"AIC-selected lag order: {best_lag}")

    fitted   = model.fit(best_lag)
    print(fitted.summary())

    fevd_obj = fitted.fevd(24)   # 24-month horizon
    fevd_arr = fevd_obj.decomp   # shape: (n_vars, horizon, n_vars)
    # fevd_arr[i, h, j] = share of var i's FEV at horizon h+1 due to var j

    core_idx   = 0               # core_goods_yoy is first column
    horizons   = [1, 3, 6, 12, 24]
    fevd_table = pd.DataFrame(index=horizons, columns=var_cols)

    for h in horizons:
        row = fevd_arr[core_idx, h - 1, :]        # contributions to core goods
        fevd_table.loc[h] = np.round(row * 100, 2)

    fevd_table.index.name = "Horizon (months)"
    print("\nFEVD for Core Goods Inflation (% of forecast error variance):")
    print(fevd_table.to_string())

    # Rank by contribution at 12-month horizon
    fevd_12     = fevd_table.loc[12].astype(float)
    predictors  = ["energy_yoy", "inf_exp", "wages_yoy"]
    fevd_pred   = fevd_12[predictors].sort_values(ascending=False)
    ranks_fevd  = {v: i + 1 for i, v in enumerate(fevd_pred.index)}

    # ── Plot FEVD over 24 months ──
    fig, ax = plt.subplots(figsize=(9, 5))
    h_range  = np.arange(1, 25)
    palette  = sns.color_palette("tab10", len(var_cols))
    bottom   = np.zeros(24)

    for j, col in enumerate(var_cols):
        vals = fevd_arr[core_idx, :, j] * 100
        ax.fill_between(h_range, bottom, bottom + vals,
                        alpha=0.8, color=palette[j], label=col)
        bottom += vals

    ax.set_xlabel("Horizon (months ahead)")
    ax.set_ylabel("% of Forecast Error Variance")
    ax.set_title("FEVD of Core Goods Inflation\n(share of forecast uncertainty by source)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(1, 24)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/2_var_fevd.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/2_var_fevd.png")

    return fevd_table, ranks_fevd


# ──────────────────────────────────────────────────────────────
# 4. METHOD 3 — XGBOOST + SHAP
# ──────────────────────────────────────────────────────────────
# XGBoost captures non-linear interactions that OLS misses.
# SHAP (SHapley Additive exPlanations) decomposes each prediction
# into per-feature contributions — consistent with the Shapley
# axioms. Mean |SHAP| measures average impact magnitude.

def xgboost_shap(df):
    print("\n══════════════════════════════════════")
    print("METHOD 3 — XGBOOST + SHAP")
    print("══════════════════════════════════════")

    feature_cols = ["energy_yoy", "inf_exp", "wages_yoy"]
    X = df[feature_cols].values
    y = df["core_goods_yoy"].values

    # TimeSeriesSplit CV to avoid lookahead bias in time-series data
    tscv    = TimeSeriesSplit(n_splits=5)
    cv_r2s  = []
    best_params = dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model_cv = xgb.XGBRegressor(**best_params, verbosity=0)
        model_cv.fit(X[train_idx], y[train_idx])
        preds    = model_cv.predict(X[test_idx])
        r2       = r2_score(y[test_idx], preds)
        cv_r2s.append(r2)
        print(f"  Fold {fold+1}: test R² = {r2:.4f}")

    print(f"  Mean CV R²: {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")

    # Final model on full data
    model_full = xgb.XGBRegressor(**best_params, verbosity=0)
    model_full.fit(X, y)

    # SHAP values — use XGBoost's native pred_contribs instead of the shap
    # library's TreeExplainer, which breaks on XGBoost 3.x / SHAP 0.43.
    # pred_contribs returns (n_obs, n_features + 1); last col is the bias term.
    dmat      = xgb.DMatrix(X, feature_names=feature_cols)
    shap_vals = model_full.get_booster().predict(dmat, pred_contribs=True)[:, :-1]
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    ranks_shap = {feature_cols[i]: int(np.argsort(-mean_abs)[list(np.argsort(-mean_abs)).index(i)] + 1)
                  for i in range(len(feature_cols))}
    # simpler rank dict
    order      = np.argsort(-mean_abs)
    ranks_shap = {feature_cols[order[i]]: i + 1 for i in range(len(feature_cols))}

    print("\nMean |SHAP| by feature:")
    for i, col in enumerate(feature_cols):
        print(f"  {col:20s}: {mean_abs[i]:.4f}")

    # ── SHAP Summary Plot (beeswarm) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_vals, X,
                      feature_names=feature_cols,
                      plot_type="dot",
                      show=False)
    plt.title("SHAP Summary Plot — Core Goods Inflation Drivers", pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3a_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/3a_shap_summary.png")

    # ── SHAP Bar Chart ──
    fig, ax = plt.subplots(figsize=(7, 4))
    shap.summary_plot(shap_vals, X,
                      feature_names=feature_cols,
                      plot_type="bar",
                      show=False)
    plt.title("Mean |SHAP| — Average Feature Impact on Core Goods Inflation", pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3b_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/3b_shap_bar.png")

    shap_df = pd.DataFrame({
        "Variable":    feature_cols,
        "Mean_AbsSHAP": mean_abs,
        "Rank_SHAP":   [ranks_shap[c] for c in feature_cols],
    }).sort_values("Mean_AbsSHAP", ascending=False).reset_index(drop=True)

    return shap_df, ranks_shap


# ──────────────────────────────────────────────────────────────
# 5. METHOD 4 — MARKOV-SWITCHING REGRESSION
# ──────────────────────────────────────────────────────────────
# Inflation dynamics are unlikely to be constant across time.
# A 2-regime Markov-switching model captures structural breaks
# (e.g., pre- vs post-pandemic inflation regimes) by allowing
# intercepts AND slope coefficients to differ by regime.
# Regime probabilities are inferred from the data.

def markov_switching(df):
    print("\n══════════════════════════════════════")
    print("METHOD 4 — MARKOV-SWITCHING REGRESSION")
    print("══════════════════════════════════════")

    y    = df["core_goods_yoy"].values
    Xcols = ["energy_yoy", "inf_exp", "wages_yoy"]
    exog = df[Xcols].values

    # Fit 2-regime Markov-switching regression with switching coefficients.
    # Try full switching model first; fall back to intercept-only switching if
    # numerical convergence fails (can happen with near-stationary regimes).
    def try_fit(switching_exog, switching_variance, reps):
        mod = MarkovRegression(
            endog=y, k_regimes=2, exog=exog,
            switching_exog=switching_exog,
            switching_variance=switching_variance,
        )
        return mod.fit(search_reps=reps, disp=False)

    ms_fit = None
    for sw_exog, sw_var, reps in [
        (True,  True,  20),   # full model
        (True,  False, 20),   # common variance
        (False, True,  20),   # switching intercept + variance only
        (False, False, 10),   # simplest fallback
    ]:
        try:
            ms_fit = try_fit(sw_exog, sw_var, reps)
            spec = f"switching_exog={sw_exog}, switching_variance={sw_var}"
            print(f"  Markov model converged with: {spec}")
            break
        except Exception as e:
            print(f"  Fallback needed ({e.__class__.__name__}): trying simpler spec...")

    if ms_fit is None:
        raise RuntimeError("Markov-switching model failed to converge under all specifications.")
    print(ms_fit.summary())

    # Smoothed regime probabilities (P(regime=k | full data))
    # Normalise to a DataFrame regardless of whether statsmodels returns
    # a DataFrame or a raw ndarray (depends on model spec used).
    raw_probs = ms_fit.smoothed_marginal_probabilities
    if isinstance(raw_probs, pd.DataFrame):
        smoothed_probs = raw_probs.reset_index(drop=True)
        smoothed_probs.index = df.index
    else:
        # ndarray shape (n_obs, k_regimes) or (k_regimes, n_obs)
        arr = np.array(raw_probs)
        if arr.shape[0] != len(df):
            arr = arr.T
        smoothed_probs = pd.DataFrame(arr, index=df.index)

    # ── Regime probability plot ──
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    dates = df.index

    axes[0].plot(dates, y, color="navy", lw=1, label="Core Goods Inflation (YoY %)")
    axes[0].axhline(0, color="grey", lw=0.5, ls="--")
    axes[0].set_ylabel("YoY %")
    axes[0].set_title("Core Goods Inflation & Markov-Switching Regimes")
    axes[0].legend(fontsize=8)

    axes[1].fill_between(dates, smoothed_probs.iloc[:, 0],
                         alpha=0.6, color="tomato", label="Regime 0 (prob)")
    axes[1].fill_between(dates, smoothed_probs.iloc[:, 1],
                         alpha=0.6, color="steelblue", label="Regime 1 (prob)")
    axes[1].set_ylabel("Smoothed Probability")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/4_markov_regime_probs.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/4_markov_regime_probs.png")

    # Extract regime-specific coefficients.
    # param_names lives on the model object; exog vars are labelled x1, x2, x3
    # in statsmodels (positional, not by column name), so we map by index.
    params      = ms_fit.params
    param_names = ms_fit.model.param_names   # e.g. ['p[0->0]','p[1->0]','const[0]',...]

    coef_table = pd.DataFrame({
        "Parameter": param_names,
        "Value":     np.round(params, 4),
    })
    # Annotate with human-readable variable names where possible
    xi_map = {f"x{k+1}": col for k, col in enumerate(Xcols)}
    coef_table["Label"] = coef_table["Parameter"].apply(
        lambda p: next((f"{xi_map[xi]}{p[len(xi):]}" for xi in xi_map if p.startswith(xi)), p)
    )
    print("\nCoefficients per regime:")
    print(coef_table[["Label", "Value"]].to_string(index=False))

    # % of sample in each regime (based on argmax of smoothed probs)
    dominant_regime = smoothed_probs.values.argmax(axis=1)
    regime_shares   = pd.Series(dominant_regime).value_counts(normalize=True) * 100
    print("\n% of sample in each regime (dominant-regime classification):")
    for r, share in regime_shares.sort_index().items():
        print(f"  Regime {r}: {share:.1f}%")

    # Rank predictors by average absolute coefficient across regimes.
    # Match by positional label (x1→col0, x2→col1, x3→col2) to handle both
    # switching_exog=True (xk[0], xk[1]) and switching_exog=False (xk).
    var_avg_coef = {}
    for k, col in enumerate(Xcols):
        xi    = f"x{k+1}"
        idxs  = [i for i, p in enumerate(param_names) if p.startswith(xi)]
        var_avg_coef[col] = np.mean([abs(params[i]) for i in idxs]) if idxs else 0.0

    order_ms   = sorted(var_avg_coef, key=var_avg_coef.get, reverse=True)
    ranks_ms   = {v: i + 1 for i, v in enumerate(order_ms)}
    print("\nPredictors ranked by avg |coeff| across regimes:")
    for col in order_ms:
        print(f"  {col:20s}: avg |β| = {var_avg_coef[col]:.4f}  Rank {ranks_ms[col]}")

    return coef_table, regime_shares, ranks_ms


# ──────────────────────────────────────────────────────────────
# 6. SUMMARY COMPARISON TABLE
# ──────────────────────────────────────────────────────────────

def print_summary(shapley_df, ranks_fevd, shap_df, ranks_ms):
    print("\n══════════════════════════════════════")
    print("FINAL SUMMARY — VARIABLE IMPORTANCE RANKINGS ACROSS METHODS")
    print("══════════════════════════════════════")
    print("(1 = most important driver of core goods inflation variability)\n")

    vars_ = ["energy_yoy", "inf_exp", "wages_yoy"]
    label = {"energy_yoy": "Energy CPI (YoY)",
             "inf_exp":    "Inflation Expectations",
             "wages_yoy":  "Wage Growth (YoY)"}

    shapley_rank = dict(zip(shapley_df["Variable"], shapley_df["Rank_Shapley"]))
    shap_rank    = dict(zip(shap_df["Variable"],    shap_df["Rank_SHAP"]))

    rows = []
    for v in vars_:
        r1 = shapley_rank.get(v, "-")
        r2 = ranks_fevd.get(v, "-")
        r3 = shap_rank.get(v, "-")
        r4 = ranks_ms.get(v, "-")
        try:
            avg = np.mean([r1, r2, r3, r4])
        except Exception:
            avg = np.nan
        rows.append([label[v], r1, r2, r3, r4, f"{avg:.1f}"])

    summary = pd.DataFrame(rows, columns=[
        "Variable",
        "Rank: Shapley R²",
        "Rank: VAR FEVD (12m)",
        "Rank: XGBoost SHAP",
        "Rank: Markov-Switching",
        "Avg Rank",
    ])
    summary = summary.sort_values("Avg Rank").reset_index(drop=True)
    print(summary.to_string(index=False))

    # ── Summary heatmap ──
    fig, ax = plt.subplots(figsize=(9, 3))
    rank_data = summary[["Rank: Shapley R²",
                          "Rank: VAR FEVD (12m)",
                          "Rank: XGBoost SHAP",
                          "Rank: Markov-Switching"]].astype(float)
    sns.heatmap(
        rank_data,
        annot=True, fmt=".0f",
        yticklabels=summary["Variable"],
        cmap="YlOrRd_r",          # green=best rank (1), red=worst (3)
        linewidths=0.5,
        ax=ax, cbar_kws={"label": "Rank (1=top driver)"},
        vmin=1, vmax=3,
    )
    ax.set_title("Variable Importance Rankings Across Methods\n(1 = primary driver of core goods inflation)")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/5_summary_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR}/5_summary_heatmap.png")

    return summary


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("INFLATION DRIVERS ANALYSIS — CORE GOODS CPI")
    print("=" * 55)

    # 1. Fetch & prepare data
    df_raw        = fetch_data()
    df_stat, _    = prepare_stationary(df_raw)

    # 2. Run all four methods
    shapley_df            = shapley_r2(df_stat)
    fevd_table, ranks_fevd = var_fevd(df_stat)
    shap_df, ranks_shap   = xgboost_shap(df_stat)
    _, regime_shares, ranks_ms = markov_switching(df_stat)

    # 3. Print consolidated summary
    summary = print_summary(shapley_df, ranks_fevd, shap_df, ranks_ms)

    print("\n── Economic interpretation summary ──")
    top_var = summary.iloc[0]["Variable"]
    print(f"""
  • The variable ranked #1 most consistently across methods is: {top_var}
  • Energy CPI typically exerts large, direct pass-through to goods prices
    through production and transport cost channels.
  • Inflation expectations capture the forward-looking, self-fulfilling
    component of wage-price spirals — important at medium horizons (FEVD).
  • Wage growth reflects labor cost push, often dominant in service-heavy
    periods but present in goods too via manufacturing labor costs.
  • The Markov-switching model reveals that these relationships change across
    inflation regimes — coefficients in high-inflation episodes (e.g. 2021-23)
    often differ materially from quiescent periods (e.g. 2010-19).
""")

    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
