"""
Freight Rate Prediction Challenge — Production Training & Inference Pipeline
=============================================================================
This script executes the complete machine learning workflow:
1. Data loading & exploratory verification
2. Preprocessing, missing value imputation, sign-error correction, and geo-coordinate resolution
3. Feature engineering (cyclical time, haversine distances, market interactions, domain features)
4. Validation strategy:
   - Temporal Hold-Out Split: Train on Jan-Sep 2025, Evaluate on Oct 2025
   - 5-Fold Cross-Validation across the development dataset
5. Model Architecture:
   - Log-transformed target modeling (log) for freight rate distribution
   - Ensemble of LightGBM Regressor and XGBoost Regressor
6. Inference:
   - Generation of final 12,000 predictions for validation.csv -> validation_predictions.csv
   - Generation of 31 daily predictions for december_chart_inputs.csv
7. Verification with score.py

NOTE ON FILE PATHS: this script expects the assessment's `data/` folder layout
(data/train_test.csv, data/validation.csv, data/december_chart_inputs.csv,
data/validation_predictions_template.csv) sitting next to this script, matching
README.md. If your local copies use different names/locations, update the
constants below.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train_test.csv")
VAL_FILE = os.path.join(DATA_DIR, "validation.csv")
DEC_FILE = os.path.join(DATA_DIR, "december_chart_inputs.csv")
VAL_TEMPLATE_FILE = os.path.join(DATA_DIR, "validation_predictions_template.csv")
VAL_PRED_OUT = "validation_predictions.csv"
RANDOM_STATE = 42

# Known coordinate dictionary derived from the training dataset for consistent spatial mapping
# Lexington, KY and Fort Wayne, IN coordinates from train-test.csv
LEXINGTON_COORDS = (36.99152, -84.99876)
FORT_WAYNE_COORDS = (41.31561, -85.36206)

EQUIPMENT_MAP = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculates great circle distance in miles between coordinate arrays."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def engineer_features(
    df: pd.DataFrame,
    city_lat_map: dict[str, float] | None = None,
    city_lon_map: dict[str, float] | None = None,
    dec_daily_signals: pd.DataFrame | None = None,
    is_dec: bool = False,
    median_weight: float = 30000.0,
    median_market_index: float = 1.05,
) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    # If processing December chart inputs, resolve coordinates and market signals
    if is_dec:
        data["pickup_lat"] = LEXINGTON_COORDS[0]
        data["pickup_lon"] = LEXINGTON_COORDS[1]
        data["delivery_lat"] = FORT_WAYNE_COORDS[0]
        data["delivery_lon"] = FORT_WAYNE_COORDS[1]
        if dec_daily_signals is not None:
            data = data.merge(dec_daily_signals, on="date", how="left")

    # If coordinates are missing for any reason, map from known city medians
    if city_lat_map and city_lon_map:
        if "pickup_lat" in data.columns:
            data["pickup_lat"] = data["pickup_lat"].fillna(data["pickup"].map(city_lat_map))
            data["pickup_lon"] = data["pickup_lon"].fillna(data["pickup"].map(city_lon_map))
            data["delivery_lat"] = data["delivery_lat"].fillna(data["delivery"].map(city_lat_map))
            data["delivery_lon"] = data["delivery_lon"].fillna(data["delivery"].map(city_lon_map))

    # Data-quality fix: ~0.6% of rows have a negative weight (sign-flip data-entry
    # errors — their absolute values match the legitimate weight distribution almost
    # exactly, min/max/mean all line up). Correct the sign before imputing missing values.
    data["weight"] = data["weight"].abs()

    # Missing value imputation using robust domain medians
    data["weight"] = data["weight"].fillna(median_weight)
    if "market_index" in data.columns:
        data["market_index"] = data["market_index"].fillna(median_market_index)
    else:
        data["market_index"] = median_market_index

    if "quote_signal" not in data.columns:
        data["quote_signal"] = 2.05

    # Equipment encoding
    data["equipment_code"] = data["equipment"].map(EQUIPMENT_MAP).fillna(0).astype(int)

    # Spatial features
    lat1, lon1 = data["pickup_lat"].values, data["pickup_lon"].values
    lat2, lon2 = data["delivery_lat"].values, data["delivery_lon"].values
    data["haversine_dist"] = haversine_np(lat1, lon1, lat2, lon2)
    data["lat_diff"] = lat2 - lat1
    data["lon_diff"] = lon2 - lon1
    data["mid_lat"] = (lat1 + lat2) / 2.0
    data["mid_lon"] = (lon1 + lon2) / 2.0
    data["circuity"] = data["distance"] / (data["haversine_dist"] + 1.0)

    # Temporal features
    dt = data["date"].dt
    data["day_of_week"] = dt.dayofweek
    data["day_of_month"] = dt.day
    data["month"] = dt.month
    data["week_of_year"] = dt.isocalendar().week.astype(int)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["sin_dow"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["cos_dow"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
    data["sin_month"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)

    # Domain interaction features
    data["dist_x_qs"] = data["distance"] * data["quote_signal"]
    data["dist_x_mkt"] = data["distance"] * data["market_index"]
    data["dist_x_weight"] = data["distance"] * (data["weight"] / 10000.0)
    data["weight_per_mile"] = data["weight"] / (data["distance"] + 1.0)

    return data


FEATURE_COLS = [
    "distance",
    "haversine_dist",
    "circuity",
    "lat_diff",
    "lon_diff",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "mid_lat",
    "mid_lon",
    "weight",
    "equipment_code",
    "market_index",
    "quote_signal",
    "dist_x_qs",
    "dist_x_mkt",
    "dist_x_weight",
    "weight_per_mile",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
]


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def main():
    print("=" * 70)
    print("   FREIGHT RATE PREDICTION — PRODUCTION PIPELINE")
    print("=" * 70)

    print("\n[Step 1/7] Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE, parse_dates=["date"])
    val_df = pd.read_csv(VAL_FILE, parse_dates=["date"])
    dec_df = pd.read_csv(DEC_FILE, parse_dates=["date"])

    print(f"  Training samples:   {len(train_df):,} rows, {train_df.shape[1]} columns")
    print(f"  Validation samples: {len(val_df):,} rows, {val_df.shape[1]} columns")
    print(f"  December inputs:    {len(dec_df):,} rows, {dec_df.shape[1]} columns")

    # City coordinate lookup dictionaries
    city_lat = pd.concat([train_df.groupby("pickup")["pickup_lat"].median(),
                          train_df.groupby("delivery")["delivery_lat"].median()]).to_dict()
    city_lon = pd.concat([train_df.groupby("pickup")["pickup_lon"].median(),
                          train_df.groupby("delivery")["delivery_lon"].median()]).to_dict()

    # Median market values for imputation (weight median computed on corrected
    # absolute values so the ~0.6% sign-flipped rows don't need special-casing)
    med_weight = float(train_df["weight"].abs().median())
    med_mkt = float(train_df["market_index"].median())

    # Daily market signals for December extracted from validation data
    dec_daily_signals = (
        val_df[val_df["date"].dt.month == 12]
        .groupby("date")[["market_index", "quote_signal"]]
        .mean()
        .reset_index()
    )

    print("\n[Step 2/7] Feature engineering across all sets...")
    train_feat = engineer_features(train_df, city_lat, city_lon, median_weight=med_weight, median_market_index=med_mkt)
    val_feat = engineer_features(val_df, city_lat, city_lon, median_weight=med_weight, median_market_index=med_mkt)
    dec_feat = engineer_features(
        dec_df,
        city_lat,
        city_lon,
        dec_daily_signals=dec_daily_signals,
        is_dec=True,
        median_weight=med_weight,
        median_market_index=med_mkt,
    )

    X_full = train_feat[FEATURE_COLS]
    y_full = train_feat["posted_rate"].values
    log_y_full = np.log(y_full)

    print(f"  Engineered {len(FEATURE_COLS)} features: {FEATURE_COLS[:8]}... (and 20 more)")

    # ──────────────────────────────────────────────────────────────────────────
    # VALIDATION 1: TEMPORAL SPLIT (Jan-Sep vs Oct 2025)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 3/7] Temporal Validation (Jan-Sep 2025 Train -> Oct 2025 Test)...")
    split_date = pd.Timestamp("2025-10-01")
    tr_mask = train_feat["date"] < split_date
    oos_mask = train_feat["date"] >= split_date

    X_tr, log_y_tr, y_tr = train_feat.loc[tr_mask, FEATURE_COLS], log_y_full[tr_mask], y_full[tr_mask]
    X_oos, log_y_oos, y_oos = train_feat.loc[oos_mask, FEATURE_COLS], log_y_full[oos_mask], y_full[oos_mask]
    print(f"  Training split (Jan-Sep): {len(X_tr):,} loads")
    print(f"  Hold-out split (Oct):     {len(X_oos):,} loads")

    lgb_holdout = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_holdout.fit(X_tr, log_y_tr)

    xgb_holdout = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_holdout.fit(X_tr, log_y_tr)

    pred_lgb_oos = np.exp(lgb_holdout.predict(X_oos))
    pred_xgb_oos = np.exp(xgb_holdout.predict(X_oos))
    pred_ens_oos = 0.5 * pred_lgb_oos + 0.5 * pred_xgb_oos

    m_lgb = evaluate_metrics(y_oos, pred_lgb_oos)
    m_xgb = evaluate_metrics(y_oos, pred_xgb_oos)
    m_ens = evaluate_metrics(y_oos, pred_ens_oos)

    print("\n  Temporal Hold-out Results:")
    print(f"  - LightGBM : RMSE=${m_lgb['RMSE']:.2f} | MAE=${m_lgb['MAE']:.2f} | R2={m_lgb['R2']:.4f} | MAPE={m_lgb['MAPE']:.2f}%")
    print(f"  - XGBoost  : RMSE=${m_xgb['RMSE']:.2f} | MAE=${m_xgb['MAE']:.2f} | R2={m_xgb['R2']:.4f} | MAPE={m_xgb['MAPE']:.2f}%")
    print(f"  - Ensemble : RMSE=${m_ens['RMSE']:.2f} | MAE=${m_ens['MAE']:.2f} | R2={m_ens['R2']:.4f} | MAPE={m_ens['MAPE']:.2f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # VALIDATION 2: 5-FOLD CROSS-VALIDATION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 4/7] 5-Fold Cross-Validation on Development Dataset...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rmses, cv_maes, cv_r2s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full), 1):
        fold_X_tr, fold_y_tr = X_full.iloc[train_idx], log_y_full[train_idx]
        fold_X_va, fold_y_va_raw = X_full.iloc[val_idx], y_full[val_idx]

        m_l = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.035, num_leaves=63, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        m_x = xgb.XGBRegressor(n_estimators=700, learning_rate=0.035, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

        m_l.fit(fold_X_tr, fold_y_tr)
        m_x.fit(fold_X_tr, fold_y_tr)

        fold_preds = 0.5 * np.exp(m_l.predict(fold_X_va)) + 0.5 * np.exp(m_x.predict(fold_X_va))
        metrics = evaluate_metrics(fold_y_va_raw, fold_preds)
        cv_rmses.append(metrics["RMSE"])
        cv_maes.append(metrics["MAE"])
        cv_r2s.append(metrics["R2"])
        print(f"  Fold {fold}: RMSE=${metrics['RMSE']:.2f}, MAE=${metrics['MAE']:.2f}, R2={metrics['R2']:.4f}")

    print(f"  Mean 5-Fold CV: RMSE=${np.mean(cv_rmses):.2f} +/- ${np.std(cv_rmses):.2f} | MAE=${np.mean(cv_maes):.2f} | R2={np.mean(cv_r2s):.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # PRODUCTION RETRAINING ON FULL 48,000 ROWS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 5/7] Retraining Final Production Ensemble on full 48,000 loads...")
    final_lgb = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    final_xgb = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    final_lgb.fit(X_full, log_y_full)
    final_xgb.fit(X_full, log_y_full)
    print("  Models successfully fitted.")

    # ──────────────────────────────────────────────────────────────────────────
    # PREDICTIONS: VALIDATION.CSV
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 6/7] Generating Predictions for validation.csv...")
    X_val = val_feat[FEATURE_COLS]
    val_pred_lgb = np.exp(final_lgb.predict(X_val))
    val_pred_xgb = np.exp(final_xgb.predict(X_val))
    val_pred_final = 0.5 * val_pred_lgb + 0.5 * val_pred_xgb
    val_pred_final = np.clip(val_pred_final, a_min=10.0, a_max=None)

    preds_by_id = pd.Series(np.round(val_pred_final, 2), index=val_df["load_id"].values)

    # Build the submission from the official template when it's available, so the
    # row order/ID set exactly matches what score.py expects, instead of trusting
    # that val_df's row order lines up with the template.
    if os.path.isfile(VAL_TEMPLATE_FILE):
        template_df = pd.read_csv(VAL_TEMPLATE_FILE)
        submission_df = template_df[["load_id"]].copy()
        submission_df["predicted_rate"] = submission_df["load_id"].map(preds_by_id)
        if submission_df["predicted_rate"].isna().any():
            missing = submission_df.loc[submission_df["predicted_rate"].isna(), "load_id"].tolist()
            raise SystemExit(f"Missing predictions for {len(missing)} template load_id(s), e.g. {missing[:5]}")
    else:
        print(f"  WARNING: template file not found at {VAL_TEMPLATE_FILE}; "
              "building submission directly from validation.csv's load_id order instead.")
        submission_df = pd.DataFrame({
            "load_id": val_df["load_id"],
            "predicted_rate": np.round(val_pred_final, 2),
        })
    submission_df.to_csv(VAL_PRED_OUT, index=False)
    print(f"  Successfully wrote {len(submission_df):,} rows to {VAL_PRED_OUT}")
    print(f"  Validation Rate Stats: Min=${submission_df['predicted_rate'].min():.2f}, "
          f"Mean=${submission_df['predicted_rate'].mean():.2f}, "
          f"Median=${submission_df['predicted_rate'].median():.2f}, "
          f"Max=${submission_df['predicted_rate'].max():.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # PREDICTIONS: DECEMBER CHART INPUTS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 7/7] Generating Predictions for december-chart-inputs.csv...")
    X_dec = dec_feat[FEATURE_COLS]
    dec_pred_lgb = np.exp(final_lgb.predict(X_dec))
    dec_pred_xgb = np.exp(final_xgb.predict(X_dec))
    dec_pred_final = 0.5 * dec_pred_lgb + 0.5 * dec_pred_xgb
    dec_pred_final = np.clip(dec_pred_final, a_min=10.0, a_max=None)

    dec_df["predicted_rate"] = np.round(dec_pred_final, 2)
    # Keep an untouched copy of the original input before overwriting it in place
    # (README step 4 asks us to fill predicted_rate into this same file).
    backup_path = DEC_FILE + ".original"
    if not os.path.isfile(backup_path):
        shutil.copyfile(DEC_FILE, backup_path)
    dec_df.to_csv(DEC_FILE, index=False)
    print(f"  Successfully wrote 31 rows to {DEC_FILE} (original backed up to {backup_path})")
    print(f"  December Rate Stats: Min=${dec_df['predicted_rate'].min():.2f}, "
          f"Mean=${dec_df['predicted_rate'].mean():.2f}, "
          f"Max=${dec_df['predicted_rate'].max():.2f}")

    # Top Feature Importances
    fi_lgb = pd.Series(final_lgb.feature_importances_, index=FEATURE_COLS)
    fi_lgb = (fi_lgb / fi_lgb.sum()).sort_values(ascending=False)
    print("\n  Top 8 LightGBM Predictive Features:")
    for feat, imp in fi_lgb.head(8).items():
        print(f"    - {feat:<20}: {imp*100:.1f}%")

    print("\n" + "=" * 70)
    print("   PIPELINE COMPLETE. Ready to run score.py verification.")
    print("=" * 70)


if __name__ == "__main__":
    main()
