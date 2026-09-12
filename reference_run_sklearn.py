
from __future__ import annotations

import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

TRAIN_FILE = "train-test.csv"
VAL_FILE = "validation.csv"
DEC_FILE = "december-chart-inputs.csv"
VAL_PRED_OUT = "validation_predictions.csv"
RANDOM_STATE = 42

LEXINGTON_COORDS = (36.99152, -84.99876)
FORT_WAYNE_COORDS = (41.31561, -85.36206)
EQUIPMENT_MAP = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


def haversine_np(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def engineer_features(df, city_lat_map=None, city_lon_map=None, dec_daily_signals=None,
                       is_dec=False, median_weight=30000.0, median_market_index=1.05):
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    if is_dec:
        data["pickup_lat"] = LEXINGTON_COORDS[0]
        data["pickup_lon"] = LEXINGTON_COORDS[1]
        data["delivery_lat"] = FORT_WAYNE_COORDS[0]
        data["delivery_lon"] = FORT_WAYNE_COORDS[1]
        if dec_daily_signals is not None:
            data = data.merge(dec_daily_signals, on="date", how="left")

    if city_lat_map and city_lon_map:
        if "pickup_lat" in data.columns:
            data["pickup_lat"] = data["pickup_lat"].fillna(data["pickup"].map(city_lat_map))
            data["pickup_lon"] = data["pickup_lon"].fillna(data["pickup"].map(city_lon_map))
            data["delivery_lat"] = data["delivery_lat"].fillna(data["delivery"].map(city_lat_map))
            data["delivery_lon"] = data["delivery_lon"].fillna(data["delivery"].map(city_lon_map))

    # Data-quality fix: correct sign-flip errors in weight before imputing.
    data["weight"] = data["weight"].abs()
    data["weight"] = data["weight"].fillna(median_weight)
    if "market_index" in data.columns:
        data["market_index"] = data["market_index"].fillna(median_market_index)
    else:
        data["market_index"] = median_market_index
    if "quote_signal" not in data.columns:
        data["quote_signal"] = 2.05

    data["equipment_code"] = data["equipment"].map(EQUIPMENT_MAP).fillna(0).astype(int)

    lat1, lon1 = data["pickup_lat"].values, data["pickup_lon"].values
    lat2, lon2 = data["delivery_lat"].values, data["delivery_lon"].values
    data["haversine_dist"] = haversine_np(lat1, lon1, lat2, lon2)
    data["lat_diff"] = lat2 - lat1
    data["lon_diff"] = lon2 - lon1
    data["mid_lat"] = (lat1 + lat2) / 2.0
    data["mid_lon"] = (lon1 + lon2) / 2.0
    data["circuity"] = data["distance"] / (data["haversine_dist"] + 1.0)

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

    data["dist_x_qs"] = data["distance"] * data["quote_signal"]
    data["dist_x_mkt"] = data["distance"] * data["market_index"]
    data["dist_x_weight"] = data["distance"] * (data["weight"] / 10000.0)
    data["weight_per_mile"] = data["weight"] / (data["distance"] + 1.0)
    return data


FEATURE_COLS = [
    "distance", "haversine_dist", "circuity", "lat_diff", "lon_diff",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "mid_lat", "mid_lon",
    "weight", "equipment_code", "market_index", "quote_signal",
    "dist_x_qs", "dist_x_mkt", "dist_x_weight", "weight_per_mile",
    "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
    "sin_dow", "cos_dow", "sin_month", "cos_month",
]


def evaluate_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def make_ensemble(seed):
    m1 = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=0.1, random_state=seed,
    )
    m2 = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
        l2_regularization=0.5, random_state=seed + 1,
    )
    return m1, m2


def main():
    print("[1/6] Loading data...")
    train_df = pd.read_csv(TRAIN_FILE, parse_dates=["date"])
    val_df = pd.read_csv(VAL_FILE, parse_dates=["date"])
    dec_df = pd.read_csv(DEC_FILE)

    city_lat = pd.concat([train_df.groupby("pickup")["pickup_lat"].median(),
                           train_df.groupby("delivery")["delivery_lat"].median()]).to_dict()
    city_lon = pd.concat([train_df.groupby("pickup")["pickup_lon"].median(),
                           train_df.groupby("delivery")["delivery_lon"].median()]).to_dict()
    med_weight = float(train_df["weight"].abs().median())
    med_mkt = float(train_df["market_index"].median())

    dec_daily_signals = (
        val_df[val_df["date"].dt.month == 12]
        .groupby("date")[["market_index", "quote_signal"]].mean().reset_index()
    )

    print("[2/6] Feature engineering...")
    train_feat = engineer_features(train_df, city_lat, city_lon, median_weight=med_weight, median_market_index=med_mkt)
    val_feat = engineer_features(val_df, city_lat, city_lon, median_weight=med_weight, median_market_index=med_mkt)
    dec_feat = engineer_features(dec_df, city_lat, city_lon, dec_daily_signals=dec_daily_signals,
                                  is_dec=True, median_weight=med_weight, median_market_index=med_mkt)

    X_full = train_feat[FEATURE_COLS]
    y_full = train_feat["posted_rate"].values
    log_y_full = np.log(y_full)

    print("[3/6] Temporal hold-out validation (Jan-Sep train -> Oct test)...")
    split_date = pd.Timestamp("2025-10-01")
    tr_mask = train_feat["date"] < split_date
    oos_mask = train_feat["date"] >= split_date
    X_tr, log_y_tr = train_feat.loc[tr_mask, FEATURE_COLS], log_y_full[tr_mask]
    X_oos, y_oos = train_feat.loc[oos_mask, FEATURE_COLS], y_full[oos_mask]
    print(f"  Train (Jan-Sep): {len(X_tr):,} | Hold-out (Oct): {len(X_oos):,}")

    m1, m2 = make_ensemble(RANDOM_STATE)
    m1.fit(X_tr, log_y_tr)
    m2.fit(X_tr, log_y_tr)
    pred_oos = 0.5 * np.exp(m1.predict(X_oos)) + 0.5 * np.exp(m2.predict(X_oos))
    holdout_metrics = evaluate_metrics(y_oos, pred_oos)
    print("  Hold-out:", {k: round(v, 3) for k, v in holdout_metrics.items()})

    print("[4/6] 5-fold CV on full development set...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rmses, cv_maes, cv_r2s, cv_mapes = [], [], [], []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
        fm1, fm2 = make_ensemble(RANDOM_STATE + fold)
        fm1.fit(X_full.iloc[tr_idx], log_y_full[tr_idx])
        fm2.fit(X_full.iloc[tr_idx], log_y_full[tr_idx])
        fold_pred = 0.5 * np.exp(fm1.predict(X_full.iloc[va_idx])) + 0.5 * np.exp(fm2.predict(X_full.iloc[va_idx]))
        m = evaluate_metrics(y_full[va_idx], fold_pred)
        cv_rmses.append(m["RMSE"]); cv_maes.append(m["MAE"]); cv_r2s.append(m["R2"]); cv_mapes.append(m["MAPE"])
        print(f"  Fold {fold}: RMSE=${m['RMSE']:.2f} MAE=${m['MAE']:.2f} R2={m['R2']:.4f} MAPE={m['MAPE']:.2f}%")
    cv_summary = {
        "RMSE_mean": np.mean(cv_rmses), "RMSE_std": np.std(cv_rmses),
        "MAE_mean": np.mean(cv_maes), "R2_mean": np.mean(cv_r2s), "MAPE_mean": np.mean(cv_mapes),
    }

    print("[5/6] Retraining production ensemble on full 48,000 rows...")
    final_m1, final_m2 = make_ensemble(RANDOM_STATE)
    final_m1.fit(X_full, log_y_full)
    final_m2.fit(X_full, log_y_full)

    X_val = val_feat[FEATURE_COLS]
    val_pred = 0.5 * np.exp(final_m1.predict(X_val)) + 0.5 * np.exp(final_m2.predict(X_val))
    val_pred = np.clip(val_pred, 10.0, None)
    submission_df = pd.DataFrame({"load_id": val_df["load_id"], "predicted_rate": np.round(val_pred, 2)})
    submission_df.to_csv(VAL_PRED_OUT, index=False)
    print(f"  Wrote {len(submission_df):,} rows -> {VAL_PRED_OUT}")

    print("[6/6] December fixed-lane predictions...")
    X_dec = dec_feat[FEATURE_COLS]
    dec_pred = 0.5 * np.exp(final_m1.predict(X_dec)) + 0.5 * np.exp(final_m2.predict(X_dec))
    dec_pred = np.clip(dec_pred, 10.0, None)
    shutil.copyfile(DEC_FILE, DEC_FILE + ".original")
    dec_df["predicted_rate"] = np.round(dec_pred, 2)
    dec_df.to_csv(DEC_FILE, index=False)
    print(f"  Wrote 31 rows -> {DEC_FILE}")
    print(f"  Dec rate range: ${dec_df['predicted_rate'].min():.2f} - ${dec_df['predicted_rate'].max():.2f}")

    fi = pd.Series(final_m1.feature_importances_ if hasattr(final_m1, "feature_importances_") else [], index=[])
    # HGB doesn't expose feature_importances_ directly; use permutation-free proxy via built-in
    import json
    with open("run_metrics.json", "w") as f:
        json.dump({"holdout": holdout_metrics, "cv": cv_summary}, f, indent=2)
    print("\nDone. Metrics saved to run_metrics.json")


if __name__ == "__main__":
    main()
