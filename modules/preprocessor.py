import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── I-80 constants ────────────────────────────────────────────────────────────
I80_VALID_LANES = {1, 2, 3, 4, 5, 6}   # exclude HOV (lane 7) and ramp misc
ONRAMP_LANE_ID = 6                      # rightmost ramp lane in I-80 dataset
MAINLINE_LANES = {1, 2, 3, 4, 5}
FEET_TO_METRES = 0.3048
FPS_TO_MPS = 0.3048
FRAME_RATE_HZ = 10                    # NGSIM records at 10 Hz
SEGMENT_LENGTH_FT = 1650                  # ~500 m in feet

# Required columns and their expected dtypes after cleaning
REQUIRED_COLS = {
    "Vehicle_ID":       "int64",
    "Frame_ID":         "int64",
    "Total_Frames":     "int64",
    "Global_Time":      "int64",
    "Local_X":          "float64",
    "Local_Y":          "float64",
    "Global_X":         "float64",
    "Global_Y":         "float64",
    "v_Length":         "float64",
    "v_Width":          "float64",
    "v_Class":          "int64",
    "v_Vel":            "float64",
    "v_Acc":            "float64",
    "Lane_ID":          "int64",
    "Preceding":        "int64",
    "Following":        "int64",
    "Space_Headway":    "float64",
    "Time_Headway":     "float64",
}

# Columns that can be coerced; if still NaN after coercion they are dropped
NULLABLE_AFTER_COERCE = {"Space_Headway",
                         "Time_Headway", "Preceding", "Following"}


def load_raw(filepath: str) -> pd.DataFrame:
    """Load raw NGSIM CSV with flexible parsing."""
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(
        filepath,
        sep=r"\s+|,",          # handles both whitespace and comma delimited
        engine="python",
        header=0,
        on_bad_lines="warn",
    )
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    logger.info(f"Raw shape: {df.shape}")
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric dtypes; non-parseable values become NaN."""
    for col, dtype in REQUIRED_COLS.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where mandatory columns are NaN."""
    mandatory = [c for c in REQUIRED_COLS if c not in NULLABLE_AFTER_COERCE]
    before = len(df)
    df = df.dropna(subset=mandatory)
    logger.info(
        f"Dropped {before - len(df)} rows with missing mandatory fields")
    # Fill nullable cols with sensible defaults
    df["Preceding"] = df["Preceding"].fillna(0).astype("int64")
    df["Following"] = df["Following"].fillna(0).astype("int64")
    df["Space_Headway"] = df["Space_Headway"].fillna(-1.0)
    df["Time_Headway"] = df["Time_Headway"].fillna(-1.0)
    return df


def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in REQUIRED_COLS.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["Vehicle_ID", "Frame_ID"], keep="first")
    logger.info(f"Removed {before - len(df)} duplicate rows")
    return df


def _filter_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within the ~500 m I-80 segment."""
    before = len(df)
    df = df[(df["Local_Y"] >= 0) & (df["Local_Y"] <= SEGMENT_LENGTH_FT)]
    df = df[df["Local_X"] > 0]
    logger.info(f"Spatial filter removed {before - len(df)} rows")
    return df


def _filter_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude HOV lane and invalid lane IDs."""
    before = len(df)
    df = df[df["Lane_ID"].isin(I80_VALID_LANES)]
    logger.info(f"Lane filter removed {before - len(df)} rows (HOV / invalid)")
    return df


def _fix_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each vehicle, frames must be monotonically increasing.
    Drop rows that break ordering (caused by sensor glitches).
    """
    df = df.sort_values(["Vehicle_ID", "Frame_ID"])
    mask = df.groupby("Vehicle_ID")["Frame_ID"].transform(
        lambda s: s == s.cummax()
    )
    before = len(df)
    df = df[mask]
    logger.info(f"Monotonicity fix removed {before - len(df)} rows")
    return df


def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert feet/fps to metres/m/s and add derived columns."""
    df["x_m"] = df["Local_X"] * FEET_TO_METRES
    df["y_m"] = df["Local_Y"] * FEET_TO_METRES
    df["speed_ms"] = df["v_Vel"] * FPS_TO_MPS
    df["accel_ms2"] = df["v_Acc"] * FPS_TO_MPS
    df["length_m"] = df["v_Length"] * FEET_TO_METRES
    df["width_m"] = df["v_Width"] * FEET_TO_METRES
    # timestamp in seconds (Global_Time is in milliseconds)
    df["time_s"] = df["Global_Time"] / 1000.0
    return df


def _add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-vehicle lag features useful for scenario detection."""
    df = df.sort_values(["Vehicle_ID", "Frame_ID"])
    grp = df.groupby("Vehicle_ID")

    df["prev_lane"] = grp["Lane_ID"].shift(1)
    df["prev_speed"] = grp["speed_ms"].shift(1)
    df["prev_x"] = grp["x_m"].shift(1)
    df["prev_y"] = grp["y_m"].shift(1)
    # lateral movement per frame
    df["lateral_disp"] = (df["x_m"] - df["prev_x"]).abs()

    # Forward fill NaN from shift for first frame of each vehicle
    df["prev_lane"] = df["prev_lane"].fillna(df["Lane_ID"])
    df["prev_speed"] = df["prev_speed"].fillna(df["speed_ms"])
    df["prev_x"] = df["prev_x"].fillna(df["x_m"])
    df["prev_y"] = df["prev_y"].fillna(df["y_m"])

    return df


def _sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with physically impossible values."""
    before = len(df)
    df = df[df["speed_ms"].between(0, 50)]       # 0–180 km/h
    df = df[df["accel_ms2"].between(-10, 10)]    # ±10 m/s²
    df = df[df["length_m"].between(1, 30)]
    df = df[df["width_m"].between(0.5, 5)]
    logger.info(f"Sanity checks removed {before - len(df)} rows")
    return df


def preprocess(filepath: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Returns a clean DataFrame ready for scenario detection.
    """
    df = load_raw(filepath)
    df = _coerce_numeric_columns(df)
    df = _drop_missing(df)
    df = _cast_dtypes(df)
    df = _remove_duplicates(df)
    df = _filter_spatial(df)
    df = _filter_lanes(df)
    df = _fix_monotonicity(df)
    df = _convert_units(df)
    df = _add_derived_fields(df)
    df = _sanity_checks(df)
    df = df.reset_index(drop=True)
    logger.info(f"Clean data shape: {df.shape}")
    logger.info(f"Unique vehicles: {df['Vehicle_ID'].nunique()}")
    logger.info(
        f"Frame range: {df['Frame_ID'].min()} – {df['Frame_ID'].max()}")
    return df
