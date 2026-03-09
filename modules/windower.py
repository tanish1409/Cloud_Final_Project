"""
Window Segmenter Module
-----------------------
Slices each detected scenario event into a 5-second (50-frame) window
and enriches it with ego vehicle + surrounding vehicle data.

Output schema (one row per window):
  scenario_type, ego_id, window_start_frame, window_end_frame,
  ego_trajectory (list of dicts), surrounding_vehicles (list of dicts),
  [scenario-specific metadata fields]
"""

import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

FRAME_RATE_HZ   = 10
WINDOW_FRAMES   = 5 * FRAME_RATE_HZ   # 50 frames = 5 seconds
SURROUND_RADIUS = 60.0                 # metres — include vehicles within this radius


def _get_surrounding_vehicles(
    df: pd.DataFrame,
    ego_id: int,
    start_frame: int,
    end_frame: int,
) -> list:
    """Return trajectory data for all vehicles near ego in the window."""
    window_df = df[
        (df["Frame_ID"] >= start_frame) &
        (df["Frame_ID"] <= end_frame) &
        (df["Vehicle_ID"] != ego_id)
    ].copy()

    ego_window = df[
        (df["Frame_ID"] >= start_frame) &
        (df["Frame_ID"] <= end_frame) &
        (df["Vehicle_ID"] == ego_id)
    ]

    if ego_window.empty or window_df.empty:
        return []

    ego_x_mean = ego_window["x_m"].mean()
    ego_y_mean = ego_window["y_m"].mean()

    # Keep only nearby vehicles
    surrounding_ids = []
    for vid, grp in window_df.groupby("Vehicle_ID"):
        dist = np.sqrt(
            (grp["x_m"].mean() - ego_x_mean) ** 2 +
            (grp["y_m"].mean() - ego_y_mean) ** 2
        )
        if dist <= SURROUND_RADIUS:
            surrounding_ids.append(vid)

    result = []
    for vid in surrounding_ids:
        v_df = window_df[window_df["Vehicle_ID"] == vid]
        result.append({
            "vehicle_id": int(vid),
            "frames":     v_df["Frame_ID"].tolist(),
            "x_m":        v_df["x_m"].round(3).tolist(),
            "y_m":        v_df["y_m"].round(3).tolist(),
            "speed_ms":   v_df["speed_ms"].round(3).tolist(),
            "lane_ids":   v_df["Lane_ID"].tolist(),
        })
    return result


def _get_ego_trajectory(
    df: pd.DataFrame,
    ego_id: int,
    start_frame: int,
    end_frame: int,
) -> list:
    """Return ego vehicle trajectory for the window."""
    ego_df = df[
        (df["Vehicle_ID"] == ego_id) &
        (df["Frame_ID"] >= start_frame) &
        (df["Frame_ID"] <= end_frame)
    ].copy()

    return ego_df[[
        "Frame_ID", "x_m", "y_m", "speed_ms", "accel_ms2", "Lane_ID"
    ]].rename(columns={"Frame_ID": "frame"}).round(3).to_dict(orient="records")


def _centre_window(event_start: int, event_end: int, df: pd.DataFrame) -> tuple:
    """
    Centre the 50-frame window around the event midpoint.
    Clamp to the available frame range.
    """
    global_min = int(df["Frame_ID"].min())
    global_max = int(df["Frame_ID"].max())

    midpoint     = (event_start + event_end) // 2
    win_start    = max(global_min, midpoint - WINDOW_FRAMES // 2)
    win_end      = min(global_max, win_start + WINDOW_FRAMES - 1)
    win_start    = max(global_min, win_end - WINDOW_FRAMES + 1)  # re-adjust start
    return win_start, win_end


def segment_windows(
    df: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build labeled 5-second windows for every detected event.

    Parameters
    ----------
    df      : clean preprocessed NGSIM DataFrame
    events  : output of detect_all_scenarios()

    Returns
    -------
    DataFrame with one row per window sample.
    """
    samples = []

    for _, event in events.iterrows():
        ego_id      = int(event["ego_id"])
        ev_start    = int(event["start_frame"])
        ev_end      = int(event["end_frame"])
        sc_type     = event["scenario_type"]

        win_start, win_end = _centre_window(ev_start, ev_end, df)

        ego_traj    = _get_ego_trajectory(df, ego_id, win_start, win_end)
        surrounding = _get_surrounding_vehicles(df, ego_id, win_start, win_end)

        if not ego_traj:
            logger.warning(f"No ego trajectory for vehicle {ego_id} in window; skipping.")
            continue

        # Build base sample
        sample = {
            "scenario_type":       sc_type,
            "ego_id":              ego_id,
            "window_start_frame":  win_start,
            "window_end_frame":    win_end,
            "ego_trajectory":      json.dumps(ego_traj),
            "surrounding_vehicles": json.dumps(surrounding),
            "num_surrounding":     len(surrounding),
        }

        # Carry over scenario-specific metadata
        for col in event.index:
            if col not in sample and col not in ("start_frame", "end_frame"):
                val = event[col]
                if pd.notna(val):
                    sample[col] = val

        samples.append(sample)

    result = pd.DataFrame(samples)
    logger.info(f"Windows segmented: {len(result)}")
    if not result.empty:
        logger.info(result["scenario_type"].value_counts().to_string())
    return result