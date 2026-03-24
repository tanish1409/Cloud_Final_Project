"""
Window segmentation module.
Copied from Phase 1 modules/windower.py — unchanged.
"""
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

FRAME_RATE_HZ = 10
WINDOW_FRAMES = 5 * FRAME_RATE_HZ
SURROUND_RADIUS = 60.0


def _get_surrounding_vehicles(df, ego_id, start_frame, end_frame):
    window_df = df[
        (df["Frame_ID"] >= start_frame) & (df["Frame_ID"] <= end_frame) & (df["Vehicle_ID"] != ego_id)
    ].copy()
    ego_window = df[
        (df["Frame_ID"] >= start_frame) & (df["Frame_ID"] <= end_frame) & (df["Vehicle_ID"] == ego_id)
    ]
    if ego_window.empty or window_df.empty:
        return []

    ego_x_mean = ego_window["x_m"].mean()
    ego_y_mean = ego_window["y_m"].mean()

    surrounding_ids = []
    for vid, grp in window_df.groupby("Vehicle_ID"):
        dist = np.sqrt((grp["x_m"].mean() - ego_x_mean)**2 + (grp["y_m"].mean() - ego_y_mean)**2)
        if dist <= SURROUND_RADIUS:
            surrounding_ids.append(vid)

    result = []
    for vid in surrounding_ids:
        v_df = window_df[window_df["Vehicle_ID"] == vid]
        result.append({
            "vehicle_id": int(vid),
            "frames": v_df["Frame_ID"].tolist(),
            "x_m": v_df["x_m"].round(3).tolist(),
            "y_m": v_df["y_m"].round(3).tolist(),
            "speed_ms": v_df["speed_ms"].round(3).tolist(),
            "lane_ids": v_df["Lane_ID"].tolist(),
        })
    return result


def _get_ego_trajectory(df, ego_id, start_frame, end_frame):
    ego_df = df[
        (df["Vehicle_ID"] == ego_id) & (df["Frame_ID"] >= start_frame) & (df["Frame_ID"] <= end_frame)
    ].copy()
    return ego_df[["Frame_ID", "x_m", "y_m", "speed_ms", "accel_ms2", "Lane_ID"]].rename(
        columns={"Frame_ID": "frame"}
    ).round(3).to_dict(orient="records")


def _centre_window(event_start, event_end, df):
    global_min = int(df["Frame_ID"].min())
    global_max = int(df["Frame_ID"].max())
    midpoint = (event_start + event_end) // 2
    win_start = max(global_min, midpoint - WINDOW_FRAMES // 2)
    win_end = min(global_max, win_start + WINDOW_FRAMES - 1)
    win_start = max(global_min, win_end - WINDOW_FRAMES + 1)
    return win_start, win_end


def segment_windows(df, events):
    samples = []
    for _, event in events.iterrows():
        ego_id = int(event["ego_id"])
        ev_start = int(event["start_frame"])
        ev_end = int(event["end_frame"])
        sc_type = event["scenario_type"]

        win_start, win_end = _centre_window(ev_start, ev_end, df)
        ego_traj = _get_ego_trajectory(df, ego_id, win_start, win_end)
        surrounding = _get_surrounding_vehicles(df, ego_id, win_start, win_end)

        if not ego_traj:
            logger.warning(f"No ego trajectory for vehicle {ego_id} in window; skipping.")
            continue

        sample = {
            "scenario_type": sc_type,
            "ego_id": ego_id,
            "window_start_frame": win_start,
            "window_end_frame": win_end,
            "ego_trajectory": json.dumps(ego_traj),
            "surrounding_vehicles": json.dumps(surrounding),
            "num_surrounding": len(surrounding),
        }
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
