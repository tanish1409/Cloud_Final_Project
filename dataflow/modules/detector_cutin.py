"""
Lane cut-in scenario detector.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

FRAME_RATE_HZ = 10
WINDOW_FRAMES = 5 * FRAME_RATE_HZ
CUTIN_SPEED_DROP_MS = 1.0

def _build_vehicle_dict(df):
    return {vid: grp.reset_index(drop=True) for vid, grp in df.groupby("Vehicle_ID")}

def detect_lane_cutin(df):
    events = []
    vdict = _build_vehicle_dict(df)
    for follower_id, follower_df in vdict.items():
        follower_sorted = follower_df.sort_values("Frame_ID").reset_index(drop=True)
        preceding_series = follower_sorted["Preceding"]
        change_mask = (preceding_series != preceding_series.shift()) & (preceding_series != 0)
        change_indices = follower_sorted[change_mask].index.tolist()
        for idx in change_indices:
            if idx == 0:
                continue
            new_leader_id = int(follower_sorted.at[idx, "Preceding"])
            old_leader_id = int(follower_sorted.at[idx - 1, "Preceding"])
            if new_leader_id == 0 or new_leader_id not in vdict:
                continue
            frame_of_cutin = int(follower_sorted.at[idx, "Frame_ID"])
            cutter_df = vdict[new_leader_id]
            cutter_window = cutter_df[
                (cutter_df["Frame_ID"] >= frame_of_cutin - 10) &
                (cutter_df["Frame_ID"] <= frame_of_cutin + 10)
            ]
            if cutter_window.empty:
                continue
            cutter_lane_changed = cutter_window["Lane_ID"].nunique() > 1
            if not cutter_lane_changed:
                continue
            before_window = follower_sorted[
                follower_sorted["Frame_ID"].between(frame_of_cutin - 20, frame_of_cutin)
            ]
            after_window = follower_sorted[
                follower_sorted["Frame_ID"].between(frame_of_cutin, frame_of_cutin + 30)
            ]
            if before_window.empty or after_window.empty:
                continue
            speed_before = before_window["speed_ms"].mean()
            speed_after = after_window["speed_ms"].mean()
            speed_drop = speed_before - speed_after
            if speed_drop < CUTIN_SPEED_DROP_MS:
                continue
            gap_after_row = follower_sorted.at[idx, "Space_Headway"]
            gap_after_m = gap_after_row * 0.3048 if gap_after_row > 0 else None
            events.append({
                "ego_id": follower_id, "cutter_id": new_leader_id,
                "old_leader_id": old_leader_id,
                "start_frame": frame_of_cutin - 20,
                "end_frame": frame_of_cutin + 30,
                "scenario_type": "lane_cutin",
                "speed_drop_ms": round(speed_drop, 3),
                "gap_after_m": round(gap_after_m, 2) if gap_after_m else None,
                "cutin_frame": frame_of_cutin,
            })
    result = pd.DataFrame(events)
    logger.info(f"Lane cut-in events detected: {len(result)}")
    return result
