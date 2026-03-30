"""
On-ramp merge scenario detector.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

FRAME_RATE_HZ = 10
WINDOW_FRAMES = 5 * FRAME_RATE_HZ
ONRAMP_LANE_ID = 6
MAINLINE_LANES = {1, 2, 3, 4, 5}
MERGE_MAX_FRAMES = WINDOW_FRAMES
MERGE_MIN_LATERAL_M = 1.5

def _build_vehicle_dict(df):
    return {vid: grp.reset_index(drop=True) for vid, grp in df.groupby("Vehicle_ID")}

def detect_onramp_merge(df):
    events = []
    vdict = _build_vehicle_dict(df)
    for ego_id, ego_df in vdict.items():
        on_ramp = ego_df[ego_df["Lane_ID"] == ONRAMP_LANE_ID]
        if on_ramp.empty:
            continue
        ego_sorted = ego_df.sort_values("Frame_ID")
        lanes = ego_sorted["Lane_ID"].values
        frames = ego_sorted["Frame_ID"].values
        for i in range(len(lanes) - 1):
            if lanes[i] == ONRAMP_LANE_ID and lanes[i + 1] in MAINLINE_LANES:
                merge_start_frame = frames[i]
                merge_end_frame = frames[i + 1]
                window_start = merge_start_frame - MERGE_MAX_FRAMES
                window_df = ego_sorted[
                    (ego_sorted["Frame_ID"] >= window_start) &
                    (ego_sorted["Frame_ID"] <= merge_end_frame)
                ]
                if window_df.empty:
                    continue
                lateral_total = window_df["lateral_disp"].sum() * 0.3048
                if lateral_total < MERGE_MIN_LATERAL_M:
                    continue
                events.append({
                    "ego_id": ego_id,
                    "start_frame": int(merge_start_frame),
                    "end_frame": int(merge_end_frame),
                    "scenario_type": "onramp_merge",
                    "from_lane": int(lanes[i]),
                    "to_lane": int(lanes[i + 1]),
                    "lateral_disp_m": round(lateral_total, 2),
                })
                break
    result = pd.DataFrame(events)
    logger.info(f"On-ramp merge events detected: {len(result)}")
    return result
