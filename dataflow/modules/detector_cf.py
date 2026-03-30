"""
Car-following scenario detector.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

FRAME_RATE_HZ = 10
CF_MAX_GAP_M = 80.0
CF_MIN_DURATION_FRAMES = 30
CF_SAME_LANE_RATIO = 0.8

def _build_vehicle_dict(df):
    return {vid: grp.reset_index(drop=True) for vid, grp in df.groupby("Vehicle_ID")}

def detect_car_following(df):
    events = []
    vdict = _build_vehicle_dict(df)
    for ego_id, ego_df in vdict.items():
        ego_with_leader = ego_df[ego_df["Preceding"] != 0].copy()
        if ego_with_leader.empty:
            continue
        ego_with_leader["leader_id"] = ego_with_leader["Preceding"]
        ego_with_leader["block"] = (
            ego_with_leader["leader_id"] != ego_with_leader["leader_id"].shift()
        ).cumsum()
        for (leader_id, block_id), block in ego_with_leader.groupby(["leader_id", "block"]):
            if len(block) < CF_MIN_DURATION_FRAMES:
                continue
            if leader_id not in vdict:
                continue
            leader_df = vdict[leader_id]
            merged = block.merge(
                leader_df[["Frame_ID", "Lane_ID", "speed_ms", "y_m"]],
                on="Frame_ID", suffixes=("_ego", "_ldr"),
            )
            if len(merged) < CF_MIN_DURATION_FRAMES:
                continue
            lane_match = (merged["Lane_ID_ego"] == merged["Lane_ID_ldr"]).mean()
            if lane_match < CF_SAME_LANE_RATIO:
                continue
            gap = (merged["y_m_ldr"] - merged["y_m_ego"]).abs()
            if gap.mean() > CF_MAX_GAP_M:
                continue
            moving = merged[merged["speed_ms_ego"] > 1.0]
            if not moving.empty:
                thw = (moving["y_m_ldr"] - moving["y_m_ego"]).abs() / moving["speed_ms_ego"]
                avg_thw = thw.mean()
                if avg_thw < 0.5 or avg_thw > 4.0:
                    continue
            events.append({
                "ego_id": ego_id, "leader_id": leader_id,
                "start_frame": int(block["Frame_ID"].iloc[0]),
                "end_frame": int(block["Frame_ID"].iloc[-1]),
                "scenario_type": "car_following",
                "avg_gap_m": round(gap.mean(), 2),
                "avg_thw_s": round(avg_thw if not moving.empty else 0, 2),
                "duration_frames": len(block),
            })
    result = pd.DataFrame(events)
    logger.info(f"Car-following events detected: {len(result)}")
    return result
