"""
Scenario Detector Module
------------------------
Applies rule-based logic to detect three driving scenarios from NGSIM I-80:

  1. Car-Following       – vehicle A follows vehicle B in the same lane
  2. On-Ramp Merge       – vehicle transitions from ramp lane → mainline
  3. Lane Cut-In         – vehicle abruptly enters the lane ahead of another

Each detector returns a DataFrame of detected events with columns:
  ego_id, start_frame, end_frame, scenario_type, [scenario-specific metadata]
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Tunable thresholds ────────────────────────────────────────────────────────
FRAME_RATE_HZ = 10
WINDOW_FRAMES = 5 * FRAME_RATE_HZ   # 5-second window = 50 frames

# Car-following
CF_MAX_GAP_M = 80.0    # max longitudinal gap between leader & follower (m)
CF_MIN_DURATION_FRAMES = 30      # must follow for ≥ 3 s
CF_SPEED_CORR_MIN = 0.0     # minimum Pearson correlation of speeds
CF_SAME_LANE_RATIO = 0.8     # fraction of window where lanes match

# On-ramp merge
ONRAMP_LANE_ID = 6
MAINLINE_LANES = {1, 2, 3, 4, 5}
MERGE_MAX_FRAMES = WINDOW_FRAMES      # must complete within 5 s
MERGE_MIN_LATERAL_M = 1.5    # minimum lateral displacement during merge (m)

# Lane cut-in
# gap before cut-in (original follower to leader)
CUTIN_MAX_GAP_BEFORE_M = 60.0
CUTIN_MAX_GAP_AFTER_M = 30.0   # gap after cut-in (original follower to cutter)
CUTIN_SPEED_DROP_MS = 1.0    # follower must slow by ≥ 1 m/s after cut-in
CUTIN_MAX_FRAMES = WINDOW_FRAMES


def _build_vehicle_dict(df: pd.DataFrame) -> dict:
    """Index data by Vehicle_ID for fast per-vehicle lookup."""
    return {vid: grp.reset_index(drop=True) for vid, grp in df.groupby("Vehicle_ID")}


# ── Scenario 1: Car-Following ─────────────────────────────────────────────────

def detect_car_following(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detection logic:
      - Follower and leader share the same lane for ≥ CF_SAME_LANE_RATIO of window
      - Longitudinal gap ≤ CF_MAX_GAP_M throughout
      - Speed time-series of both vehicles are correlated (≥ CF_SPEED_CORR_MIN)
      - Event lasts ≥ CF_MIN_DURATION_FRAMES frames
    Uses NGSIM 'Preceding' field as the ground-truth leader pointer.
    """
    events = []
    vdict = _build_vehicle_dict(df)

    for ego_id, ego_df in vdict.items():
        # Only keep frames where a leader is recorded
        ego_with_leader = ego_df[ego_df["Preceding"] != 0].copy()
        if ego_with_leader.empty:
            continue

        # Group consecutive frames with the SAME leader
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

            # Align frames
            merged = block.merge(
                leader_df[["Frame_ID", "Lane_ID", "speed_ms", "y_m"]],
                on="Frame_ID", suffixes=("_ego", "_ldr"),
            )
            if len(merged) < CF_MIN_DURATION_FRAMES:
                continue

            # Lane match ratio
            lane_match = (merged["Lane_ID_ego"] ==
                          merged["Lane_ID_ldr"]).mean()
            if lane_match < CF_SAME_LANE_RATIO:
                continue

            # Gap check
            gap = (merged["y_m_ldr"] - merged["y_m_ego"]).abs()
            if gap.mean() > CF_MAX_GAP_M:
                continue

            # Speed correlation
            # Speed correlation — skip check in congested low-speed conditions
            avg_speed = merged["speed_ms_ego"].mean()
            if avg_speed < 3.0:
                corr = 1.0  # congested stop-and-go, correlation unreliable
            elif merged["speed_ms_ego"].std() < 0.01 or merged["speed_ms_ldr"].std() < 0.01:
                corr = 1.0  # both nearly stationary
            else:
                corr = merged["speed_ms_ego"].corr(merged["speed_ms_ldr"])
            if pd.isna(corr) or corr < CF_SPEED_CORR_MIN:
                continue

            events.append({
                "ego_id":       ego_id,
                "leader_id":    leader_id,
                "start_frame":  int(block["Frame_ID"].iloc[0]),
                "end_frame":    int(block["Frame_ID"].iloc[-1]),
                "scenario_type": "car_following",
                "avg_gap_m":    round(gap.mean(), 2),
                "speed_corr":   round(corr, 3),
                "duration_frames": len(block),
            })

    result = pd.DataFrame(events)
    logger.info(f"Car-following events detected: {len(result)}")
    return result


# ── Scenario 2: On-Ramp Merge ─────────────────────────────────────────────────

def detect_onramp_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detection logic:
      - Vehicle starts in ramp lane (Lane_ID == ONRAMP_LANE_ID)
      - Within MERGE_MAX_FRAMES, Lane_ID changes to a mainline lane
      - Cumulative lateral displacement ≥ MERGE_MIN_LATERAL_M
    """
    events = []
    vdict = _build_vehicle_dict(df)

    for ego_id, ego_df in vdict.items():
        # Find frames where vehicle is on the ramp
        on_ramp = ego_df[ego_df["Lane_ID"] == ONRAMP_LANE_ID]
        if on_ramp.empty:
            continue

        # Detect transition: last ramp frame → first mainline frame
        ego_sorted = ego_df.sort_values("Frame_ID")
        lanes = ego_sorted["Lane_ID"].values
        frames = ego_sorted["Frame_ID"].values

        for i in range(len(lanes) - 1):
            if lanes[i] == ONRAMP_LANE_ID and lanes[i + 1] in MAINLINE_LANES:
                merge_start_frame = frames[i]
                merge_end_frame = frames[i + 1]

                # Find frames ≤ MERGE_MAX_FRAMES before the transition
                window_start = merge_start_frame - MERGE_MAX_FRAMES
                window_df = ego_sorted[
                    (ego_sorted["Frame_ID"] >= window_start) &
                    (ego_sorted["Frame_ID"] <= merge_end_frame)
                ]

                if window_df.empty:
                    continue

                lateral_total = window_df["lateral_disp"].sum(
                ) * 0.3048  # ft→m
                if lateral_total < MERGE_MIN_LATERAL_M:
                    continue

                events.append({
                    "ego_id":         ego_id,
                    "start_frame":    int(merge_start_frame),
                    "end_frame":      int(merge_end_frame),
                    "scenario_type":  "onramp_merge",
                    "from_lane":      int(lanes[i]),
                    "to_lane":        int(lanes[i + 1]),
                    "lateral_disp_m": round(lateral_total, 2),
                })
                break  # one merge event per vehicle is enough

    result = pd.DataFrame(events)
    logger.info(f"On-ramp merge events detected: {len(result)}")
    return result


# ── Scenario 3: Lane Cut-In ───────────────────────────────────────────────────

def detect_lane_cutin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detection logic:
      - Cutter vehicle changes lane and becomes the 'Preceding' vehicle of the follower
      - The gap between follower and its new leader drops significantly
      - The follower decelerates (speed drops ≥ CUTIN_SPEED_DROP_MS) after the event
    """
    events = []
    vdict = _build_vehicle_dict(df)

    # For every follower vehicle, track changes in their 'Preceding' pointer
    for follower_id, follower_df in vdict.items():
        follower_sorted = follower_df.sort_values(
            "Frame_ID").reset_index(drop=True)

        # Detect when the Preceding vehicle changes (new vehicle cuts in)
        preceding_series = follower_sorted["Preceding"]
        change_mask = (preceding_series != preceding_series.shift()) & (
            preceding_series != 0)
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

            # Check cutter changed lane around the cutin frame
            cutter_window = cutter_df[
                (cutter_df["Frame_ID"] >= frame_of_cutin - 10) &
                (cutter_df["Frame_ID"] <= frame_of_cutin + 10)
            ]
            if cutter_window.empty:
                continue

            cutter_lane_changed = cutter_window["Lane_ID"].nunique() > 1
            if not cutter_lane_changed:
                continue

            # Check follower speed drop after cutin
            before_window = follower_sorted[
                follower_sorted["Frame_ID"].between(
                    frame_of_cutin - 20, frame_of_cutin)
            ]
            after_window = follower_sorted[
                follower_sorted["Frame_ID"].between(
                    frame_of_cutin, frame_of_cutin + 30)
            ]

            if before_window.empty or after_window.empty:
                continue

            speed_before = before_window["speed_ms"].mean()
            speed_after = after_window["speed_ms"].mean()
            speed_drop = speed_before - speed_after

            if speed_drop < CUTIN_SPEED_DROP_MS:
                continue

            # Gap after cut-in
            gap_after_row = follower_sorted.at[idx, "Space_Headway"]
            gap_after_m = gap_after_row * 0.3048 if gap_after_row > 0 else None

            events.append({
                "ego_id":         follower_id,
                "cutter_id":      new_leader_id,
                "old_leader_id":  old_leader_id,
                "start_frame":    frame_of_cutin - 20,
                "end_frame":      frame_of_cutin + 30,
                "scenario_type":  "lane_cutin",
                "speed_drop_ms":  round(speed_drop, 3),
                "gap_after_m":    round(gap_after_m, 2) if gap_after_m else None,
                "cutin_frame":    frame_of_cutin,
            })

    result = pd.DataFrame(events)
    logger.info(f"Lane cut-in events detected: {len(result)}")
    return result


# ── Combined entry point ───────────────────────────────────────────────────────

def detect_all_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three detectors and return a unified events DataFrame."""
    cf = detect_car_following(df)
    merge = detect_onramp_merge(df)
    cutin = detect_lane_cutin(df)

    all_events = pd.concat([cf, merge, cutin], ignore_index=True)
    all_events = all_events.sort_values(
        ["ego_id", "start_frame"]).reset_index(drop=True)
    logger.info(f"Total events detected: {len(all_events)}")
    return all_events
