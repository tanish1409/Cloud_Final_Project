"""
Window segmentation module.
Updated for Phase 2 to handle frame gaps caused by preprocessing.

Instead of selecting a frame-ID range and hoping all 50 frames exist,
this version queries the ego vehicle's actual present frames in the
clean data and selects 50 consecutive *present* frames centred around
the event midpoint.  This guarantees every window contains exactly 50
contiguous data points for the ego vehicle with no gaps.

Surrounding vehicles are filtered to only include those present for at
least 70 % of the ego's window frames (MIN_SURROUND_COVERAGE).
"""
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

FRAME_RATE_HZ = 10
WINDOW_FRAMES = 5 * FRAME_RATE_HZ          # 50 present frames required
SURROUND_RADIUS = 60.0                      # metres
MIN_SURROUND_COVERAGE = 0.70                # 70 % frame presence for surrounding vehicles


# ── Build a lookup of each vehicle's sorted frame array ───────────────────────

def _build_ego_frame_index(df):
    """Return {vehicle_id: sorted numpy array of Frame_IDs} for fast lookup."""
    return {
        vid: grp["Frame_ID"].sort_values().values
        for vid, grp in df.groupby("Vehicle_ID")
    }


# ── Select 50 consecutive present frames centred on the event ─────────────────

def _select_present_frames(ego_frames, event_start, event_end):
    """
    Given the ego vehicle's sorted array of *present* Frame_IDs, return the
    start and end Frame_IDs of a 50-consecutive-present-frame window centred
    as close to the event midpoint as possible.

    Returns (win_start_frame, win_end_frame, frame_array) or None if the ego
    does not have enough frames.
    """
    n = len(ego_frames)
    if n < WINDOW_FRAMES:
        return None                          # not enough data for this vehicle

    midpoint = (event_start + event_end) // 2

    # Find the index in ego_frames closest to the midpoint
    mid_idx = int(np.searchsorted(ego_frames, midpoint, side="left"))
    mid_idx = min(mid_idx, n - 1)

    # Centre a 50-frame slice around mid_idx
    half = WINDOW_FRAMES // 2
    start_idx = mid_idx - half
    end_idx = start_idx + WINDOW_FRAMES      # exclusive

    # Clamp to array bounds
    if start_idx < 0:
        start_idx = 0
        end_idx = WINDOW_FRAMES
    if end_idx > n:
        end_idx = n
        start_idx = n - WINDOW_FRAMES

    selected = ego_frames[start_idx:end_idx]
    return int(selected[0]), int(selected[-1]), selected


# ── Extract ego trajectory for the selected frames ────────────────────────────

def _get_ego_trajectory(df, ego_id, frame_set):
    """Return the ego trajectory for exactly the frames in *frame_set*."""
    ego_df = df[
        (df["Vehicle_ID"] == ego_id) & (df["Frame_ID"].isin(frame_set))
    ].sort_values("Frame_ID").copy()

    return ego_df[["Frame_ID", "x_m", "y_m", "speed_ms", "accel_ms2", "Lane_ID"]].rename(
        columns={"Frame_ID": "frame"}
    ).round(3).to_dict(orient="records")


# ── Extract surrounding vehicles with coverage check ──────────────────────────

def _get_surrounding_vehicles(df, ego_id, frame_set):
    """
    Return surrounding vehicles that are:
      1. Within SURROUND_RADIUS metres of the ego's mean position, AND
      2. Present for at least MIN_SURROUND_COVERAGE of the ego's window frames.
    """
    n_ego_frames = len(frame_set)

    # Ego mean position
    ego_window = df[
        (df["Vehicle_ID"] == ego_id) & (df["Frame_ID"].isin(frame_set))
    ]
    if ego_window.empty:
        return []
    ego_x_mean = ego_window["x_m"].mean()
    ego_y_mean = ego_window["y_m"].mean()

    # All other vehicles in the frame range
    window_df = df[
        (df["Frame_ID"].isin(frame_set)) & (df["Vehicle_ID"] != ego_id)
    ].copy()
    if window_df.empty:
        return []

    result = []
    for vid, grp in window_df.groupby("Vehicle_ID"):
        # Coverage check: vehicle must be present for >= 70 % of ego frames
        coverage = len(grp) / n_ego_frames
        if coverage < MIN_SURROUND_COVERAGE:
            continue

        # Distance check
        dist = np.sqrt(
            (grp["x_m"].mean() - ego_x_mean) ** 2 +
            (grp["y_m"].mean() - ego_y_mean) ** 2
        )
        if dist > SURROUND_RADIUS:
            continue

        grp_sorted = grp.sort_values("Frame_ID")
        result.append({
            "vehicle_id": int(vid),
            "frames": grp_sorted["Frame_ID"].tolist(),
            "x_m": grp_sorted["x_m"].round(3).tolist(),
            "y_m": grp_sorted["y_m"].round(3).tolist(),
            "speed_ms": grp_sorted["speed_ms"].round(3).tolist(),
            "lane_ids": grp_sorted["Lane_ID"].tolist(),
        })
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def segment_windows(df, events):
    ego_frame_idx = _build_ego_frame_index(df)
    samples = []
    skipped_no_frames = 0

    for _, event in events.iterrows():
        ego_id = int(event["ego_id"])
        ev_start = int(event["start_frame"])
        ev_end = int(event["end_frame"])
        sc_type = event["scenario_type"]

        # Get the ego's present frames
        ego_frames = ego_frame_idx.get(ego_id)
        if ego_frames is None or len(ego_frames) < WINDOW_FRAMES:
            skipped_no_frames += 1
            logger.warning(
                f"Ego {ego_id} has insufficient frames "
                f"({0 if ego_frames is None else len(ego_frames)} < {WINDOW_FRAMES}); skipping."
            )
            continue

        result = _select_present_frames(ego_frames, ev_start, ev_end)
        if result is None:
            skipped_no_frames += 1
            continue

        win_start, win_end, frame_set = result
        frame_set_py = set(int(f) for f in frame_set)

        ego_traj = _get_ego_trajectory(df, ego_id, frame_set_py)
        surrounding = _get_surrounding_vehicles(df, ego_id, frame_set_py)

        if not ego_traj:
            logger.warning(f"No ego trajectory for vehicle {ego_id} in window; skipping.")
            continue

        sample = {
            "scenario_type": sc_type,
            "ego_id": ego_id,
            "window_start_frame": win_start,
            "window_end_frame": win_end,
            "ego_frame_count": len(ego_traj),
            "ego_trajectory": json.dumps(ego_traj),
            "surrounding_vehicles": json.dumps(surrounding),
            "num_surrounding": len(surrounding),
        }
        # Carry over event-specific metadata
        for col in event.index:
            if col not in sample and col not in ("start_frame", "end_frame"):
                val = event[col]
                if pd.notna(val):
                    sample[col] = val
        samples.append(sample)

    if skipped_no_frames:
        logger.info(f"Skipped {skipped_no_frames} events due to insufficient ego frames.")

    result_df = pd.DataFrame(samples)
    logger.info(f"Windows segmented: {len(result_df)}")
    if not result_df.empty:
        logger.info(result_df["scenario_type"].value_counts().to_string())
    return result_df
