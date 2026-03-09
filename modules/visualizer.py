"""
Visualizer Module
-----------------
Generates plots to visually confirm that each scenario type was correctly extracted.

Plot types:
  1. Car-Following  – time-series of gap + speed comparison (ego vs leader)
  2. On-Ramp Merge  – XY trajectory with lane boundaries overlaid
  3. Lane Cut-In    – position snapshot before/after + ego speed time-series

Usage:
    from modules.visualizer import visualize_scenario
    visualize_scenario(df, sample_row, save_path="output/event_42.png")
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless rendering (no display needed in Cloud Run)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import logging

logger = logging.getLogger(__name__)

# I-80 approximate lane boundary X positions (metres, from centre line)
# Lane 1 is leftmost; lane 6 (ramp) rightmost
LANE_BOUNDARIES_M = [0, 3.7, 7.4, 11.1, 14.8, 18.5, 22.5]
LANE_COLOURS = {
    "car_following": "#2196F3",
    "onramp_merge":  "#4CAF50",
    "lane_cutin":    "#F44336",
}
SCENARIO_TITLES = {
    "car_following": "Scenario 1 – Car Following",
    "onramp_merge":  "Scenario 2 – On-Ramp Merge",
    "lane_cutin":    "Scenario 3 – Lane Cut-In",
}


def _parse_json_col(val):
    if isinstance(val, str):
        return json.loads(val)
    return val if val is not None else []


def _draw_lane_boundaries(ax, y_min, y_max):
    for x in LANE_BOUNDARIES_M:
        ax.axhline(y=x, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    for i in range(len(LANE_BOUNDARIES_M) - 1):
        mid = (LANE_BOUNDARIES_M[i] + LANE_BOUNDARIES_M[i + 1]) / 2
        ax.text(y_min + 5, mid, f"L{i+1}", fontsize=7, color="gray", va="center")


# ── Car-Following Visualization ───────────────────────────────────────────────

def _plot_car_following(df: pd.DataFrame, sample: dict, ax_list):
    ego_traj = _parse_json_col(sample["ego_trajectory"])
    surrounding = _parse_json_col(sample["surrounding_vehicles"])

    ego_frames = [r["frame"] for r in ego_traj]
    ego_speed  = [r["speed_ms"] for r in ego_traj]
    ego_y      = [r["y_m"] for r in ego_traj]

    ax_gap, ax_speed = ax_list

    # Find leader trajectory
    leader_id = sample.get("leader_id")
    ldr_frames, ldr_speed, ldr_y = [], [], []
    for sv in surrounding:
        if sv["vehicle_id"] == leader_id:
            ldr_frames = sv["frames"]
            ldr_speed  = sv["speed_ms"]
            ldr_y      = sv["y_m"]
            break

    # Gap plot
    if ldr_y and ego_y:
        common_frames = sorted(set(ego_frames) & set(ldr_frames))
        ego_y_dict  = dict(zip(ego_frames, ego_y))
        ldr_y_dict  = dict(zip(ldr_frames, ldr_y))
        gaps = [abs(ldr_y_dict[f] - ego_y_dict[f]) for f in common_frames]
        time_s = [(f - common_frames[0]) / 10.0 for f in common_frames]
        ax_gap.plot(time_s, gaps, color="#2196F3", linewidth=2)
        ax_gap.axhline(y=80, color="red", linestyle=":", linewidth=1, label="Max gap threshold (80 m)")
        ax_gap.set_ylabel("Longitudinal Gap (m)")
        ax_gap.set_xlabel("Time (s)")
        ax_gap.set_title("Gap between Ego and Leader")
        ax_gap.legend(fontsize=8)
        ax_gap.grid(True, alpha=0.3)

    # Speed comparison
    time_ego = [(f - ego_frames[0]) / 10.0 for f in ego_frames]
    ax_speed.plot(time_ego, ego_speed, label="Ego", color="#2196F3", linewidth=2)
    if ldr_speed:
        time_ldr = [(f - ego_frames[0]) / 10.0 for f in ldr_frames
                    if f >= ego_frames[0]]
        ldr_speed_clipped = [s for f, s in zip(ldr_frames, ldr_speed)
                             if f >= ego_frames[0]]
        ax_speed.plot(time_ldr, ldr_speed_clipped,
                      label=f"Leader (ID {leader_id})", color="#FF9800",
                      linewidth=2, linestyle="--")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_title("Speed Comparison")
    ax_speed.legend(fontsize=8)
    ax_speed.grid(True, alpha=0.3)


# ── On-Ramp Merge Visualization ───────────────────────────────────────────────

def _plot_onramp_merge(df: pd.DataFrame, sample: dict, ax_list):
    ego_traj    = _parse_json_col(sample["ego_trajectory"])
    surrounding = _parse_json_col(sample["surrounding_vehicles"])

    ax_traj, ax_lane = ax_list

    ego_y    = [r["y_m"]      for r in ego_traj]
    ego_x    = [r["x_m"]      for r in ego_traj]
    ego_lane = [r["Lane_ID"]  for r in ego_traj]
    frames   = [r["frame"]    for r in ego_traj]
    time_s   = [(f - frames[0]) / 10.0 for f in frames]

    # XY trajectory coloured by time
    sc = ax_traj.scatter(ego_y, ego_x, c=time_s, cmap="plasma", s=15, zorder=3)
    plt.colorbar(sc, ax=ax_traj, label="Time (s)")

    # Surrounding vehicles as light lines
    for sv in surrounding[:8]:  # cap for clarity
        ax_traj.plot(sv["y_m"], sv["x_m"], color="gray", alpha=0.3, linewidth=1)

    _draw_lane_boundaries(ax_traj, min(ego_y), max(ego_y))
    ax_traj.set_xlabel("Longitudinal position (m)")
    ax_traj.set_ylabel("Lateral position (m)")
    ax_traj.set_title("Ego XY Trajectory During Merge")
    ax_traj.grid(True, alpha=0.2)

    # Lane ID over time
    ax_lane.step(time_s, ego_lane, color="#4CAF50", linewidth=2, where="post")
    ax_lane.set_yticks([1, 2, 3, 4, 5, 6])
    ax_lane.set_yticklabels(["L1", "L2", "L3", "L4", "L5", "Ramp"])
    ax_lane.axhline(y=5.5, color="red", linestyle=":", linewidth=1,
                    label="Ramp → Mainline boundary")
    ax_lane.set_ylabel("Lane ID")
    ax_lane.set_xlabel("Time (s)")
    ax_lane.set_title("Lane Transition")
    ax_lane.legend(fontsize=8)
    ax_lane.grid(True, alpha=0.3)


# ── Lane Cut-In Visualization ─────────────────────────────────────────────────

def _plot_lane_cutin(df: pd.DataFrame, sample: dict, ax_list):
    ego_traj    = _parse_json_col(sample["ego_trajectory"])
    surrounding = _parse_json_col(sample["surrounding_vehicles"])

    ax_pos, ax_speed = ax_list

    ego_y    = [r["y_m"]     for r in ego_traj]
    ego_x    = [r["x_m"]     for r in ego_traj]
    ego_spd  = [r["speed_ms"]for r in ego_traj]
    frames   = [r["frame"]   for r in ego_traj]
    time_s   = [(f - frames[0]) / 10.0 for f in frames]

    cutin_frame = sample.get("cutin_frame", (frames[0] + frames[-1]) // 2)
    cutin_t     = (cutin_frame - frames[0]) / 10.0

    cutter_id   = sample.get("cutter_id")

    # Position plot
    ax_pos.plot(ego_y, ego_x, color="#F44336", linewidth=2, label="Ego (follower)")
    for sv in surrounding:
        if sv["vehicle_id"] == cutter_id:
            ax_pos.plot(sv["y_m"], sv["x_m"], color="#FF5722", linewidth=2,
                        linestyle="--", label=f"Cutter (ID {cutter_id})")
        else:
            ax_pos.plot(sv["y_m"], sv["x_m"], color="gray", alpha=0.2, linewidth=1)

    _draw_lane_boundaries(ax_pos, min(ego_y), max(ego_y))
    ax_pos.set_xlabel("Longitudinal position (m)")
    ax_pos.set_ylabel("Lateral position (m)")
    ax_pos.set_title("Positions During Cut-In")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.2)

    # Speed plot with cut-in marker
    ax_speed.plot(time_s, ego_spd, color="#F44336", linewidth=2, label="Ego speed")
    ax_speed.axvline(x=cutin_t, color="black", linestyle="--", linewidth=1.5,
                     label=f"Cut-in moment ({cutin_t:.1f}s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_title("Ego Speed Response to Cut-In")
    ax_speed.legend(fontsize=8)
    ax_speed.grid(True, alpha=0.3)


# ── Public API ────────────────────────────────────────────────────────────────

def visualize_scenario(
    df: pd.DataFrame,
    sample: dict,
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Generate a 2-panel diagnostic figure for one scenario sample.

    Parameters
    ----------
    df         : clean preprocessed DataFrame
    sample     : one row from the windowed samples DataFrame (as dict)
    save_path  : file path to save PNG (auto-generated if None)
    show       : call plt.show() — useful for Jupyter

    Returns
    -------
    Path to the saved PNG file.
    """
    sc_type = sample["scenario_type"]
    ego_id  = sample["ego_id"]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        f"{SCENARIO_TITLES.get(sc_type, sc_type)}  |  Ego Vehicle {ego_id}  "
        f"|  Frames {sample['window_start_frame']}–{sample['window_end_frame']}",
        fontsize=13, fontweight="bold", color=LANE_COLOURS.get(sc_type, "black"),
    )
    gs   = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])

    if sc_type == "car_following":
        _plot_car_following(df, sample, [ax1, ax2])
    elif sc_type == "onramp_merge":
        _plot_onramp_merge(df, sample, [ax1, ax2])
    elif sc_type == "lane_cutin":
        _plot_lane_cutin(df, sample, [ax1, ax2])
    else:
        ax1.text(0.5, 0.5, "Unknown scenario type", ha="center")

    if save_path is None:
        os.makedirs("output/plots", exist_ok=True)
        save_path = f"output/plots/{sc_type}_ego{ego_id}.png"

    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {save_path}")

    if show:
        plt.show()

    return save_path


def visualize_all(
    df: pd.DataFrame,
    samples: pd.DataFrame,
    output_dir: str = "output/plots",
    max_per_type: int = 3,
) -> list:
    """
    Generate plots for up to max_per_type examples of each scenario type.
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for sc_type in ["car_following", "onramp_merge", "lane_cutin"]:
        subset = samples[samples["scenario_type"] == sc_type].head(max_per_type)
        for i, (_, row) in enumerate(subset.iterrows()):
            save_path = os.path.join(output_dir, f"{sc_type}_{i+1}.png")
            try:
                path = visualize_scenario(df, row.to_dict(), save_path=save_path)
                paths.append(path)
            except Exception as e:
                logger.warning(f"Visualization failed for {sc_type} row {i}: {e}")

    return paths


def plot_summary_dashboard(samples: pd.DataFrame, save_path: str = "output/plots/summary.png") -> str:
    """
    Overview dashboard: scenario counts, ego speed distributions, window durations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Scenario Extraction Summary", fontsize=14, fontweight="bold")

    # 1. Count bar chart
    counts = samples["scenario_type"].value_counts()
    colours = [LANE_COLOURS.get(s, "steelblue") for s in counts.index]
    axes[0].bar(counts.index, counts.values, color=colours, edgecolor="white")
    axes[0].set_title("Event Counts by Scenario")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=15)
    for bar, v in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(v), ha="center", fontsize=9)

    # 2. Window duration distribution
    if "window_start_frame" in samples and "window_end_frame" in samples:
        durations = (samples["window_end_frame"] - samples["window_start_frame"]) / 10.0
        axes[1].hist(durations, bins=20, color="#607D8B", edgecolor="white")
        axes[1].set_title("Window Duration Distribution")
        axes[1].set_xlabel("Duration (s)")
        axes[1].set_ylabel("Count")
        axes[1].axvline(x=5, color="red", linestyle="--", linewidth=1, label="5 s target")
        axes[1].legend(fontsize=8)

    # 3. Surrounding vehicle count
    if "num_surrounding" in samples:
        axes[2].hist(samples["num_surrounding"].dropna().astype(int),
                     bins=range(0, 20), color="#9C27B0", edgecolor="white", align="left")
        axes[2].set_title("Surrounding Vehicle Count per Window")
        axes[2].set_xlabel("Count")
        axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Summary dashboard saved: {save_path}")
    return save_path