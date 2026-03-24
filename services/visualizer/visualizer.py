import logging
import os
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

LANE_X_CENTRES = {1: 1.85, 2: 5.55, 3: 9.25, 4: 12.95, 5: 16.65, 6: 20.5}
LANE_BOUNDARIES = [0, 3.7, 7.4, 11.1, 14.8, 18.5, 22.5]
CAR_LENGTH_M = 4.5
CAR_WIDTH_M = 2.0
EGO_COLOUR = "#E53935"
LEADER_COLOUR = "#1E88E5"
CUTTER_COLOUR = "#F57C00"
OTHER_COLOUR = "#546E7A"
ROAD_COLOUR = "#0d1117"
SCENARIO_TITLES = {
    "car_following": "Scenario 1 — Car Following",
    "onramp_merge": "Scenario 2 — On-Ramp Merge",
    "lane_cutin": "Scenario 3 — Lane Cut-In",
}
SCENE_LABELS = {
    "car_following": "Ego follows Leader in the same lane",
    "onramp_merge": "Ego merges from ramp lane → mainline",
    "lane_cutin": "Cutter cuts in front of Ego",
}

def _parse_json_col(val):
    if isinstance(val, str): return json.loads(val)
    return val if val is not None else []

def _draw_road(ax, y_min, y_max, padding=15):
    ax.set_facecolor(ROAD_COLOUR)
    ax.set_xlim(y_min - padding, y_max + padding)
    ax.set_ylim(-1.5, max(LANE_BOUNDARIES) + 1.5)
    for i in range(len(LANE_BOUNDARIES) - 1):
        shade = "#1a2535" if i % 2 == 0 else "#1e2a3a"
        ax.axhspan(LANE_BOUNDARIES[i], LANE_BOUNDARIES[i+1], color=shade, zorder=0)
    for i, x in enumerate(LANE_BOUNDARIES):
        style = "-" if i in (0, len(LANE_BOUNDARIES)-1) else "--"
        lw = 2.0 if i in (0, len(LANE_BOUNDARIES)-1) else 0.8
        ax.axhline(y=x, color="white", linestyle=style, linewidth=lw, alpha=0.5, zorder=1)
    for lane_id, cx in LANE_X_CENTRES.items():
        label = "Ramp" if lane_id == 6 else f"Lane {lane_id}"
        ax.text(y_min - padding + 1, cx, label, color="white", fontsize=6.5, alpha=0.45, va="center", ha="left", zorder=2)
    ax.set_xlabel("Longitudinal Position (m)", color="white", fontsize=9)
    ax.set_ylabel("Lateral Position (m)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2a2a")

def _draw_car(ax, y_c, x_c, colour, label, alpha=1.0, zorder=5):
    rect = mpatches.FancyBboxPatch((y_c - CAR_LENGTH_M/2, x_c - CAR_WIDTH_M/2), CAR_LENGTH_M, CAR_WIDTH_M, boxstyle="round,pad=0.1", linewidth=1.5, edgecolor="white", facecolor=colour, alpha=alpha, zorder=zorder)
    ax.add_patch(rect)
    ax.text(y_c, x_c, label, color="white", fontsize=6, fontweight="bold", ha="center", va="center", zorder=zorder + 1, path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

def _draw_trail(ax, y_pos, x_pos, colour, zorder=3):
    if len(y_pos) < 2: return
    ax.plot(y_pos, x_pos, color=colour, linestyle=":", linewidth=1.8, alpha=0.5, zorder=zorder)
    ax.scatter(y_pos[:-1], x_pos[:-1], color=colour, s=7, alpha=0.35, zorder=zorder)

def _draw_arrow(ax, y_pos, x_pos, colour, zorder=6):
    if len(y_pos) < 2: return
    dy = y_pos[-1] - y_pos[-2]; dx = x_pos[-1] - x_pos[-2]
    norm = np.sqrt(dy**2 + dx**2)
    if norm < 0.01: return
    scale = 3.0
    tip_y = y_pos[-1] + (dy / norm) * scale; tip_x = x_pos[-1] + (dx / norm) * scale
    ax.annotate("", xy=(tip_y, tip_x), xytext=(y_pos[-1], x_pos[-1]), arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8, mutation_scale=12), zorder=zorder)

def _subsample(lst, n=15):
    if len(lst) <= n: return lst
    idx = np.linspace(0, len(lst)-1, n, dtype=int)
    return [lst[i] for i in idx]

def _render_scene(ax, ego_traj, surrounding, highlight_ids, id_colour_map):
    ego_y = [r["y_m"] for r in ego_traj]; ego_x = [r["x_m"] for r in ego_traj]
    ego_spd = [r["speed_ms"] for r in ego_traj]; ego_frm = [r["frame"] for r in ego_traj]
    all_y = ego_y[:]
    for sv in surrounding: all_y.extend(sv.get("y_m", []))
    y_min = min(all_y) if all_y else 0; y_max = max(all_y) if all_y else 100
    _draw_road(ax, y_min, y_max)
    _draw_trail(ax, _subsample(ego_y, 15), _subsample(ego_x, 15), EGO_COLOUR)
    _draw_arrow(ax, ego_y, ego_x, EGO_COLOUR)
    _draw_car(ax, ego_y[-1], ego_x[-1], EGO_COLOUR, "Ego", zorder=8)
    car_counter = 1
    for sv in surrounding:
        sv_id = sv["vehicle_id"]; sv_y = sv.get("y_m", []); sv_x = sv.get("x_m", [])
        if not sv_y: continue
        colour = id_colour_map.get(sv_id, OTHER_COLOUR)
        label = highlight_ids.get(sv_id, f"Car {car_counter}")
        is_key = sv_id in highlight_ids
        _draw_trail(ax, _subsample(sv_y, 15), _subsample(sv_x, 15), colour)
        _draw_arrow(ax, sv_y, sv_x, colour)
        _draw_car(ax, sv_y[-1], sv_x[-1], colour, label, alpha=1.0 if is_key else 0.5, zorder=7 if is_key else 4)
        if not is_key: car_counter += 1
    return ego_y, ego_x, ego_spd, ego_frm

def _style_panel(ax):
    ax.set_facecolor("#111827"); ax.tick_params(colors="white", labelsize=7)
    ax.grid(True, alpha=0.12, color="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2a2a")

def _panel_car_following(ax, ego_spd, ego_frm, surrounding, leader_id, sample):
    _style_panel(ax); t = [(f - ego_frm[0]) / 10.0 for f in ego_frm]
    ax.plot(t, ego_spd, color=EGO_COLOUR, linewidth=2, label="Ego")
    for sv in surrounding:
        if sv["vehicle_id"] == leader_id:
            lspd = sv.get("speed_ms", []); lfrm = sv.get("frames", ego_frm)
            ax.plot([(f - ego_frm[0]) / 10.0 for f in lfrm], lspd, color=LEADER_COLOUR, linewidth=2, linestyle="--", label="Leader"); break
    gap = sample.get("avg_gap_m"); info = f"Avg gap: {gap:.1f} m" if gap else ""
    ax.set_title(f"Speed over Time\n{info}", color="white", fontsize=8)
    ax.set_xlabel("Time (s)", color="white", fontsize=8); ax.set_ylabel("Speed (m/s)", color="white", fontsize=8)
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

def _panel_onramp_merge(ax, ego_traj, ego_frm, sample):
    _style_panel(ax); t = [(f - ego_frm[0]) / 10.0 for f in ego_frm]
    lanes = [r["Lane_ID"] for r in ego_traj]
    ax.step(t, lanes, color="#4CAF50", linewidth=2, where="post")
    ax.fill_between(t, lanes, alpha=0.12, color="#4CAF50", step="post")
    ax.axhline(y=5.5, color="red", linestyle=":", linewidth=1, alpha=0.8, label="Ramp → Mainline")
    ax.set_yticks([1, 2, 3, 4, 5, 6]); ax.set_yticklabels(["L1", "L2", "L3", "L4", "L5", "Ramp"], color="white", fontsize=7)
    ax.set_title("Lane Transition", color="white", fontsize=8)
    ax.set_xlabel("Time (s)", color="white", fontsize=8); ax.set_ylabel("Lane", color="white", fontsize=8)
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

def _panel_cutin(ax, ego_spd, ego_frm, sample):
    _style_panel(ax); t = [(f - ego_frm[0]) / 10.0 for f in ego_frm]
    cutin_frame = sample.get("cutin_frame"); cutin_t = (cutin_frame - ego_frm[0]) / 10.0 if cutin_frame else None
    ax.plot(t, ego_spd, color=EGO_COLOUR, linewidth=2, label="Ego speed")
    if cutin_t is not None:
        ax.axvline(x=cutin_t, color=CUTTER_COLOUR, linestyle="--", linewidth=1.8, label=f"Cut-in at {cutin_t:.1f}s")
        if t: ax.axvspan(cutin_t, max(t), alpha=0.07, color=CUTTER_COLOUR)
    drop = sample.get("speed_drop_ms"); info = f"Speed drop: {drop:.2f} m/s" if drop else ""
    ax.set_title(f"Ego Speed Response\n{info}", color="white", fontsize=8)
    ax.set_xlabel("Time (s)", color="white", fontsize=8); ax.set_ylabel("Speed (m/s)", color="white", fontsize=8)
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

def visualize_scenario(df, sample, save_path=None, show=False):
    sc_type = sample["scenario_type"]; ego_id = int(sample["ego_id"])
    ego_traj = _parse_json_col(sample["ego_trajectory"]); surrounding = _parse_json_col(sample["surrounding_vehicles"])
    if not ego_traj: return None
    highlight_ids = {}; id_colour_map = {}
    if sc_type == "car_following" and sample.get("leader_id"): highlight_ids[int(sample["leader_id"])] = "Leader"; id_colour_map[int(sample["leader_id"])] = LEADER_COLOUR
    elif sc_type == "lane_cutin" and sample.get("cutter_id"): highlight_ids[int(sample["cutter_id"])] = "Cutter"; id_colour_map[int(sample["cutter_id"])] = CUTTER_COLOUR
    for sv in surrounding:
        if sv["vehicle_id"] not in id_colour_map: id_colour_map[sv["vehicle_id"]] = OTHER_COLOUR
    fig = plt.figure(figsize=(16, 6), facecolor=ROAD_COLOUR)
    fig.suptitle(f"{SCENARIO_TITLES.get(sc_type, sc_type)}   ·   Ego Vehicle {ego_id}   ·   Frames {sample['window_start_frame']}–{sample['window_end_frame']}", color="white", fontsize=11, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28, width_ratios=[2.2, 1])
    ax_scene = fig.add_subplot(gs[0, 0]); ax_panel = fig.add_subplot(gs[0, 1])
    ego_y, ego_x, ego_spd, ego_frm = _render_scene(ax_scene, ego_traj, surrounding, highlight_ids, id_colour_map)
    ax_scene.set_title(SCENE_LABELS.get(sc_type, ""), color="white", fontsize=9, pad=6)
    patches = [mpatches.Patch(color=EGO_COLOUR, label="Ego")]
    if sc_type == "car_following": patches.append(mpatches.Patch(color=LEADER_COLOUR, label="Leader"))
    elif sc_type == "lane_cutin": patches.append(mpatches.Patch(color=CUTTER_COLOUR, label="Cutter"))
    patches.append(mpatches.Patch(color=OTHER_COLOUR, label="Other vehicles"))
    ax_scene.legend(handles=patches, loc="lower right", fontsize=7, facecolor="#1a1a2e", labelcolor="white", framealpha=0.85)
    if sc_type == "car_following": _panel_car_following(ax_panel, ego_spd, ego_frm, surrounding, sample.get("leader_id"), sample)
    elif sc_type == "onramp_merge": _panel_onramp_merge(ax_panel, ego_traj, ego_frm, sample)
    elif sc_type == "lane_cutin": _panel_cutin(ax_panel, ego_spd, ego_frm, sample)
    if save_path is None: os.makedirs("output/plots", exist_ok=True); save_path = f"output/plots/{sc_type}_ego{ego_id}.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)
    logger.info(f"Saved plot: {save_path}"); return save_path

def visualize_all(df, samples, output_dir="output/plots", max_per_type=3):
    os.makedirs(output_dir, exist_ok=True); paths = []
    for sc_type in ["car_following", "onramp_merge", "lane_cutin"]:
        subset = samples[samples["scenario_type"] == sc_type].head(max_per_type)
        for i, (_, row) in enumerate(subset.iterrows()):
            sp = os.path.join(output_dir, f"{sc_type}_{i+1}.png")
            try:
                p = visualize_scenario(df, row.to_dict(), save_path=sp)
                if p: paths.append(p)
            except Exception as e: logger.warning(f"Viz failed for {sc_type} #{i}: {e}")
    return paths

def plot_summary_dashboard(samples, save_path="output/plots/summary.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=ROAD_COLOUR)
    fig.suptitle("Scenario Extraction Summary", color="white", fontsize=13, fontweight="bold")
    COLOURS = {"car_following": EGO_COLOUR, "onramp_merge": "#4CAF50", "lane_cutin": CUTTER_COLOUR}
    for ax in axes: ax.set_facecolor("#111827"); ax.tick_params(colors="white", labelsize=8); [s.set_edgecolor("#2a2a2a") for s in ax.spines.values()]
    counts = samples["scenario_type"].value_counts(); colours = [COLOURS.get(s, "steelblue") for s in counts.index]
    bars = axes[0].bar(counts.index, counts.values, color=colours, edgecolor=ROAD_COLOUR)
    axes[0].set_title("Event Counts by Scenario", color="white", fontsize=10); axes[0].set_ylabel("Count", color="white"); axes[0].tick_params(axis="x", rotation=12)
    for bar, v in zip(bars, counts.values): axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4, str(v), ha="center", color="white", fontsize=9)
    if "window_start_frame" in samples and "window_end_frame" in samples:
        dur = (samples["window_end_frame"] - samples["window_start_frame"]) / 10.0
        axes[1].hist(dur, bins=20, color="#607D8B", edgecolor=ROAD_COLOUR); axes[1].axvline(x=5, color="red", linestyle="--", linewidth=1, label="5 s target")
        axes[1].set_title("Window Duration", color="white", fontsize=10); axes[1].set_xlabel("Duration (s)", color="white"); axes[1].set_ylabel("Count", color="white")
        axes[1].legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    if "num_surrounding" in samples:
        axes[2].hist(samples["num_surrounding"].dropna().astype(int), bins=range(0, 20), color="#9C27B0", edgecolor=ROAD_COLOUR, align="left")
        axes[2].set_title("Surrounding Vehicles per Window", color="white", fontsize=10); axes[2].set_xlabel("Count", color="white"); axes[2].set_ylabel("Frequency", color="white")
    plt.tight_layout(); fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)
    logger.info(f"Summary dashboard saved: {save_path}"); return save_path
