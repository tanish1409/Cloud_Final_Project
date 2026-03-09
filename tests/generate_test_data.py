"""
Generate a synthetic NGSIM I-80-like CSV for pipeline testing.
Injects known instances of all 3 scenario types so we can verify detection.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
FRAME_RATE = 10   # Hz
FPS_TO_MPH = 0.681818

def make_base_row(vid, frame, lane, local_x, local_y, speed_fps, accel_fps2,
                  preceding=0, following=0, v_class=2):
    return {
        "Vehicle_ID":   vid,
        "Frame_ID":     frame,
        "Total_Frames": 500,
        "Global_Time":  1109230000 + frame * 100,   # ms
        "Local_X":      round(local_x, 3),
        "Local_Y":      round(local_y, 3),
        "Global_X":     round(6068600 + local_x, 3),
        "Global_Y":     round(2131000 + local_y, 3),
        "v_length":     14.5,
        "v_Width":      6.0,
        "v_Class":      v_class,
        "v_Vel":        round(speed_fps, 3),
        "v_Acc":        round(accel_fps2, 3),
        "Lane_ID":      lane,
        "Preceding":    preceding,
        "Following":    following,
        "Space_Headway": 0.0,
        "Time_Headway":  0.0,
    }


rows = []

# ── Scenario 1: Car-Following (vehicles 1 & 2 in lane 3) ─────────────────────
# Vehicle 2 leads, Vehicle 1 follows at ~40 m gap, correlated speeds
FRAMES = range(100, 400)
leader_y = 500.0  # feet
follower_y = 370.0  # ~40 m behind (40/0.3048 ≈ 131 ft)
speed_leader = 80.0  # fps ≈ 55 mph

for f in FRAMES:
    # Leader slightly oscillates speed
    spd = speed_leader + 5 * np.sin(f / 20.0)
    leader_y += spd / FRAME_RATE
    rows.append(make_base_row(2, f, 3, 42.0, leader_y, spd, 0.0))

    # Follower tracks leader with slight lag
    fspd = speed_leader + 4 * np.sin((f - 3) / 20.0)
    follower_y += fspd / FRAME_RATE
    gap_ft = leader_y - follower_y
    rows.append(make_base_row(1, f, 3, 42.0, follower_y, fspd, 0.0,
                              preceding=2, following=0))
    # Inject Space_Headway
    rows[-1]["Space_Headway"] = round(gap_ft, 2)

# ── Scenario 2: On-Ramp Merge (vehicle 3: ramp → lane 5) ─────────────────────
# Vehicle starts in lane 6 (ramp) and transitions to lane 5
merge_y = 200.0
lateral_x_ramp = 61.0   # feet, ramp X
lateral_x_main = 55.0   # feet, mainline lane 5 X

for f in range(200, 350):
    if f < 280:
        lane = 6
        local_x = lateral_x_ramp - (f - 200) * 0.05  # drifting towards mainline
    else:
        lane = 5
        local_x = lateral_x_main
    merge_y += 80 / FRAME_RATE
    rows.append(make_base_row(3, f, lane, local_x, merge_y, 80.0, 0.0))

# ── Scenario 3: Lane Cut-In (vehicle 5 cuts in front of vehicle 4) ───────────
# Vehicle 4 follows vehicle 6 in lane 2. Vehicle 5 cuts in from lane 3 → lane 2.
base_y = 600.0

# Vehicle 6 (original leader of vehicle 4)
for f in range(50, 350):
    base_y_ldr = 600.0 + (f - 50) * 80 / FRAME_RATE
    rows.append(make_base_row(6, f, 2, 28.0, base_y_ldr, 80.0, 0.0))

# Vehicle 4 (follower, lane 2)
for f in range(50, 350):
    y4 = 480.0 + (f - 50) * 79 / FRAME_RATE
    # Before cut-in (frame 200) follows vehicle 6
    # After cut-in (frame 200+) follows vehicle 5
    prec = 6 if f < 200 else 5
    spd = 79.0 if f < 205 else max(70.0, 79.0 - (f - 205) * 0.3)  # slow down after cutin
    rows.append(make_base_row(4, f, 2, 28.0, y4, spd, 0.0, preceding=prec))
    rows[-1]["Space_Headway"] = 35.0

# Vehicle 5 (cutter, lane 3 → lane 2 at frame 200)
for f in range(150, 300):
    if f < 200:
        lane = 3
        x5 = 36.0 - (f - 150) * 0.16  # lateral movement towards lane 2
    else:
        lane = 2
        x5 = 28.0
    y5 = 530.0 + (f - 150) * 82 / FRAME_RATE
    rows.append(make_base_row(5, f, lane, x5, y5, 82.0, 0.0))

# ── Background vehicles (random traffic) ─────────────────────────────────────
for vid in range(10, 40):
    lane = np.random.choice([1, 2, 3, 4, 5])
    start_y = np.random.uniform(50, 800)
    start_f = np.random.randint(1, 200)
    speed   = np.random.uniform(60, 100)
    x_pos   = [14.5, 14.5, 28.0, 36.0, 42.0, 55.0][lane]
    for f in range(start_f, min(start_f + np.random.randint(50, 200), 500)):
        y = start_y + (f - start_f) * speed / FRAME_RATE
        if y > 1650:
            break
        rows.append(make_base_row(vid, f, lane, x_pos,
                                  round(y, 2), speed,
                                  np.random.uniform(-2, 2)))

# ── Add some noise/dirt to test preprocessing ─────────────────────────────────
dirty_rows = rows.copy()

# Inject a few bad rows
dirty_rows.append({**rows[0], "v_Vel": "CORRUPT", "Lane_ID": 7})   # HOV lane + bad speed
dirty_rows.append({**rows[5], "Local_Y": -50.0})                    # out of segment
dirty_rows.append(rows[10].copy())                                   # duplicate

# Shuffle to simulate real messy CSV
import random
random.shuffle(dirty_rows)

df = pd.DataFrame(dirty_rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/i80_test.csv", index=False)
print(f"Synthetic dataset written: {len(df)} rows, {df['Vehicle_ID'].nunique()} vehicles")
print(f"Columns: {list(df.columns)}")