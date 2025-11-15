import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Load Data ----------
with open("mock-data/historical_heatmap.json", "r") as f:
    historical_data = json.load(f)

with open("mock-data/game_pitches.json", "r") as f:
    game_data = json.load(f)

# ---------- Extract Historical Heatmap ----------
x_vals = [d["x"] for d in historical_data["data"]]
y_vals = [d["y"] for d in historical_data["data"]]
intensity_vals = [d["intensity"] for d in historical_data["data"]]

# Create a grid for heatmap interpolation
x_unique = np.linspace(min(x_vals), max(x_vals), 50)
y_unique = np.linspace(min(y_vals), max(y_vals), 50)
X, Y = np.meshgrid(x_unique, y_unique)

# Simple interpolation (nearest neighbour style)
Z = np.zeros_like(X)
for i in range(len(x_vals)):
    xi = np.argmin(np.abs(x_unique - x_vals[i]))
    yi = np.argmin(np.abs(y_unique - y_vals[i]))
    Z[yi, xi] = intensity_vals[i]

# ---------- Extract Pitch Locations ----------
pitch_x = [p["x"] for p in game_data["pitch_data"]]
pitch_y = [p["y"] for p in game_data["pitch_data"]]
pitch_types = [p["pitch_type"] for p in game_data["pitch_data"]]

# ---------- Plot ----------
plt.figure(figsize=(6, 8))

# Draw heatmap
sns.heatmap(
    Z,
    xticklabels=False,
    yticklabels=False,
    cmap="RdBu_r",
    cbar_kws={'label': 'Batter Performance Intensity'}
)

# Overlay scatter points for pitches
plt.scatter(
    np.interp(pitch_x, (min(x_vals), max(x_vals)), (0, Z.shape[1])),
    np.interp(pitch_y, (min(y_vals), max(y_vals)), (0, Z.shape[0])),
    color='black',
    s=60,
    label="Pitch Locations"
)

plt.title(f"Batter: {historical_data['batter']} — Game Pitch Overlay")
plt.xlabel("Horizontal Pitch Location")
plt.ylabel("Vertical Pitch Location")
plt.legend(loc="upper right")
plt.show()
