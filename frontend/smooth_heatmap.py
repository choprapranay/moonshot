import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# --- Load JSON data ---
with open("mock-data/historical_heatmap.json") as f:
    hist = json.load(f)
with open("mock-data/game_pitches.json") as f:
    game = json.load(f)

# --- Extract data ---
x = [d["x"] for d in hist["data"]]
y = [d["y"] for d in hist["data"]]
weights = [d["intensity"] for d in hist["data"]]

pitch_x = [p["x"] for p in game["pitch_data"]]
pitch_y = [p["y"] for p in game["pitch_data"]]

# --- Create plot ---
plt.figure(figsize=(6, 8))
ax = plt.gca()

# Dark background for contrast
ax.set_facecolor("#101020")

# Shifted colormap to bias away from white
cmap = mcolors.LinearSegmentedColormap.from_list(
    "shifted_rdBu",
    [(0, "darkblue"), (0.4, "blue"), (0.5, "lightblue"), (0.6, "salmon"), (1, "darkred")]
)

# KDE-based heatmap (smooth red-blue)
sns.kdeplot(
    x=x,
    y=y,
    weights=weights,
    fill=True,
    cmap=cmap,
    bw_adjust=0.6,
    alpha=0.8,
    levels=100,
    thresh=0.05
)

# Overlay pitch dots
plt.scatter(
    pitch_x,
    pitch_y,
    s=50,
    edgecolors="white",
    facecolors="black",
    linewidths=1.2,
    label="Pitches"
)

# Strike zone box
strike_zone = plt.Rectangle(
    (-0.83, 0.5),  # bottom-left corner (x, y)
    1.66,          # width
    1.5,           # height
    linewidth=1.5,
    edgecolor="white",
    facecolor="none"
)
ax.add_patch(strike_zone)

# --- Styling ---
plt.text(-0.7, 1.9, f"Hitter: {hist['batter']}", color="white", fontsize=12, fontweight="bold")
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
plt.legend(facecolor="white", loc="upper right")
plt.title("", color="white")
plt.tight_layout()
plt.show()