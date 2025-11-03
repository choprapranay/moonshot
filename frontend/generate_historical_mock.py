import json, random

# --- Make results reproducible ---
random.seed(42)

data = []
# Generate 200 random points in strike zone (-1.0 to 1.0 in x, 0 to 2.0 in y)
for _ in range(200):
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(0.0, 2.0)
    # Hotter near center of strike zone
    intensity = max(0, min(1, random.gauss(0.5 + 0.4 * (1 - abs(x)), 0.25)))
    data.append({"x": x, "y": y, "intensity": round(intensity, 2)})

mock = {
    "batter": "J. Doe",
    "zone": {"x_range": [-1.0, 1.0], "y_range": [0.0, 2.0]},
    "data": data
}

with open("mock-data/historical_heatmap.json", "w") as f:
    json.dump(mock, f, indent=2)
