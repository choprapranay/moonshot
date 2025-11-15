import json, random

# --- Make results reproducible ---
random.seed(99)

pitch_types = ["Fastball", "Slider", "Curveball", "Changeup"]
results = ["Strike", "Ball", "Foul", "Hit", "Swinging Strike"]

pitches = []
for _ in range(50):
    x = random.uniform(-1.2, 1.2)
    y = random.uniform(0.0, 2.2)
    pitch_type = random.choice(pitch_types)
    result = random.choice(results)
    pitches.append({
        "x": round(x, 2),
        "y": round(y, 2),
        "pitch_type": pitch_type,
        "result": result
    })

mock = {
    "game_id": "2025-07-12-BlueJays-Yankees",
    "pitch_data": pitches
}

with open("mock-data/game_pitches.json", "w") as f:
    json.dump(mock, f, indent=2)
