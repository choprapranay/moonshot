from datetime import datetime, timedelta
from app.load_pitch_data import load_pitch_data

# Load data for a season day by day
def load_range(start_date: str, end_date: str):
    current = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    while current <= end:
        day = current.strftime("%Y-%m-%d")
        print(f"Loading day: {day}")

        try:
            load_pitch_data(day, day)
        except Exception as e:
            print(f"Error on {day}: {e}")
        current += timedelta(days=1)


if __name__ == "__main__":
    print("Starting full season load")

    # 2023 Season
    load_range("2023-03-30", "2023-10-01")

    # 2024 Season
    load_range("2024-03-28", "2024-09-29")

    print("All seasons loaded successfully")