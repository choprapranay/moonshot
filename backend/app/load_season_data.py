from datetime import datetime, timedelta
from .load_pitch_data import load_pitch_data

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

    load_range("2023-03-30", "2023-10-01")

    load_range("2024-03-28", "2024-09-29")

    print("All seasons loaded successfully")