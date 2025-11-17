import sys
import os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(1, parent_dir)

from backend.services.pitch_data_service import PitchDataService
from backend.services.cleaning_service import CleaningService
from backend.infrastructure.supabase_repository import SupabaseRepository

BATCH_SIZE = 500
RETRY_LIMIT = 3

def batch_insert(repo, table, rows):
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]

        attempts = 0
        while attempts < RETRY_LIMIT:
            try:
                repo.insert_pitch_data_rows(table, batch)
                break
            except Exception as e:
                attempts += 1
                print(f"Batch insert failed (attempt {attempts}/{RETRY_LIMIT}): {e}")
                time.sleep(1.5)

        if attempts == RETRY_LIMIT:
            print("Failed to insert batch after 3 retries. Skipping batch.")

def load_pitch_data(start_date: str, end_date: str):


    print(f"\nExtracting {start_date} → {end_date} ...")

    raw = PitchDataService.get_pitch_data(start_date, end_date)
    clean = CleaningService.fix_na(raw)
    rows = clean.to_dict(orient='records')

    if not rows:
        print("No valid rows found after cleaning.")
        return

    repo = SupabaseRepository()
    print(f"Uploading {len(rows)} rows in batches of {BATCH_SIZE}...")

    batch_insert(repo, "pitch_data_raw", rows)

    print(f"COMPLETED — inserted ~{len(rows)} rows for {start_date}")


if __name__ == "__main__":
    # test load
    load_pitch_data("2025-10-14", "2025-10-15")
