from services.pitch_data_service import PitchDataService
from services.cleaning_service import CleaningService
from infrastructure.supabase_repository import SupabaseRepository

def load_pitch_data(start_date: str, end_date: str):

    raw = PitchDataService.get_pitch_data(start_date, end_date)
    clean = CleaningService.fix_na(raw)

    repo = SupabaseRepository()
    repo.insert_pitch_data_rows("statcast_raw", clean.to_dict(orient='records'))

    print(f"ETL COMPLETE — inserted {len(clean)} rows")


if __name__ == "__main__":
    # test run
    load_pitch_data("2025-10-14", "2025-10-15")