import pandas as pd
import json
from machine_learning.infrastructure.pybaseball_repository import PyBaseballRepository
from machine_learning.domain.run_value_calculation import RunValueCalculator


def load_season_safe(repo: PyBaseballRepository, year) -> pd.DataFrame:
    months = [
        ('03-30', '04-30'),
        ('05-01', '05-31'),
        ('06-01', '06-30'),
        ('07-01', '07-31'),
        ('08-01', '08-31'),
        ('09-01', '10-01')
    ]
    
    data_chunks = []
    for start_month, end_month in months:
        try:
            chunk = repo.fetch_pitch_data(
                start_dt=f"{year}-{start_month}", 
                end_dt=f"{year}-{end_month}"
            )
            print(f"Loaded {year}-{start_month} to {year}-{end_month}: {len(chunk)} rows")
            data_chunks.append(chunk)
        except Exception as e:
            print(f"Error loading {year}-{start_month}: {e}")
            continue
    
    return pd.concat(data_chunks, ignore_index=True)

def main(): 

    repo = PyBaseballRepository()

    data_2023 = load_season_safe(repo, 2023)
    data_2024 = load_season_safe(repo, 2024)

    data = pd.concat([data_2023, data_2024], ignore_index=True)

    calculation = RunValueCalculator()

    run_values = calculation.calculate(data)

    print(run_values)

    with open('run_values.json', 'w') as f:
        json.dump(run_values, f, indent=2)

if __name__ == "__main__": # pragma: no cover
    main()

    