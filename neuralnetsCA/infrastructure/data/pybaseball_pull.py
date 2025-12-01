import pandas as pd
from pybaseball import statcast

from domain.interfaces import DataAccessInterface


class PybaseballDataAccess(DataAccessInterface):
    def __init__(self):
        self._source_name = "pybaseball"
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            data = statcast(start_date, end_date)
            return data
        except Exception as e:
            print(f"Error fetching data from pybaseball: {e}")
            return pd.DataFrame()
    
    def get_source_name(self) -> str:
        return self._source_name