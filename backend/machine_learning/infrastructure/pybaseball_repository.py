import pandas as pd
from pybaseball import statcast

class PyBaseballRepository: 
    
    def fetch_pitch_data(self, start_date: str, end_date: str) -> pd.DataFrame: 
        return statcast(start_date, end_date)
    
