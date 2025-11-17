from typing import Optional
import pandas as pd
from pybaseball import statcast

from domain.interfaces import PitchEventRepositoryInterface
class PybaseballPitchEventRepository(PitchEventRepositoryInterface):
    def __init__(self, default_start: Optional[str] = None, default_end: Optional[str] = None):
        self.default_start = default_start
        self.default_end = default_end

    def fetch_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = start_date or self.default_start
        end = end_date or self.default_end
        df = statcast(start, end)
        return df