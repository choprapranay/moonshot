from pybaseball import statcast
import pandas as pd

COLUMNS = ['game_date', 'batter', 'pitch_type', 'description', 'plate_x', 'plate_z', 'events',
           'release_speed', 'release_pos_x', 'launch_speed', 'launch_angle',
           'effective_speed', 'release_spin_rate', 'release_extension',
           'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0',
           'ax', 'ay', 'az', 'sz_top', 'sz_bot',
           'estimated_ba_using_speedangle', 'launch_speed_angle']


class PitchDataService:

    @staticmethod
    def get_pitch_data(start_date: str, end_date: str) -> pd.DataFrame:
        raw = statcast(start_dt=start_date, end_dt=end_date)
        return raw[COLUMNS].copy()