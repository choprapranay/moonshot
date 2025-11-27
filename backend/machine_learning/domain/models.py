from dataclasses import dataclass

@dataclass
class Pitch:
    game_date: str
    batter: float
    pitch_type: str
    description: str
    plate_x: float
    plate_z: float
    events: str
    release_speed: float
    release_pos_x: float
    launch_speed: float
    launch_angle: float
    effective_speed: float
    release_spin_rate: float
    release_extension: float
    hc_x: float
    hc_y: float
    vx0: float
    vy0: float
    vz0: float
    ax: float
    ay: float
    az: float
    sz_top: float
    sz_bot: float
    estimated_ba_using_speedangle: float
    launch_speed_angle: float