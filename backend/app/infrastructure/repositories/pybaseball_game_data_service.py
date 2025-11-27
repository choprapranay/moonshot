import pandas as pd
from pybaseball import statcast_single_game

from app.domain.interfaces import GameDataServiceInterface


class PybaseballGameDataService(GameDataServiceInterface):
    def get_game_data(self, game_pk: int):
        return statcast_single_game(game_pk)

