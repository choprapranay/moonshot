import pandas as pd

from app.domain.interfaces import GameDataServiceInterface


class FakeGameDataService(GameDataServiceInterface):
    def __init__(self):
        self.game_data = {}

    def set_game_data(self, game_pk: int, data: pd.DataFrame):
        self.game_data[game_pk] = data

    def get_game_data(self, game_pk: int):
        return self.game_data.get(game_pk, pd.DataFrame())

