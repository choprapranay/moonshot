from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterable


class GameRepositoryInterface(ABC):
    @abstractmethod
    def get_game_by_pk(self, game_pk: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def upsert_game(self, payload: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_game_id_by_pk(self, game_pk: int) -> str:
        pass


class PitchRepositoryInterface(ABC):
    @abstractmethod
    def get_pitches_with_player_by_game_id(self, game_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def upsert_pitches(self, payloads: List[Dict[str, Any]]) -> None:
        pass


class PlayerRepositoryInterface(ABC):
    @abstractmethod
    def upsert_players(self, payloads: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def get_player_id_map_by_names(self, names: Iterable[str]) -> Dict[str, str]:
        pass


class GameDataServiceInterface(ABC):
    @abstractmethod
    def get_game_data(self, game_pk: int):
        pass

