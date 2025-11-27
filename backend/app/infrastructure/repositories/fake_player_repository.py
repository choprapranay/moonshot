from typing import Any, Dict, Iterable, List

from app.domain.interfaces import PlayerRepositoryInterface


class FakePlayerRepository(PlayerRepositoryInterface):
    def __init__(self):
        self.players = {}

    def upsert_players(self, payloads: List[Dict[str, Any]]) -> None:
        for payload in payloads:
            name = payload["name"]
            self.players[name] = {
                "id": payload.get("id", f"player_{name}"),
                "name": name,
                "diagram_index": payload.get("diagram_index"),
            }

    def get_player_id_map_by_names(self, names: Iterable[str]) -> Dict[str, str]:
        return {name: self.players[name]["id"] for name in names if name in self.players}

