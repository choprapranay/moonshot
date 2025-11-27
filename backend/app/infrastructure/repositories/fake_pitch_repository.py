from typing import Any, Dict, List

from app.domain.interfaces import PitchRepositoryInterface


class FakePitchRepository(PitchRepositoryInterface):
    def __init__(self):
        self.pitches = {}

    def get_pitches_with_player_by_game_id(self, game_id: str) -> List[Dict[str, Any]]:
        return self.pitches.get(game_id, [])

    def upsert_pitches(self, payloads: List[Dict[str, Any]]) -> None:
        for payload in payloads:
            game_id = payload["game_id"]
            if game_id not in self.pitches:
                self.pitches[game_id] = []
            pitch_data = {
                "pitch_type": payload.get("pitch_type", ""),
                "speed": payload.get("speed"),
                "description": payload.get("description"),
                "players": {
                    "name": None,
                    "diagram_index": None,
                }
            }
            self.pitches[game_id].append(pitch_data)

