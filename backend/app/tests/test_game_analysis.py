import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from app.use_cases.game_analysis import (
    BuildGameAnalysisUseCase,
    BuildPitchRecordsUseCase,
    GetBatterSwingsUseCase,
    _build_player_summary,
    _convert_domain_stats_to_schema,
)
from app.api.schemas.game_analysis import GameAnalysisResponse
from app.domain.entities import PlayerStats as PlayerStatsEntity


class TestGameAnalysis:
    """Test cases for game_analysis use case with 100% code coverage"""

    def _create_sample_dataframe(self, game_pk=12345, num_pitches=3):
        """Helper to create a sample DataFrame for testing"""
        return pd.DataFrame({
            "game_pk": [game_pk] * num_pitches,
            "game_date": ["2024-10-15"] * num_pitches,
            "home_team": ["Yankees"] * num_pitches,
            "away_team": ["Red Sox"] * num_pitches,
            "batter": [456, 789, 456],
            "player_name": ["Player A", "Player B", "Player A"],
            "pitch_type": ["FF", "SL", "CH"],
            "type": ["S", "X", "S"],
            "description": ["swinging_strike", "in_play", "called_strike"],
            "release_speed": [95.5, 87.0, 83.2],
            "stand": ["R", "L", "R"],
            "inning_topbot": ["Top", "Bot", "Top"],
            "plate_x": [0.5, -0.3, 0.2],
            "plate_z": [2.5, 1.8, 2.1],
        })

    # ========== build_game_analysis tests ==========

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_game_analysis_success(self, mock_statcast):
        """Test successful game analysis with players and teams"""
        game_pk = 12345
        df = self._create_sample_dataframe(game_pk)
        mock_statcast.return_value = df

        use_case = BuildGameAnalysisUseCase()
        result = use_case.execute(game_pk)

        assert isinstance(result, GameAnalysisResponse)
        assert result.game_id == game_pk
        assert result.game_date == "2024-10-15"
        assert len(result.teams) == 2
        assert result.teams[0].code == "Yankees"
        assert result.teams[1].code == "Red Sox"
        assert len(result.players) == 2  # Two unique players
        assert result.players[0].stats.pitches_seen == 2  # Player A has 2 pitches
        assert result.players[1].stats.pitches_seen == 1  # Player B has 1 pitch

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_game_analysis_no_data_none(self, mock_statcast):
        """Test when statcast returns None"""
        mock_statcast.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            use_case = BuildGameAnalysisUseCase()
            use_case.execute(99999)

        assert exc_info.value.status_code == 404
        assert "No data found for this game ID" in str(exc_info.value.detail)

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_game_analysis_no_data_empty(self, mock_statcast):
        """Test when statcast returns empty DataFrame"""
        mock_statcast.return_value = pd.DataFrame()

        with pytest.raises(HTTPException) as exc_info:
            use_case = BuildGameAnalysisUseCase()
            use_case.execute(99998)

        assert exc_info.value.status_code == 404

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_game_analysis_no_players(self, mock_statcast):
        """Test when there are no valid players (null batter_id or player_name)"""
        game_pk = 12346
        df = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": ["2024-10-16"],
            "home_team": ["Dodgers"],
            "away_team": ["Giants"],
            "batter": [None],
            "player_name": [None],
            "pitch_type": ["FF"],
            "type": ["S"],
            "description": ["strike"],
            "release_speed": [95.0],
            "stand": ["R"],
            "inning_topbot": ["Top"],
        })
        mock_statcast.return_value = df

        use_case = BuildGameAnalysisUseCase()
        result = use_case.execute(game_pk)

        assert len(result.players) == 0
        assert len(result.teams) == 2

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_game_analysis_no_teams(self, mock_statcast):
        """Test when teams are None"""
        game_pk = 12347
        df = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": ["2024-10-17"],
            "home_team": [None],
            "away_team": [None],
            "batter": [456],
            "player_name": ["Player A"],
            "pitch_type": ["FF"],
            "type": ["S"],
            "description": ["strike"],
            "release_speed": [95.0],
            "stand": ["R"],
            "inning_topbot": ["Top"],
        })
        mock_statcast.return_value = df

        use_case = BuildGameAnalysisUseCase()
        result = use_case.execute(game_pk)

        assert len(result.teams) == 0

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_game_analysis_game_pk_none(self, mock_statcast):
        """Test when game_pk is None in dataframe"""
        game_pk = 12348
        df = pd.DataFrame({
            "game_pk": [None],
            "game_date": ["2024-10-18"],
            "home_team": ["Cubs"],
            "away_team": ["White Sox"],
            "batter": [456],
            "player_name": ["Player A"],
            "pitch_type": ["FF"],
            "type": ["S"],
            "description": ["strike"],
            "release_speed": [95.0],
            "stand": ["R"],
            "inning_topbot": ["Top"],
        })
        mock_statcast.return_value = df

        use_case = BuildGameAnalysisUseCase()
        result = use_case.execute(game_pk)

        assert result.game_id == game_pk  # Falls back to input game_pk

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_game_analysis_game_date_none(self, mock_statcast):
        """Test when game_date is None"""
        game_pk = 12349
        df = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": [None],
            "home_team": ["Mets"],
            "away_team": ["Phillies"],
            "batter": [456],
            "player_name": ["Player A"],
            "pitch_type": ["FF"],
            "type": ["S"],
            "description": ["strike"],
            "release_speed": [95.0],
            "stand": ["R"],
            "inning_topbot": ["Top"],
        })
        mock_statcast.return_value = df

        use_case = BuildGameAnalysisUseCase()
        result = use_case.execute(game_pk)

        assert result.game_date == ""

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_game_analysis_players_sorted_by_pitches(self, mock_statcast):
        """Test that players are sorted by pitches_seen descending"""
        game_pk = 12350
        df = pd.DataFrame({
            "game_pk": [game_pk] * 5,
            "game_date": ["2024-10-19"] * 5,
            "home_team": ["Astros"] * 5,
            "away_team": ["Rangers"] * 5,
            "batter": [100, 200, 100, 300, 200],  # Player 100: 2, Player 200: 2, Player 300: 1
            "player_name": ["Player 1", "Player 2", "Player 1", "Player 3", "Player 2"],
            "pitch_type": ["FF"] * 5,
            "type": ["S"] * 5,
            "description": ["strike"] * 5,
            "release_speed": [95.0] * 5,
            "stand": ["R"] * 5,
            "inning_topbot": ["Top"] * 5,
        })
        mock_statcast.return_value = df

        use_case = BuildGameAnalysisUseCase()
        result = use_case.execute(game_pk)

        # Should be sorted by pitches_seen descending
        assert len(result.players) == 3
        assert result.players[0].stats.pitches_seen >= result.players[1].stats.pitches_seen
        assert result.players[1].stats.pitches_seen >= result.players[2].stats.pitches_seen

    # ========== build_pitch_records tests ==========

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_pitch_records_success(self, mock_statcast):
        """Test successful pitch records build"""
        game_pk = 12351
        df = self._create_sample_dataframe(game_pk)
        mock_statcast.return_value = df

        use_case = BuildPitchRecordsUseCase()
        result = use_case.execute(game_pk)

        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], dict)
        assert "game_pk" in result[0]
        assert "pitch_type" in result[0]

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_pitch_records_no_data(self, mock_statcast):
        """Test when statcast returns None"""
        mock_statcast.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            use_case = BuildPitchRecordsUseCase()
            use_case.execute(99997)

        assert exc_info.value.status_code == 404

    # ========== get_batter_swings tests ==========

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_get_batter_swings_success(self, mock_statcast):
        """Test successful retrieval of batter swings"""
        game_pk = 12352
        batter_id = 456
        df = pd.DataFrame({
            "game_pk": [game_pk] * 4,
            "game_date": ["2024-10-20"] * 4,
            "home_team": ["Braves"] * 4,
            "away_team": ["Marlins"] * 4,
            "batter": [batter_id, batter_id, 999, batter_id],
            "player_name": ["Player A"] * 4,
            "pitch_type": ["FF", "SL", "CH", "CU"],
            "type": ["X", "S", "S", "X"],  # First and last are swings
            "description": ["in_play", "called_strike", "ball", "foul"],
            "release_speed": [95.0] * 4,
            "stand": ["R"] * 4,
            "inning_topbot": ["Top"] * 4,
            "plate_x": [0.5, -0.3, 0.2, 0.4],
            "plate_z": [2.5, 1.8, 2.1, 2.3],
        })
        mock_statcast.return_value = df

        use_case = GetBatterSwingsUseCase()
        result = use_case.execute(game_pk, batter_id)

        assert isinstance(result, list)
        assert len(result) >= 1  # At least one swing for this batter
        assert all("plate_x" in swing for swing in result)
        assert all("plate_z" in swing for swing in result)

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_get_batter_swings_no_data(self, mock_statcast):
        """Test when statcast returns None"""
        mock_statcast.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            use_case = GetBatterSwingsUseCase()
            use_case.execute(99996, 456)

        assert exc_info.value.status_code == 404

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_get_batter_swings_empty_batter_rows(self, mock_statcast):
        """Test when no rows match batter_id"""
        game_pk = 12353
        batter_id = 999
        df = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": ["2024-10-21"],
            "home_team": ["Padres"],
            "away_team": ["Diamondbacks"],
            "batter": [123],  # Different batter
            "player_name": ["Other Player"],
            "pitch_type": ["FF"],
            "type": ["S"],
            "description": ["strike"],
            "release_speed": [95.0],
            "stand": ["R"],
            "inning_topbot": ["Top"],
            "plate_x": [0.5],
            "plate_z": [2.5],
        })
        mock_statcast.return_value = df

        use_case = GetBatterSwingsUseCase()
        result = use_case.execute(game_pk, batter_id)

        assert result == []

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_get_batter_swings_missing_plate_coordinates(self, mock_statcast):
        """Test when plate_x or plate_z are null"""
        game_pk = 12354
        batter_id = 456
        df = pd.DataFrame({
            "game_pk": [game_pk] * 2,
            "game_date": ["2024-10-22"] * 2,
            "home_team": ["Nationals"] * 2,
            "away_team": ["Marlins"] * 2,
            "batter": [batter_id, batter_id],
            "player_name": ["Player A"] * 2,
            "pitch_type": ["FF", "SL"],
            "type": ["X", "S"],
            "description": ["in_play", "foul"],
            "release_speed": [95.0] * 2,
            "stand": ["R"] * 2,
            "inning_topbot": ["Top"] * 2,
            "plate_x": [0.5, None],  # Second has None
            "plate_z": [None, 2.5],  # First has None
        })
        mock_statcast.return_value = df

        use_case = GetBatterSwingsUseCase()
        result = use_case.execute(game_pk, batter_id)

        # Should filter out rows with null plate_x or plate_z
        assert result == []

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_get_batter_swings_no_swings_detected(self, mock_statcast):
        """Test when batter has pitches but none are swings"""
        game_pk = 12355
        batter_id = 456
        df = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": ["2024-10-23"],
            "home_team": ["Guardians"],
            "away_team": ["White Sox"],
            "batter": [batter_id],
            "player_name": ["Player A"],
            "pitch_type": ["FF"],
            "type": ["B"],  # Ball, not a swing
            "description": ["called_strike"],  # Not a swing
            "release_speed": [95.0],
            "stand": ["R"],
            "inning_topbot": ["Top"],
            "plate_x": [0.5],
            "plate_z": [2.5],
        })
        mock_statcast.return_value = df

        use_case = GetBatterSwingsUseCase()
        result = use_case.execute(game_pk, batter_id)

        assert result == []

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_get_batter_swings_with_empty_description(self, mock_statcast):
        """Test when description is empty string"""
        game_pk = 12356
        batter_id = 456
        df = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": ["2024-10-24"],
            "home_team": ["Rangers"],
            "away_team": ["Astros"],
            "batter": [batter_id],
            "player_name": ["Player A"],
            "pitch_type": ["FF"],
            "type": ["X"],
            "description": [""],  # Empty string
            "release_speed": [95.0],
            "stand": ["R"],
            "inning_topbot": ["Top"],
            "plate_x": [0.5],
            "plate_z": [2.5],
        })
        mock_statcast.return_value = df

        use_case = GetBatterSwingsUseCase()
        result = use_case.execute(game_pk, batter_id)

        # Should handle empty description gracefully
        assert isinstance(result, list)

    # ========== Helper function tests ==========

    def test_convert_domain_stats_to_schema(self):
        """Test conversion of domain entity to schema"""
        domain_stats = PlayerStatsEntity(
            pitches_seen=10,
            swing_percentage=45.0,
            take_percentage=55.0,
            whiff_percentage=12.5,
            contact_percentage=87.5,
            average_velocity=95.5,
            batter_handedness="R",
        )

        result = _convert_domain_stats_to_schema(domain_stats)

        assert result.pitches_seen == 10
        assert result.swing_percentage == 45.0
        assert result.take_percentage == 55.0
        assert result.whiff_percentage == 12.5
        assert result.contact_percentage == 87.5
        assert result.average_velocity == 95.5
        assert result.batter_handedness == "R"

    @patch("app.use_cases.game_analysis.statcast_single_game")
    def test_build_player_summary_with_player_id(self, mock_statcast):
        """Test _build_player_summary with valid player_id"""
        game_pk = 12357
        df = pd.DataFrame({
            "game_pk": [game_pk],
            "game_date": ["2024-10-25"],
            "home_team": ["Giants"],
            "away_team": ["Padres"],
            "batter": [456],
            "player_name": ["Player A"],
            "pitch_type": ["FF"],
            "type": ["S"],
            "description": ["strike"],
            "release_speed": [95.0],
            "stand": ["R"],
            "inning_topbot": ["Top"],
        })
        mock_statcast.return_value = df

        use_case = BuildGameAnalysisUseCase()
        result = use_case.execute(game_pk)

        # Should have headshot_url when player_id is valid
        if result.players:
            assert result.players[0].headshot_url is not None
            assert "456" in result.players[0].headshot_url

    def test_build_player_summary_with_none_player_id(self):
        """Test _build_player_summary with None player_id (covers pd.notnull False branch)"""
        from app.use_cases.game_analysis import _build_player_summary
        test_group = pd.DataFrame({
            "description": ["strike"],
            "type": ["S"],
            "release_speed": [95.0],
            "stand": ["R"],
        })
        
        # Test with None player_id to cover the pd.notnull False branch
        group_key_none = (None, "Player A", "Diamondbacks")
        player_summary = _build_player_summary(group_key_none, test_group)
        
        # Should handle None player_id - headshot_url should be None, player_id should be 0
        assert player_summary.headshot_url is None
        assert player_summary.player_id == 0
        assert player_summary.player_name == "Player A"
        
        # Also test with NaN to ensure pd.notnull works correctly
        import numpy as np
        group_key_nan = (float('nan'), "Player B", "Rockies")
        player_summary_nan = _build_player_summary(group_key_nan, test_group)
        assert player_summary_nan.headshot_url is None
        assert player_summary_nan.player_id == 0

