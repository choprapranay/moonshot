class LabelingService:

    @staticmethod
    def swing_type(outcome_text: str):
        if not isinstance(outcome_text, str):
            return None

        out_text = outcome_text.lower()

        swing_pattern = [
            'field_out', 'single', 'double', 'triple',
            'home_run', 'grounded_into_double_play', 'force_out',
            'sac_fly', 'field_error', 'fielders_choice', 'fielders_choice_out',
            'double_play', 'triple_play', 'swinging_strike', 'foul',
            'foul_tip', 'swinging_strike_blocked'
        ]

        take_pattern = [
            'ball', 'walk', 'hit_by_pitch', 'called_strike', 'blocked_ball'
        ]

        for p in swing_pattern:
            if p in out_text:
                return 1

        for p in take_pattern:
            if p in out_text:
                return 0

        return None