from typing import Dict, Set

class EVCalculator: 
    
    SWING_OUTCOMES = {
        'home_run', 'triple', 'double', 'single', 'field_out',
        'grounded_into_double_play', 'strikeout', 'swinging_strike',
        'foul', 'force_out', 'sac_fly', 'field_error'
    }

    TAKE_OUTCOMES = {
        'ball', 'walk', 'hit_by_pitch', 'called_strike'
    }

    def __init__(self, run_values: Dict[str, float]):
        self.run_values = run_values
     
    def calculate_expected_value(self, probability_dict: Dict[str, float],outcome_set: Set[str]) -> float:
        
        total_prob = sum(
            probability_dict.get(outcome, 0)
            for outcome in outcome_set
        )
        
        if total_prob == 0:
            return 0.0
        
        ev = 0.0
        for outcome in outcome_set:
            if outcome in probability_dict and outcome in self.run_values:
                normalized_prob = probability_dict[outcome] / total_prob
                ev += normalized_prob * self.run_values[outcome]
        
        return ev
    
    def calculate_swing_vs_take(self, probability_dict: Dict[str, float]) -> Dict[str, float]:
    
        ev_swing = self.calculate_expected_value(
            probability_dict,
            self.SWING_OUTCOMES
        )
        
        ev_take = self.calculate_expected_value(
            probability_dict,
            self.TAKE_OUTCOMES
        )
        
        return {
            'ev_swing': ev_swing,
            'ev_take': ev_take,
            'ev_diff': ev_swing - ev_take
        }