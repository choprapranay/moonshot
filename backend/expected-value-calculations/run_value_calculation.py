import pandas as pd
from sklearn.linear_model import LinearRegression
    
class RunValueCalculator: 

    def __init__(self):
        self.model = LinearRegression()
        
    def _validate_input(self, data: pd.DataFrame) -> None:
            required_cols = [
                'game_pk', 'inning', 'inning_topbot', 'at_bat_number',
                'events', 'bat_score', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b'
            ]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
            prepared = data.sort_values(
                ['game_pk', 'inning', 'inning_topbot', 'at_bat_number']
            ).copy()
            
            prepared = prepared[prepared['events'].notna()]
            
            if len(prepared) == 0:
                raise ValueError("No valid events found in data")
            
            return prepared

    def _calculate_runs_rest_of_inning(self, data: pd.DataFrame) -> pd.DataFrame:
            
            def calc_for_half_inning(group):
                group = group.copy()
                final_score = group['bat_score'].iloc[-1]
                group['runs_rest_of_inning'] = final_score - group['bat_score']
                return group
            
            return data.groupby(
                ['game_pk', 'inning', 'inning_topbot']
            ).apply(calc_for_half_inning).reset_index(drop=True)


    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:

            X_outcomes = pd.get_dummies(data['events'], prefix='outcome')

            X_outs = pd.get_dummies(data['outs_when_up'], prefix='outs')
            
            X_bases = data[['on_1b', 'on_2b', 'on_3b']].notna().astype(int)
            X_bases.columns = ['runner_1b', 'runner_2b', 'runner_3b']
        
            return pd.concat([X_outcomes, X_outs, X_bases], axis=1)

    def _extract_run_values(self, feature_columns: pd.Index) -> dict[str, float]:
            
            run_values = {}
            for name, coef in zip(feature_columns, self.model.coef_):
                if name.startswith('outcome_'):
                    outcome_name = name.replace('outcome_', '')
                    run_values[outcome_name] = round(float(coef), 3)
            
            return run_values

    def calculate(self, pitch_data: pd.DataFrame) -> dict[str, float]:

        self._validate_input(pitch_data)
        data = self._prepare_data(pitch_data)
        data = self._calculate_runs_rest_of_inning(data)

        X = self._build_features(data)
        y = data['runs_rest_of_inning']

        self.model.fit(X, y)

        return self._extract_run_values(X.columns)

