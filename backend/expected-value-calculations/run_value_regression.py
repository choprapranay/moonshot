from pybaseball import statcast
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json


def load_season_safe(year):
    """Load season data in monthly chunks to avoid parsing errors"""
    months = [
        ('03-30', '04-30'),
        ('05-01', '05-31'),
        ('06-01', '06-30'),
        ('07-01', '07-31'),
        ('08-01', '08-31'),
        ('09-01', '10-01')
    ]
    
    data_chunks = []
    for start_month, end_month in months:
        try:
            chunk = statcast(
                start_dt=f"{year}-{start_month}", 
                end_dt=f"{year}-{end_month}"
            )
            print(f"Loaded {year}-{start_month} to {year}-{end_month}: {len(chunk)} rows")
            data_chunks.append(chunk)
        except Exception as e:
            print(f"Error loading {year}-{start_month}: {e}")
            continue
    
    return pd.concat(data_chunks, ignore_index=True)

data_2023 = load_season_safe(2023)
data_2024 = load_season_safe(2024)

data = pd.concat([data_2023, data_2024], ignore_index=True)

data = data.sort_values(['game_pk', 'inning', 'inning_topbot', 'at_bat_number'])

data = data[data['events'].notna()]

def calc_remaining_runs(group): 
    group = group.copy()
    final_score = group['bat_score'].iloc[-1]
    group['runs_rest_of_inning'] = final_score - group['bat_score']
    return group

data = data.groupby(['game_pk', 'inning', 'inning_topbot']).apply(calc_remaining_runs).reset_index(drop=True)

X_outcomes = pd.get_dummies(data['events'], prefix="outcome")
X_outs = pd.get_dummies(data['outs_when_up'], prefix='outs')
X_bases = data[['on_1b', 'on_2b', 'on_3b']].notna().astype(int)
X_bases.columns = ['runner_1b', 'runner_2b', 'runner_3b']

X = pd.concat([X_outcomes, X_outs, X_bases], axis=1)
y = data['runs_rest_of_inning']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

model.predict(X_test)

score = model.score(X_test, y_test)
#Keep the scoring, train_test_split functionalities till the model achieves a reasonable accuracy.

print(f"R² Score: {score:.3f}")


run_values = {name.replace('outcome_', ''): round(coef, 3)
    for name, coef in zip(X.columns, model.coef_)
    if name.startswith('outcome_')
}


print(run_values)

with open('run_values.json', 'w') as f:
    json.dump(run_values, f, indent=2)