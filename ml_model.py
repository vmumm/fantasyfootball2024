import joblib
import pandas as pd
import os

def load_models():
    positions = ['QB', 'RB', 'WR', 'TE']
    target_columns = [
        'passing_yards', 'passing_tds', 'interceptions', 'passing_2pt_conversions',
        'rushing_yards', 'rushing_tds', 'rushing_2pt_conversions',
        'receptions', 'receiving_yards', 'receiving_tds', 'receiving_2pt_conversions',
        'special_teams_tds', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost'
    ]
    
    models = {}
    for position in positions:
        models[position] = {}
        for target in target_columns:
            filename = f"models/{position}_{target}_model.joblib"
            if os.path.exists(filename):
                models[position][target] = joblib.load(filename)
            else:
                print(f"Warning: Model file not found: {filename}")
                models[position][target] = None  # or some default model
    return models

def predict_scores(models, player_data):
    predictions = {}
    print(f"Unique positions in player_data: {player_data['position'].unique()}")
    for position, pos_models in models.items():
        print(f"Processing position: {position}")
        position_data = player_data[player_data['position'] == position]
        print(f"Number of rows for {position}: {len(position_data)}")
        if not position_data.empty:
            predictions[position] = {}
            for target, model in pos_models.items():
                print(f"Processing target: {target}")
                feature_columns = [col for col in position_data.columns if col not in ['position', 'player_id', 'player_name', 'season']]
                print(f"Number of feature columns: {len(feature_columns)}")
                X = position_data[feature_columns]
                if model is not None:
                    predictions[position][target] = model.predict(X)
                    print(f"Predictions made for {position} - {target}")
                else:
                    print(f"Model is None for {position} - {target}")
    return predictions

def aggregate_season_totals(predictions):
    season_totals = {}
    for position, pos_predictions in predictions.items():
        season_totals[position] = {}
        for target, values in pos_predictions.items():
            season_totals[position][target] = values.sum()
    return season_totals

# Load models when the module is imported
loaded_models = load_models()