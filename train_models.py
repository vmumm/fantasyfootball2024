import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from data_preprocessing import load_and_preprocess_data
from ml_model import load_models, predict_scores
import os

def prepare_data(data, position, target_columns):
    position_data = data[data['position'] == position]

    # Select features (you may need to adjust this based on your dataset)
    feature_columns = [col for col in position_data.columns if col not in target_columns + ['position', 'player_id', 'player_name', 'season']]
    X = position_data[feature_columns]
    y = position_data[target_columns]
    return X, y

def train_models(X, y, position):
    # Convert object columns to category
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')

    models = {}
    for target in y.columns:
        print(f"Training model for {target}")
        model = xgb(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            enable_categorical=True  # Enable categorical feature support
        )
        model.fit(X, y[target])
        models[target] = model

    return models

def save_models(models, position):
    for target, model in models.items():
        filename = f"models/{position}_{target}_model.joblib"
        joblib.dump(model, filename)
        print(f"Saved model: {filename}")

def evaluate_2023_predictions(all_models, validation_data, target_columns):
    print("\nEvaluating 2023 predictions:")
    
    # Prepare the validation data
    feature_columns = [col for col in validation_data.columns if col not in ['position', 'player_id', 'player_name', 'season', 'draft_pick', 'draft_round', 'draft_team', 'fantasy_points_ppr', 'ht', 'merge_name', 'name', 'ppr_sh', 'twitter_username'] + target_columns and not col.endswith('_id')]
    X_val = validation_data[feature_columns]
    y_val = validation_data[target_columns]

    # Make predictions
    predictions = predict_scores(all_models, validation_data)
    print("Predictions after predict_scores:", predictions)
    
    print("Predictions structure:", predictions)
    print("Available positions in predictions:", list(predictions.keys()))
    for position in predictions:
        print(f"Available targets for {position}:", list(predictions[position].keys()))
    
    # Evaluate predictions
    results = {}
    detailed_results = {}
    for position in all_models.keys():  # Iterate over positions in models
        if position not in predictions:
            predictions[position] = {}
        
        results[position] = {}
        detailed_results[position] = {}
        position_mask = validation_data['position'] == position
        for target in all_models[position].keys():  # Iterate over targets for each position
            if target not in predictions[position]:
                print(f"Warning: No predictions for {position} - {target}")
                continue
            
            y_true = y_val[position_mask][target]
            y_pred = predictions[position][target]
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            results[position][target] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            detailed_results[position][target] = {
                'true_values': y_true.tolist(),
                'predicted_values': y_pred.tolist(),
                'errors': (y_true - y_pred).tolist()
            }
            
            print(f"\n{position} - {target} 2023 Prediction Evaluation:")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R2 Score: {r2:.2f}")
    
    return results, detailed_results

def main():
    positions = ['QB', 'RB', 'WR', 'TE']
    target_columns = [
        'passing_yards', 'passing_tds', 'interceptions', 'passing_2pt_conversions',
        'rushing_yards', 'rushing_tds', 'rushing_2pt_conversions',
        'receptions', 'receiving_yards', 'receiving_tds', 'receiving_2pt_conversions',
        'special_teams_tds', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost'
    ]

    all_models_exist = all(os.path.exists(f"models/{position}_{target}_model.joblib") 
                           for position in positions 
                           for target in target_columns)

    if all_models_exist:
        print("All models exist. Loading models and evaluating 2023 predictions...")
        all_models = load_models()
        _, validation_data = load_and_preprocess_data()
    else:
        print("Some models are missing. Creating new models...")
        train_data, validation_data = load_and_preprocess_data()
        
        all_models = {}
        for position in positions:
            print(f"\nTraining models for {position}")
            X, y = prepare_data(train_data, position, target_columns)
            models = train_models(X, y, position)
            save_models(models, position)
            all_models[position] = models

    # Evaluate 2023 predictions
    prediction_results, detailed_results = evaluate_2023_predictions(all_models, validation_data, target_columns)
    
    # Print the structure of detailed_results
    print("\nStructure of detailed_results:")
    for position in detailed_results:
        print(f"  {position}:")
        for target in detailed_results[position]:
            print(f"    {target}: {list(detailed_results[position][target].keys())}")
    
    # Save both results
    joblib.dump(prediction_results, "models/2023_prediction_results.joblib")
    joblib.dump(detailed_results, "models/2023_detailed_prediction_results.joblib")
    print("Saved 2023 prediction results and detailed results.")

if __name__ == "__main__":
    main()