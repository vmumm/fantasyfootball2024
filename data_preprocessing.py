import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nfl_data_py as nfl
import os

def load_and_preprocess_data():
    print("Creating merged data...")
    
    print("Loading seasonal data...")
    seasonal_data = nfl.import_seasonal_data(years=list(range(2005, 2024)), s_type='REG')
    
    print("Loading and merging IDs...")
    ids = nfl.import_ids()
    merged_data = pd.merge(seasonal_data, ids, left_on=['player_id'], right_on=['gsis_id'], how='left', suffixes=('', '_ids'))
    merged_data = merged_data[merged_data['position'].isin(['QB', 'WR', 'RB', 'TE'])]
    del seasonal_data, ids
    
    print("Loading and merging combine data...")
    combine_data = nfl.import_combine_data(years=list(range(2005, 2024)), positions=['QB', 'RB', 'WR', 'TE'])
    merged_data = pd.merge(merged_data, combine_data, on=['pfr_id'], how='left', suffixes=('', '_combine'))
    del combine_data
    
    print("Loading and merging advanced passing data...")
    advanced_passing_data = nfl.import_ngs_data(stat_type='passing', years=list(range(2005, 2024)))
    advanced_passing_data = advanced_passing_data[advanced_passing_data['season_type'] == 'REG']
    merged_data = pd.merge(merged_data, advanced_passing_data, left_on=['gsis_id', 'season'], right_on=['player_gsis_id', 'season'], how='left', suffixes=('', '_passing'))
    del advanced_passing_data
    
    print("Loading and merging advanced rushing data...")
    advanced_rushing_data = nfl.import_ngs_data(stat_type='rushing', years=list(range(2005, 2024)))
    advanced_rushing_data = advanced_rushing_data[advanced_rushing_data['season_type'] == 'REG']
    merged_data = pd.merge(merged_data, advanced_rushing_data, left_on=['gsis_id', 'season'], right_on=['player_gsis_id', 'season'], how='left', suffixes=('', '_rushing'))
    del advanced_rushing_data
    
    print("Loading and merging advanced receiving data...")
    advanced_receiving_data = nfl.import_ngs_data(stat_type='receiving', years=list(range(2005, 2024)))
    advanced_receiving_data = advanced_receiving_data[advanced_receiving_data['season_type'] == 'REG']
    merged_data = pd.merge(merged_data, advanced_receiving_data, left_on=['gsis_id', 'season'], right_on=['player_gsis_id', 'season'], how='left', suffixes=('', '_receiving'))
    del advanced_receiving_data

    # This code does the following:
    # 1. It creates a new DataFrame from 'merged_data' by selecting only certain columns:
    #    - The '~' operator negates the condition that follows it.
    #    - 'merged_data.columns.str.contains()' checks if column names contain specific suffixes.
    #    - The suffixes checked are '_ids', '_combine', '_passing', '_rushing', and '_receiving'.
    #    - Columns containing these suffixes are excluded from the new DataFrame.
    # 2. This effectively removes duplicate columns that were created during the merging process,
    #    keeping only the original columns and discarding the suffixed versions.
    merged_data = merged_data.loc[:, ~merged_data.columns.str.contains('_ids|_combine|_passing|_rushing|_receiving')]
    merged_data = merged_data.drop('yac_sh', axis=1, errors='ignore')

    merged_data = handle_missing_values_and_types(merged_data)

    merged_data = normalize_features(merged_data)

    train_data, validation_data = split_data(merged_data, year=2023)
    del merged_data

    print("Data preprocessing complete.")
    return train_data, validation_data

def handle_missing_values_and_types(data):
    # Identify object columns
    object_columns = data.select_dtypes(include=['object']).columns
    
    # Convert object columns to category
    for col in object_columns:
        data[col] = data[col].astype('category')
    
    return data

def normalize_features(data):
    scaler = StandardScaler()
    columns_to_exclude = [
        # General info
        'season', 'position', 'name', 'merge_name', 'team', 'birthdate', 'draft_year',
        'twitter_username', 'college', 'db_season', 'draft_team', 'player_name',
        'pos', 'school',

        # IDs
        'mfl_id', 'sportradar_id', 'fantasypros_id', 'gsis_id', 'pff_id', 'sleeper_id',
        'nfl_id', 'espn_id', 'yahoo_id', 'fleaflicker_id', 'cbs_id', 'pfr_id', 'cfbref_id',
        'rotowire_id', 'rotoworld_id', 'ktc_id', 'stats_id', 'stats_global_id',
        'fantasy_data_id', 'swish_id', 'cfb_id',

        # Advanced passing data
        'season_type_advanced_passing', 'player_display_name', 'player_position',
        'team_abbr', 'player_gsis_id', 'player_first_name', 'player_last_name',
        'player_short_name'
    ]
    numeric_columns = [col for col in data.select_dtypes(include=[np.number]).columns if col not in columns_to_exclude]
        
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def split_data(merged_data, year):
    # Split into training and validation sets
    train_data = merged_data[merged_data['season'] < year]
    validation_data = merged_data[merged_data['season'] >= year]
    return train_data, validation_data