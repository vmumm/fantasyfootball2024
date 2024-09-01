from flask import Flask, request, jsonify
from ml_model import loaded_models, predict_scores, aggregate_season_totals

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    player_data = request.json['player_data']
    predictions = predict_scores(loaded_models, player_data)
    season_totals = aggregate_season_totals(predictions)
    return jsonify(season_totals)

@app.route('/best_pick', methods=['POST'])
def best_pick():
    current_draft = request.json['current_draft']
    available_players = request.json['available_players']
    # Implement logic to calculate value above replacement and determine best pick
    # This will depend on your specific scoring system and draft strategy
    best_pick = calculate_best_pick(current_draft, available_players)
    return jsonify(best_pick)

def calculate_best_pick(current_draft, available_players):
    # Implement your logic here
    pass

if __name__ == '__main__':
    app.run(debug=True)