from flask import Flask, request, jsonify
from flask_cors import CORS
import pybaseball
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  

@app.route('/api/pitch-data/<game_id>', methods=['GET'])
def get_pitch_data(game_id):
    """
    Fetch pitch-by-pitch data for a given STATCAST game ID
    """
    try:       
        pitch_data = pybaseball.statcast_single_game(int(game_id))
        
        if pitch_data.empty:
            return jsonify({
                'error': f'No data found for game ID {game_id}. Please check the game ID format.',
                'data': []
            }), 404
        
        available_columns = pitch_data.columns.tolist()
        columns_to_use = []
        
        column_mapping = {
            'game_pk': 'game_pk',
            'at_bat_number': 'at_bat_number',
            'pitch_number': 'pitch_number',
            'pitch_type': 'pitch_type',
            'pitch_name': 'pitch_name',
            'release_speed': 'release_speed',
            'plate_x': 'plate_x',
            'plate_z': 'plate_z',
            'zone': 'zone',
            'balls': 'balls',
            'strikes': 'strikes',
            'description': 'description',
            'events': 'events',
            'batter': 'batter',
            'pitcher': 'pitcher',
            'inning': 'inning',
            'inning_topbot': 'inning_topbot',
            'stand': 'stand',
            'p_throws': 'p_throws'
        }
        
        for col_name, col_key in column_mapping.items():
            if col_key in available_columns:
                columns_to_use.append(col_key)
        
        if not columns_to_use:
            columns_to_use = available_columns[:10]
        
        filtered_data = pitch_data[columns_to_use].copy()
        filtered_data = filtered_data.fillna('N/A')
        data_list = filtered_data.to_dict('records')
                        
        return jsonify({
            'success': True,
            'game_id': game_id,
            'total_pitches': len(data_list),
            'available_columns': available_columns,
            'data': data_list
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error fetching data: {str(e)}',
            'data': []
        }), 500

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify backend is running"""
    return jsonify({
        'message': 'Backend is running!',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Moonshot Pitch Data API'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
