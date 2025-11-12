import torch
import numpy as np
import pandas as pd
from modeltrain import SuperModel

def load_model_and_encoders(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model with saved parameters
    model = SuperModel(
        num_batters=checkpoint['num_batters'],
        num_pitch_types=checkpoint['num_pitches'],
        num_input_data_types=20,
        output_dim=checkpoint['num_outcomes']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return {
        'model': model,
        'batter_encoder': checkpoint['batter_encoder'],
        'pitch_encoder': checkpoint['pitch_encoder'],
        'outcome_encoder': checkpoint['outcome_encoder'],
        'batter_pattern_encoder': checkpoint['batter_pattern_encoder'],
        'launch_speed_angle_encoder': checkpoint['launch_speed_angle_encoder'],
        'scaler': checkpoint['scaler'],
        'labels': checkpoint['labels'],
        'device': device
    }


def predict_and_decode(model_data, batter_id_raw, pitch_type_raw, features_raw):
    model = model_data['model']
    device = model_data['device']
    
    # Encode inputs
    try:
        batter_id_encoded = model_data['batter_encoder'].transform([batter_id_raw])[0]
    except ValueError:
        print(f"Warning: Batter {batter_id_raw} not in training data, using 0")
        batter_id_encoded = 0
    
    try:
        pitch_type_encoded = model_data['pitch_encoder'].transform([pitch_type_raw])[0]
    except ValueError:
        print(f"Warning: Pitch type {pitch_type_raw} not in training data, using 0")
        pitch_type_encoded = 0
    
    # Prepare features
    if isinstance(features_raw, dict):
        feature_names = ['release_speed', 'release_pos_x', 'plate_x', 'plate_z', 'launch_speed', 
                        'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 
                        'batting_pattern_id', 'launch_speed_angle_id', 'hc_x', 'hc_y', 
                        'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle']
        features_array = np.array([features_raw[name] for name in feature_names])
    else:
        features_array = np.array(features_raw)
    
    # Scale continuous features (first 17 features, excluding batting_pattern_id and launch_speed_angle_id)
    features_to_scale = features_array[[0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,19]]
    features_scaled = features_array.copy()
    features_scaled[[0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,19]] = model_data['scaler'].transform(
        features_to_scale.reshape(1, -1)
    )[0]
    
    # Convert to tensors
    batter_tensor = torch.tensor([batter_id_encoded], dtype=torch.long).to(device)
    pitch_tensor = torch.tensor([pitch_type_encoded], dtype=torch.long).to(device)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.inference_mode():
        logits = model(batter_tensor, pitch_tensor, features_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Decode output
    predictions_with_probs = []
    for i, label in enumerate(model_data['labels']):
        predictions_with_probs.append({
            'outcome': label,
            'probability': float(probs[i])
        })
    
    # Sort by probability
    predictions_with_probs.sort(key=lambda x: x['probability'], reverse=True)
    
    # Get top prediction
    top_prediction = predictions_with_probs[0]
    
    return {
        'batter': batter_id_raw,
        'pitch_type': pitch_type_raw,
        'top_prediction': top_prediction['outcome'],
        'confidence': top_prediction['probability'],
        'all_predictions': predictions_with_probs
    }


def main():
    model_path = 'batter_outcome_model.pth'
    
    print("Loading model and encoders...")
    model_data = load_model_and_encoders(model_path)
    print(f"Model loaded successfully!")
    print(f"Available batters: {len(model_data['batter_encoder'].classes_)}")
    print(f"Available pitch types: {list(model_data['pitch_encoder'].classes_)}")
    print(f"Possible outcomes: {model_data['labels']}\n")
    
    # Example 1: Using a batter from training data
    print("=" * 60)
    print("Example 1: Prediction with encoded output")
    print("=" * 60)
    
    # Get a sample batter ID from the encoder
    sample_batter = model_data['batter_encoder'].classes_[0]
    
    example_features = np.array([
        95.0,    
        2.5,     
        0.5,     
        2.0,     
        90.0,    
        15.0,    
        94.5,    
        2200.0,  
        6.0,     
        1,       
        3,       
        125.0,   
        75.0,    
        -5.0,    
        -130.0,  
        -8.0,    
        10.0,    
        25.0,    
        -20.0,   
        0.5      
    ])
    
    result = predict_and_decode(model_data, sample_batter, 'FF', example_features)
    
    print(f"Batter ID: {result['batter']}")
    print(f"Pitch Type: {result['pitch_type']}")
    print(f"Top Prediction: {result['top_prediction']}")
    print(f"Confidence: {result['confidence']:.4f}\n")
    

if __name__ == "__main__":
    main()