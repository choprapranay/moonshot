import pybaseball
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, parent_dir)

from backend.services.dataset_loader_service import DatasetLoaderService
from backend.infrastructure.model_storage import ModelStorage



def swing_type(outcome_text):     
    out_text = outcome_text.lower()
    swing_pattern = [r'field_out', r'single', r'double', r'triple',
                      r'home_run', r'grounded_into_double_play', r'force_out',
                      r'sac_fly', r'field_error', r'fielders_choice', r'fielders_choice_out',
                      r'double_play', r'triple_play', r'swinging_strike', r'foul', r'foul_tip', r'swinging_strike_blocked']
    take_pattern = [r'ball', r'walk', r'hit_by_pitch', r'called_strike', r'called_strike', r'blocked_ball']

    for p in swing_pattern:
        if p in out_text:
            return 1
    for p in take_pattern:
        if p in out_text:
            return 0
    return None

class SuperDataSet(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    
    def __getitem__(self, index):
        row = self.df.loc[index]
        batter_id = int(row['batter_id'])
        pitch_type_id = int(row['pitch_type_id'])
        features = torch.tensor(row[['release_speed', 'release_pos_x', 'plate_x', 'plate_z', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 'batting_pattern_id', 'launch_speed_angle_id', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle']].values.astype(np.float32), dtype=torch.float32)
        y = int(row['outcome_id'])
        return batter_id, pitch_type_id, features, y
    
    def __len__(self):
        return len(self.df)


class SuperModel(nn.Module):
    def __init__(self, num_batters, num_pitch_types, num_input_data_types, output_dim, batter_embedding=15, pitch_embedding=5, hidden_dim=64):
        super().__init__()
        self.batter_embedding = nn.Embedding(num_batters, batter_embedding)
        self.pitch_embedding = nn.Embedding(num_pitch_types, pitch_embedding)
        self.linear1 = nn.Linear(num_input_data_types, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)

        self.fc1 = nn.Linear(batter_embedding + pitch_embedding + 16, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)
    
    def forward(self, batter_id, pitch_id, input_data):
        b = self.batter_embedding(batter_id)
        p = self.pitch_embedding(pitch_id) 
        i = self.linear1(input_data)
        i = self.relu(i)
        i = self.linear2(i)
        i = self.relu(i)
        i = self.linear3(i)
        i = self.relu(i)
        x = torch.cat([b, p, i], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def calculate_accuracy(model, data_loader, device):
    """Calculate accuracy of the model on a given dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for batter_id, pitch_type_id, features, y in data_loader:
            batter_id = batter_id.to(device).long()
            pitch_type_id = pitch_type_id.to(device).long()
            features = features.to(device)
            y = y.to(device)
            
            logits = model(batter_id, pitch_type_id, features)
            predictions = torch.argmax(logits, dim=1)
            
            correct += (predictions == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def decode_probable(probable_list, batter_enc, pitch_enc):
    decoded = []
    for confidence, outcome, batter_id, pitch_type_id in probable_list:
        decoded.append({
            'confidence': float(confidence),
            'outcome': outcome,
            'batter': batter_enc.inverse_transform([batter_id])[0],
            'pitch_type': pitch_enc.inverse_transform([pitch_type_id])[0]
        })
    return decoded

def main():
    parser = argparse.ArgumentParser(description='Train batter outcome prediction model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')  # Lower default LR
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--save_path', type=str, default='batter_outcome_model.pth', help='Path to save the model (default: batter_outcome_model.pth)')
    
    args = parser.parse_args()
    
    print(f"Training with parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save path: {args.save_path}")
    
    print("\nLoading data...")
    dataset = DatasetLoaderService().get_training_dataset() #THE IMPORT CHANGE
    #batter_info = pybaseball.statcast('2023-04-01', '2023-04-03')
    #batter_info_2024 = pybaseball.statcast('2024-03-28', '2024-09-29')
    #batter_info = pd.concat([batter_info, batter_info_2024], ignore_index=True)

    shortened_data = dataset[['batter', 'pitch_type', 'description', 'plate_x', 'plate_z', 'events', 'release_speed', 'release_pos_x', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'estimated_ba_using_speedangle', 'launch_speed_angle']]
    pruned_data = shortened_data
    print(f"Data shape: {pruned_data.shape}")
    
    both_na = pruned_data['hc_x'].isna() & pruned_data['hc_y'].isna()
    both_na_2 = pruned_data['launch_speed'].isna() & pruned_data['launch_angle'].isna()
    both_na_3 = pruned_data['launch_speed_angle'].isna() & pruned_data['estimated_ba_using_speedangle'].isna()

    pruned_data['outcome_text'] = pruned_data['events']
    pruned_data['outcome_text'] = pruned_data['description'].where(both_na, pruned_data['events'])

    pruned_data.loc[both_na, 'hc_x'] = 0
    pruned_data.loc[both_na, 'hc_y'] = 0
    pruned_data.loc[both_na_2, 'launch_speed'] = 0
    pruned_data.loc[both_na_2, 'launch_angle'] = 0
    pruned_data.loc[both_na_3, 'launch_speed_angle'] = 0
    pruned_data.loc[both_na_3, 'estimated_ba_using_speedangle'] = 0

    pruned_data = pruned_data.dropna(subset=['outcome_text'])
    
    pruned_data['batting_pattern'] = pruned_data['outcome_text'].apply(swing_type)

    pruned_data = pruned_data.dropna(subset=['batting_pattern'])
    
    feature_cols = ['release_speed', 'release_pos_x', 'plate_x', 'plate_z', 'launch_speed', 
                    'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 
                    'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle']
    
    print(f"\nRows before NaN removal: {len(pruned_data)}")
    print("NaN counts per column:")
    print(pruned_data[feature_cols].isna().sum())
    
    pruned_data = pruned_data.dropna(subset=feature_cols)
    print(f"Rows after NaN removal: {len(pruned_data)}")
    
    batter_enc = LabelEncoder() 
    pruned_data['batter_id'] = batter_enc.fit_transform(pruned_data['batter'])
    pitch_enc = LabelEncoder()
    pruned_data['pitch_type_id'] = pitch_enc.fit_transform(pruned_data['pitch_type'])
    outcome_enc = LabelEncoder()
    pruned_data['outcome_id'] = outcome_enc.fit_transform(pruned_data['outcome_text'])
    batter_pattern_enc = LabelEncoder()
    pruned_data['batting_pattern_id'] = batter_pattern_enc.fit_transform(pruned_data['batting_pattern'])
    launch_speed_angle_enc = LabelEncoder()
    pruned_data['launch_speed_angle_id'] = launch_speed_angle_enc.fit_transform(pruned_data['launch_speed_angle'])

    NUM_BATTERS = pruned_data['batter'].nunique()
    NUM_PITCHES = pruned_data['pitch_type_id'].nunique()
    NUM_OUTCOMES = pruned_data['outcome_id'].nunique()
    NUM_BATTER_PATTERNS = pruned_data['batting_pattern_id'].nunique()
    NUM_LAUNCH_SPEED_ANGLE = pruned_data['launch_speed_angle_id'].nunique()

    scaler = StandardScaler()
    pruned_data[feature_cols] = scaler.fit_transform(pruned_data[feature_cols].astype(float))
    
    print("\nChecking for inf/nan after scaling:")
    inf_count = np.isinf(pruned_data[feature_cols]).sum().sum()
    nan_count = np.isnan(pruned_data[feature_cols]).sum().sum()
    print(f"Inf values: {inf_count}, NaN values: {nan_count}")
    
    if inf_count > 0 or nan_count > 0:
        print("Removing rows with inf/nan values...")
        pruned_data = pruned_data.replace([np.inf, -np.inf], np.nan)
        pruned_data = pruned_data.dropna(subset=feature_cols)
        print(f"Rows after inf/nan removal: {len(pruned_data)}")
    
    print("\nOutcome distribution:")
    print(pruned_data['outcome_text'].value_counts())
    
    labels = list(outcome_enc.classes_)
    print(f"\nNumber of unique outcomes: {NUM_OUTCOMES}")
    print(f"Number of pitch types: {NUM_PITCHES}")
    print(f"Number of batter patterns: {NUM_BATTER_PATTERNS}")
    
    train_df, test_df = train_test_split(pruned_data, test_size=0.2, random_state=42)
    
    train_dataset = SuperDataSet(train_df)
    test_dataset = SuperDataSet(test_df)
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model_t = SuperModel(num_batters=NUM_BATTERS, num_pitch_types=NUM_PITCHES, num_input_data_types=20, output_dim=NUM_OUTCOMES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_t.parameters(), lr=args.lr)
    
    print(f"\nModel architecture:\n{model_t}")
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        model_t.train()
        epoch_loss = 0
        batch_count = 0
        for i, (batter_id, pitch_type_id, features, y) in enumerate(train_loader):
            batter_id = batter_id.to(device).long()
            pitch_type_id = pitch_type_id.to(device).long()
            features = features.to(device)
            y = y.to(device)
            
            
            logits = model_t(batter_id, pitch_type_id, features)
            
            
            loss_value = loss_fn(logits, y)
            
            
            optimizer.zero_grad()
            loss_value.backward()
            
            torch.nn.utils.clip_grad_norm_(model_t.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss_value.item()
            batch_count += 1
        
        if batch_count == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: No valid batches!")
            break
            
        avg_loss = epoch_loss / batch_count
        
        train_accuracy = calculate_accuracy(model_t, train_loader, device)
        test_accuracy = calculate_accuracy(model_t, test_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    
    print("\nFinal Evaluation:")
    final_train_accuracy = calculate_accuracy(model_t, train_loader, device)
    final_test_accuracy = calculate_accuracy(model_t, test_loader, device)
    print(f"Final Train Accuracy: {final_train_accuracy:.4f} ({final_train_accuracy*100:.2f}%)")
    print(f"Final Test Accuracy: {final_test_accuracy:.4f} ({final_test_accuracy*100:.2f}%)")
    
    print("\nGenerating predictions...")
    model_t.eval()
    all_predictions = []
    probable = []
    
    with torch.inference_mode():
        for i, (batter_id, pitch_type_id, features, y) in enumerate(test_loader):
            batter_id = batter_id.to(device).long()
            pitch_type_id = pitch_type_id.to(device).long()
            features = features.to(device)
            y = y.to(device)
            
            logits = model_t(batter_id, pitch_type_id, features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            for batch_idx, probs_row in enumerate(probs):
                pred_dict = {}
                max_value = 0
                outcome_probable = None
                
                for j in range(len(labels)):
                    if float(probs_row[j]) > max_value:
                        max_value = probs_row[j]
                        outcome_probable = labels[j]
                    pred_dict[labels[j]] = float(probs_row[j])
                
                all_predictions.append(pred_dict)
                probable.append((max_value, outcome_probable, batter_id[batch_idx].item(), pitch_type_id[batch_idx].item()))
    
    decoded_probable = decode_probable(probable, batter_enc, pitch_enc)
    
    print(f"\nTotal predictions made: {len(decoded_probable)}")
    print("\nSample decoded predictions:")
    for pred in decoded_probable[:5]:
        print(f"  Batter: {pred['batter']}, Pitch: {pred['pitch_type']}, Outcome: {pred['outcome']}, Confidence: {pred['confidence']:.4f}")

    save_dict = {
        'model_state_dict': model_t.state_dict(),
        'batter_encoder': batter_enc,
        'pitch_encoder': pitch_enc,
        'outcome_encoder': outcome_enc,
        'batter_pattern_encoder': batter_pattern_enc,
        'launch_speed_angle_encoder': launch_speed_angle_enc,
        'scaler': scaler,
        'num_batters': NUM_BATTERS,
        'num_pitches': NUM_PITCHES,
        'num_outcomes': NUM_OUTCOMES,
        'labels': labels
    }
    
    torch.save(save_dict, args.save_path)
    print(f"\nModel saved as '{args.save_path}'")
    store = ModelStorage()

    store.upload_model(
        file_path=args.save_path,
        dest_path="moonshot_v1/final_batter_outcome_model.pth"
    )


if __name__ == "__main__":
    main()