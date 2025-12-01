from typing import Any, Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from domain.interfaces import ModelInterface
from domain.entities import TrainingConfig, ModelArtifacts
from infrastructure.data.torch_dataset import BatterDataset, InferenceDataset


class SuperModel(nn.Module):
    def __init__(self, num_batters: int, num_pitch_types: int, num_input_data_types: int = 20, output_dim: int = 10, batter_embedding_dim: int = 15, pitch_embedding_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        
        self.batter_embedding = nn.Embedding(num_batters, batter_embedding_dim)
        self.pitch_embedding = nn.Embedding(num_pitch_types, pitch_embedding_dim)
        
        self.linear1 = nn.Linear(num_input_data_types, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        
        combined_dim = batter_embedding_dim + pitch_embedding_dim + 16
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, batter_id: torch.Tensor, pitch_id: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
        b = self.batter_embedding(batter_id)
        p = self.pitch_embedding(pitch_id)
        
        i = self.relu(self.linear1(input_data))
        i = self.relu(self.linear2(i))
        i = self.relu(self.linear3(i))
        
        x = torch.cat([b, p, i], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ImprovedSuperModel(nn.Module):
    def __init__(self, num_batters: int, num_pitch_types: int, num_input_data_types: int = 20, output_dim: int = 10,
                 batter_embedding_dim: int = 32, pitch_embedding_dim: int = 16, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        
        self.batter_embedding = nn.Embedding(num_batters, batter_embedding_dim)
        self.pitch_embedding = nn.Embedding(num_pitch_types, pitch_embedding_dim)
        
        self.feature_net = nn.Sequential(
            nn.Linear(num_input_data_types, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        combined_dim = batter_embedding_dim + pitch_embedding_dim + 32
        
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Tanh(),
            nn.Linear(combined_dim, combined_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, batter_id: torch.Tensor, pitch_id: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
        b = self.batter_embedding(batter_id)
        p = self.pitch_embedding(pitch_id)
        f = self.feature_net(input_data)
        
        combined = torch.cat([b, p, f], dim=1)
        
        attention_weights = self.attention(combined)
        combined = combined * attention_weights
        
        output = self.classifier(combined)
        return output


class ModelAdapter(ModelInterface):
    def __init__(self, device: Optional[str] = None, model_type: str = 'standard'):
        if device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        self._model_type = model_type
    
    def create_model(self, num_batters: int, num_pitch_types: int, num_outcomes: int, device: Optional[str] = None) -> nn.Module:
        use_device = device or self._device
        
        if self._model_type == 'improved':
            model = ImprovedSuperModel(
                num_batters=num_batters,
                num_pitch_types=num_pitch_types,
                num_input_data_types=20,
                output_dim=num_outcomes
            ).to(use_device)
        else:
            model = SuperModel(
                num_batters=num_batters,
                num_pitch_types=num_pitch_types,
                num_input_data_types=20,
                output_dim=num_outcomes
            ).to(use_device)
        
        return model
    
    def train(self, model: nn.Module, train_data: pd.DataFrame, test_data: pd.DataFrame, config: TrainingConfig, progress_callback: Optional[Callable] = None) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
        train_dataset = BatterDataset(train_data)
        test_dataset = BatterDataset(test_data)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        training_history = []
        
        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batter_id, pitch_type_id, features, y in train_loader:
                batter_id = batter_id.to(self._device).long()
                pitch_type_id = pitch_type_id.to(self._device).long()
                features = features.to(self._device)
                y = y.to(self._device)
                
                logits = model(batter_id, pitch_type_id, features)
                loss = loss_fn(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            
            train_acc = self._calculate_accuracy(model, train_loader)
            test_acc = self._calculate_accuracy(model, test_loader)
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
            training_history.append(metrics)
            
            if progress_callback:
                progress_callback(metrics)
        
        return model.state_dict(), training_history
    
    def _calculate_accuracy(self, model: nn.Module, data_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        
        with torch.inference_mode():
            for batter_id, pitch_type_id, features, y in data_loader:
                batter_id = batter_id.to(self._device).long()
                pitch_type_id = pitch_type_id.to(self._device).long()
                features = features.to(self._device)
                y = y.to(self._device)
                
                logits = model(batter_id, pitch_type_id, features)
                predictions = torch.argmax(logits, dim=1)
                
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def predict(self, model: nn.Module, input_data: pd.DataFrame, artifacts: ModelArtifacts, device: Optional[str] = None) -> List[Dict[str, Any]]:
        use_device = device or self._device
        model = model.to(use_device)
        model.eval()
        
        dataset = InferenceDataset(input_data)
        labels = artifacts.labels
        
        results = []
        
        with torch.inference_mode():
            for idx in range(len(dataset)):
                batter_id, pitch_type_id, features, batter_name, pitch_type, actual_outcome = dataset[idx]
                
                batter_id_t = torch.tensor([batter_id], dtype=torch.long).to(use_device)
                pitch_type_id_t = torch.tensor([pitch_type_id], dtype=torch.long).to(use_device)
                features_t = features.unsqueeze(0).to(use_device)
                
                logits = model(batter_id_t, pitch_type_id_t, features_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                top_idx = int(np.argmax(probs))
                
                all_probs = {label: float(probs[i]) for i, label in enumerate(labels)}
                
                results.append({
                    'batter': str(batter_name),
                    'pitch_type': str(pitch_type),
                    'predicted_outcome': labels[top_idx],
                    'confidence': float(probs[top_idx]),
                    'all_probabilities': all_probs,
                    'actual_outcome': str(actual_outcome)
                })
        
        return results
    
    def load_model_state(self, model: nn.Module, state_dict: Dict[str, Any]) -> nn.Module:
        model.load_state_dict(state_dict)
        return model