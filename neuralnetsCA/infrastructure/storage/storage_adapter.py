import sys
import os
from typing import Optional, Any, Callable
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from domain.interfaces import ModelStorageInterface
from domain.entities import Model


class ModelStorageAdapter(ModelStorageInterface):
    def __init__(self, cloud_storage: Optional[Any] = None):
        self._cloud_storage = cloud_storage
    
    def save_model(self, model: Model, path: str) -> bool:
        try:
            save_dict = model.to_save_dict()
            torch.save(save_dict, path)
            return True
        except Exception as e:
            print(f"Error saving model to {path}: {e}")
            return False
    
    def load_model(self, path: str) -> Optional[Model]:
        try:
            if not os.path.exists(path):
                print(f"Model file not found: {path}")
                return None
            
            save_dict = torch.load(path, weights_only=False)
            model = Model.from_save_dict(save_dict, model_path=path)
            return model
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return None
    
    