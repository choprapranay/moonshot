import json
from pathlib import Path
from typing import Dict

class RunValueRepository: 
     
    def __init__(self, file_path: str = None):
    
        if file_path is None:
            # Default: expected-value-calculations/run_values.json
            backend_dir = Path(__file__).resolve().parent.parent
            file_path = backend_dir / "use_cases" / "run_values.json"
        
        self.file_path = Path(file_path)
        self._run_values = None

    def load_run_values(self) -> Dict[str, float]:
        with open(self.file_path, 'r') as f:
            self._run_values = json.load(f)
        
        return self._run_values
    
    def get_run_value(self, outcome: str) -> float:

        if self._run_values is None:
            self.load_run_values()
        
        return self._run_values.get(outcome, 0.0)
    
    