import sys
import os
import argparse
import pandas as pd

_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_dir)
sys.path.insert(0, os.path.abspath(os.path.join(_current_dir, '..')))

from domain.entities import TrainingConfig

from application.build_dataset_use_case import BuildDatasetUseCase
from application.train_use_case import TrainModelUseCase
from application.pred_use_case import InferenceUseCase

from infrastructure.data.pybaseball_pull import PybaseballDataAccess
from infrastructure.data.supabase_pull import SupabaseDataAccess
from infrastructure.data.preprocess_adapter import PreprocessorAdapter
from infrastructure.data.model_adapter import ModelAdapter
from infrastructure.storage.storage_adapter import ModelStorageAdapter

from interface_adapters.controllers.dataset_controller import DatasetController
from interface_adapters.controllers.train_controller import TrainController
from interface_adapters.controllers.inference_controller import InferenceController
from interface_adapters.presenters.dataset_presenter import DatasetPresenter
from interface_adapters.presenters.train_presenter import TrainPresenter
from interface_adapters.presenters.inference_presenter import InferencePresenter
from interface_adapters.presenters.view_model import ViewModel

from presentation.cli.cli_view import CLIView, BatchView


class CAEngine:
    def __init__(self, data_source: str = 'pybaseball', verbose: bool = True, model_type: str = 'standard'):
        self.view_model = ViewModel()
        
        if verbose:
            self.view = CLIView(self.view_model)
        else:
            self.view = BatchView(self.view_model, verbose=True)
        
        if data_source == 'supabase':
            self._data_access = SupabaseDataAccess()
        else:
            self._data_access = PybaseballDataAccess()
        
        self._preprocessor = PreprocessorAdapter()
        self._model_adapter = ModelAdapter(model_type=model_type)
        self._storage = ModelStorageAdapter()
        
        self._dataset_presenter = DatasetPresenter(self.view_model)
        self._train_presenter = TrainPresenter(self.view_model)
        self._inference_presenter = InferencePresenter(self.view_model)
        
        self._build_dataset_use_case = BuildDatasetUseCase(
            data_access=self._data_access,
            preprocessor=self._preprocessor,
            output_boundary=self._dataset_presenter
        )
        
        self._train_use_case = TrainModelUseCase(
            model_adapter=self._model_adapter,
            preprocessor=self._preprocessor,
            storage=self._storage,
            output_boundary=self._train_presenter
        )
        
        self._inference_use_case = InferenceUseCase(
            model_adapter=self._model_adapter,
            preprocessor=self._preprocessor,
            storage=self._storage,
            output_boundary=self._inference_presenter
        )
        
        self.dataset_controller = DatasetController(self._build_dataset_use_case)
        self.train_controller = TrainController(self._train_use_case)
        self.inference_controller = InferenceController(self._inference_use_case)
        
        self._current_dataset: pd.DataFrame | None = None
        self._raw_data: pd.DataFrame | None = None
    
    def fetch_raw_data(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        raw_data = self._data_access.fetch_data(start_date, end_date)
        self._raw_data = raw_data
        return raw_data
    
    def build_dataset(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        self.dataset_controller.build_dataset(
            start_date=start_date,
            end_date=end_date,
            data_source=self._data_access.get_source_name()
        )
        self._current_dataset = self.view_model.dataset_view_model.dataset
        self._raw_data = self.view_model.dataset_view_model.raw_data
        return self._current_dataset
    
    def train(
        self,
        dataset: pd.DataFrame | None = None,
        epochs: int = 20,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        save_path: str = 'batter_outcome_model.pth'
    ) -> bool:
        data = dataset if dataset is not None else self._raw_data
        
        if data is None:
            print("Error: No dataset available. Build or provide a dataset first.")
            return False
        
        self._train_presenter.set_total_epochs(epochs)
        
        self.train_controller.train_model(
            dataset=data,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            save_path=save_path
        )
        
        return self.view_model.training_result_view_model.success
    
    def inference(
        self,
        dataset: pd.DataFrame | None = None,
        model_path: str = 'batter_outcome_model.pth'
    ) -> list | None:
        data = dataset if dataset is not None else self._current_dataset
        
        if data is None:
            print("Error: No dataset available. Build or provide a dataset first.")
            return None
        
        self.inference_controller.run_inference(
            dataset=data,
            model_path=model_path
        )
        
        if self.view_model.inference_result_view_model.success:
            return self.view_model.inference_result_view_model.predictions
        return None


def run_cli(args):
    
    if args.mode == 'train':
        print(f"Training Configuration:")
        print(f"  Data Source: {args.data_source}")
        print(f"  Model Type: {args.model_type}")
        print(f"  Date Range: {args.start_date} to {args.end_date}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning Rate: {args.lr}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Save Path: {args.save_path}")
        
        engine = CAEngine(data_source=args.data_source, verbose=True, model_type=args.model_type)
        
        print("Building dataset...")
        dataset = engine.build_dataset(args.start_date, args.end_date)
        
        if dataset is not None and not dataset.empty:
            print("Training model...")
            success = engine.train(
                epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                save_path=args.save_path
            )
            
            if success:
                print(f" Training complete! Model saved to {args.save_path}")
            else:
                print("Training failed!")
        else:
            print("Failed to build dataset!")
    
    elif args.mode == 'inference':
        print(f"Inference Configuration:")
        print(f"  Model Path: {args.model_path}")
        print(f"  Dataset: {args.dataset if args.dataset else 'Will fetch new data'}")
        
        engine = CAEngine(data_source=args.data_source, verbose=True)
        
        if args.dataset:
            dataset = pd.read_csv(args.dataset)
        else:
            print("Fetching sample data...")
            dataset = engine.fetch_raw_data(args.start_date, args.end_date)
        
        if dataset is not None and not dataset.empty:
            print(f"  Raw data: {len(dataset):,} rows")
            print("Running inference...")
            sample = dataset.sample(min(100, len(dataset)))
            predictions = engine.inference(
                dataset=sample,
                model_path=args.model_path
            )
            
            if predictions:
                print(f"Inference complete! {len(predictions)} predictions made.")
            else:
                print("Inference failed!")
        else:
            print("No dataset available!")


def main():
    parser = argparse.ArgumentParser(
        description='Batter Outcome Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    python main.py --mode train --start-date 2023-03-30 --end-date 2023-10-01 --epochs 10
  
  Run inference:
    python main.py --mode inference --model-path model.pth --dataset data.csv
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'inference'],
        required=True,
        help='Mode to run: train or inference'
    )
    parser.add_argument(
        '--data-source',
        choices=['pybaseball', 'supabase'],
        default='pybaseball',
        help='Data source to use (default: pybaseball)'
    )
    parser.add_argument(
        '--model-type',
        choices=['standard', 'improved'],
        default='standard',
        help='Model architecture: standard (faster) or improved (more accurate, slower)'
    )
    parser.add_argument(
        '--start-date',
        default='2023-03-30',
        help='Start date for data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        default='2023-04-30',
        help='End date for data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--save-path',
        default='batter_outcome_model.pth',
        help='Path to save the model (default: batter_outcome_model.pth)'
    )
    parser.add_argument(
        '--model-path',
        default='batter_outcome_model.pth',
        help='Path to load the model for inference'
    )
    parser.add_argument(
        '--dataset',
        help='Path to CSV dataset for inference'
    )
    
    args = parser.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
