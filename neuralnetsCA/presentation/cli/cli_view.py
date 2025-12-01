import pandas as pd
from typing import Optional, Callable

from interface_adapters.presenters.view_model import (
    ViewModel,
    DatasetViewModel,
    TrainingProgressViewModel,
    TrainingResultViewModel,
    InferenceResultViewModel
)


class CLIView:
    def __init__(self, view_model: ViewModel):
        self._view_model = view_model
        
        self._view_model.set_dataset_update_callback(self._on_dataset_update)
        self._view_model.set_training_progress_callback(self._on_training_progress)
        self._view_model.set_training_complete_callback(self._on_training_complete)
        self._view_model.set_inference_complete_callback(self._on_inference_complete)
    
    def _on_dataset_update(self, vm: DatasetViewModel) -> None:
        if vm.success:
            print(f"Successfully built dataset")
            print(f"  Samples: {vm.num_samples:,}")
            print(f"  Unique Batters: {vm.num_batters:,}")
            print(f"  Pitch Types: {vm.num_pitch_types}")
            print(f"  Outcome Classes: {vm.num_outcomes}")
            
            if vm.outcome_distribution:
                print("\nOutcome Distribution (top 10):")
                sorted_outcomes = sorted(
                    vm.outcome_distribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for outcome, count in sorted_outcomes:
                    print(f"  {outcome}: {count:,}")
        else:
            print(f"Failed to build dataset")
            print(f"  Error: {vm.error_message}")
    
    def _on_training_progress(self, vm: TrainingProgressViewModel) -> None:
        progress_pct = (vm.current_epoch / vm.total_epochs * 100) if vm.total_epochs > 0 else 0
        print(
            f"Epoch {vm.current_epoch}/{vm.total_epochs} "
            f"[{progress_pct:5.1f}%] "
            f"Loss: {vm.train_loss:.4f} | "
            f"Train Acc: {vm.train_accuracy:.4f} | "
            f"Test Acc: {vm.test_accuracy:.4f}"
        )
    
    def _on_training_complete(self, vm: TrainingResultViewModel) -> None:
        if vm.success:
            print(f"Model trained successfully")
            print(f"  Final Train Accuracy: {vm.final_train_accuracy:.4f} ({vm.final_train_accuracy*100:.2f}%)")
            print(f"  Final Test Accuracy: {vm.final_test_accuracy:.4f} ({vm.final_test_accuracy*100:.2f}%)")
            print(f"  Total Epochs: {vm.total_epochs}")
            print(f"  Model saved to: {vm.save_path}")
        else:
            print(f"Training failed")
            print(f"  Error: {vm.error_message}")
    
    def _on_inference_complete(self, vm: InferenceResultViewModel) -> None:
        print("\n" + "=" * 60)
        print("INFERENCE COMPLETE")
        print("=" * 60)
        
        if vm.success:
            print(f"Inference completed on {vm.num_predictions:,} samples")
            
            for i, pred in enumerate(vm.predictions[:10]):
                actual = pred.actual_outcome if pred.actual_outcome else "N/A"
                print(f"\n--- Prediction {i+1} ---")
                print(f"Batter: {pred.batter_name}")
                print(f"Pitch Type: {pred.pitch_type}")
                print(f"Predicted: {pred.predicted_outcome} (confidence: {pred.confidence:.4f})")
                print(f"Actual: {actual}")
                
                if pred.all_probabilities:
                    print("Outcome Probabilities:")
                    sorted_probs = sorted(
                        pred.all_probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    for outcome, prob in sorted_probs:
                        bar = "█" * int(prob * 20)
                        print(f"  {outcome:<25} {prob:6.2%} {bar}")
        else:
            print(f"Inference failed")
            print(f"  Error: {vm.error_message}")


class BatchView:
    def __init__(self, view_model: ViewModel, verbose: bool = True):
        self._view_model = view_model
        self._verbose = verbose
        
        self._view_model.set_dataset_update_callback(self._on_dataset_update)
        self._view_model.set_training_progress_callback(self._on_training_progress)
        self._view_model.set_training_complete_callback(self._on_training_complete)
        self._view_model.set_inference_complete_callback(self._on_inference_complete)
    
    def _on_dataset_update(self, vm: DatasetViewModel) -> None:
        if self._verbose:
            if vm.success:
                print(f"Dataset built: {vm.num_samples:,} samples, {vm.num_batters:,} batters")
            else:
                print(f"Dataset build failed: {vm.error_message}")
    
    def _on_training_progress(self, vm: TrainingProgressViewModel) -> None:
        if self._verbose:
            print(
                f"Epoch {vm.current_epoch}/{vm.total_epochs} - "
                f"Loss: {vm.train_loss:.4f}, Test Acc: {vm.test_accuracy:.4f}"
            )
    
    def _on_training_complete(self, vm: TrainingResultViewModel) -> None:
        if vm.success:
            print(f"Training complete - Test Accuracy: {vm.final_test_accuracy:.4f}")
            print(f"Model saved to: {vm.save_path}")
        else:
            print(f"Training failed: {vm.error_message}")
    
    def _on_inference_complete(self, vm: InferenceResultViewModel) -> None:
        if vm.success:
            print(f"Inference complete: {vm.num_predictions:,} predictions")
        else:
            print(f"Inference failed: {vm.error_message}")
