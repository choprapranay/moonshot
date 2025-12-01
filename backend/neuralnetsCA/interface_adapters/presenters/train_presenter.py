from domain.boundaries import (
    TrainModelOutputBoundary,
    TrainModelOutputData
)
from domain.entities import TrainingMetrics
from interface_adapters.presenters.view_model import (
    ViewModel, 
    TrainingProgressViewModel, 
    TrainingResultViewModel
)


class TrainPresenter(TrainModelOutputBoundary):
    def __init__(self, view_model: ViewModel, total_epochs: int = 0):
        self._view_model = view_model
        self._total_epochs = total_epochs
    
    def set_total_epochs(self, total_epochs: int) -> None:
        self._total_epochs = total_epochs
    
    def present_training_progress(self, metrics: TrainingMetrics) -> None:
        progress_vm = TrainingProgressViewModel(
            current_epoch=metrics.epoch,
            total_epochs=self._total_epochs,
            train_loss=metrics.train_loss,
            train_accuracy=metrics.train_accuracy,
            test_accuracy=metrics.test_accuracy
        )
        
        self._view_model.update_training_progress(progress_vm)
    
    def present_training_complete(self, output_data: TrainModelOutputData) -> None:
        history = [metrics.to_dict() for metrics in output_data.training_history]
        
        result_vm = TrainingResultViewModel(
            success=output_data.success,
            final_train_accuracy=output_data.final_train_accuracy,
            final_test_accuracy=output_data.final_test_accuracy,
            total_epochs=len(output_data.training_history),
            save_path=output_data.save_path,
            error_message=output_data.error_message,
            training_history=history
        )
        
        self._view_model.update_training_result(result_vm)
